"""Orange-expansion summary: untrained / SFT (3 training prompts) / RL arms per model.

Reads every per-rollout eval JSON under the roots in config `analysis.sources`
(sft/runs, rl/runs, baseline/runs: untrained models and the 27B mechanism check =
2A adapters under `nopersona`) plus, for the mechanism table, 2A's own val evals of
the same 27B adapters (persona / direct / stated) from `analysis.orange_2a_runs`.

Per rollout: honest compliance (cotcontrol.eval.grading.is_honest), raw strict
compliance, accuracy, reasoning chars, degenerate (< analysis.degen_chars),
compl_long (compliance among traces at least as long as the untrained median of
that model x condition x mode). Questions are keyed by (dataset, id, mode).
Conditions are inferred from each JSON's own system prompt (matched against
prompts.eval_block), never from file names. Cells carry 95% bootstrap intervals
over questions (x seeds where there are several); differences are paired on
questions (kaushik/core/stats.py).

Hit rule (PLAN.md, applied mechanically, modes pooled and per mode): a trained arm
beats BOTH untrained AND the same model's sft_standard arm by >= analysis.hit_points
honest compliance with the 95% CI of the difference excluding 0 (RL arms: untrained
only). A hit whose gain is not also present in compl_long is labelled length-driven.

    .venv/bin/python kaushik/run.py orange_expand/sft analyze [--boot 2000]
Outputs (results/): summary.md, cells.csv, differences.csv, hits.csv, mechanism_cells.csv, rollouts.csv.gz.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import stats as st
from kaushik.core.config import model_key_of, resolve, results_dir
from kaushik.core.io import load_callable
from kaushik.core.results import (
    infer_condition,
    load_rollouts,
    parse_eval_json,
    run_meta,
)
from kaushik.core.stats import POOLED

EXP = "orange_expand/sft"
MECH_CONDITIONS = ["persona", "nopersona", "direct", "standard"]
ARM_ORDER = ["untrained", "sft_eugene", "sft_direct", "sft_standard", "rl_eugene", "rl_direct", "rl_standard",
             "orange", "orange_nopersona"]
ROLLOUT_COLUMNS = ["source", "run", "block", "arm", "seed", "model", "condition", "mode", "step", "lora", "qid",
                   "task", "compliance", "honest", "correct", "chars", "truncated"]
TIMESTAMP_RE = re.compile(r"-\d{8}_\d{6}$")
RUN_RE = re.compile(r"^(?:oex-)?(?P<arm>.+?)-(?P<model>[A-Za-z0-9_.]+)-seed(?P<seed>\d+)(?P<rest>.*)$")


# ----------------------------------------------------------------------------- loading


def _arm_seed(source: str, run: str, meta: dict, lora_path: str | None) -> tuple[str, object]:
    if meta.get("arm"):
        return meta["arm"], meta.get("seed", meta.get("train", {}).get("seed", 0))
    # baseline/: the run dir is the model, so the adapter's volume dir names the arm; elsewhere the run dir.
    ref = Path(lora_path).parts[2] if source == "baseline" and lora_path and len(Path(lora_path).parts) > 2 else run
    m = RUN_RE.match(TIMESTAMP_RE.sub("", ref))
    if source == "baseline":
        if not lora_path:
            return "untrained", 0
        return (m["arm"], int(m["seed"])) if m else ("adapter", run)
    if m:
        arm = m["arm"].replace("-", "_")
        if source == "rl" and not arm.startswith("rl"):
            arm = f"rl_{arm}"
        # a run-name suffix after the seed (e.g. "-lr5e-5") is a distinct arm, not another seed
        if source != "sft" and m["rest"].strip("-_"):
            arm = f"{arm}_{m['rest'].strip('-_')}"
        return arm, int(m["seed"])
    return (f"rl_{run}" if source == "rl" else run), 0


def make_extra(prompts, conditions: list[str]):
    """Row hook for core.results: arm / seed / model key / condition per rollout."""
    def extra(row: dict, cfg: dict, path: Path) -> dict:
        arm, seed = _arm_seed(row["source"], row["run"], run_meta(path.parents[2]), row["lora"])
        cond = infer_condition(row["system_prompt"], row["state_requirement"], row["mode"], row["base_model"],
                               prompts, conditions, block=row["block"])
        if cond == "other":
            cond = "persona" if "EUGENE" in row["system_prompt"] else ("standard" if row["block"].startswith("stated_") else cond)
        return {"arm": arm, "seed": seed, "condition": cond,
                "model": model_key_of(row["base_model"]) or row["base_model"].split("/")[-1].lower()}
    return extra


def load_2a_mech(orange_runs: Path, mech_model: str, arms: list[str], seeds: list[int], modes: list[str],
                 extra) -> pd.DataFrame:
    """2A's val evals (persona/direct/stated x the screen modes) of the same 27B adapters:
    checkpoint-50 of orange / orange_nopersona seeds, and checkpoint-0 of the orange runs
    as `untrained` (zero-init adapter)."""
    rows = []
    for arm in arms:
        for seed in seeds:
            run_dir = orange_runs / f"{arm}-{mech_model}-seed{seed}"
            for cond in ("persona", "direct", "stated"):
                for mode in modes:
                    for step in ((50, 0) if arm == arms[0] else (50,)):
                        f = run_dir / "eval" / f"{cond}_{mode}" / f"checkpoint-{step}.json"
                        if not f.exists():
                            continue
                        for r in parse_eval_json(f, "orange2a", extra):
                            r["arm"] = "untrained" if step == 0 else arm
                            r["seed"] = seed
                            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics

CELL_KEYS = ["model", "arm", "condition", "mode"]


def add_long(df: pd.DataFrame) -> pd.DataFrame:
    return st.add_long(df, st.untrained_bars(df[df.arm == "untrained"]))


def cell_stats(g: pd.DataFrame, rng, B: int, degen_chars: int) -> dict:
    out = {"n": len(g), "n_seeds": g.seed.nunique(), "step": ",".join(sorted(map(str, g.step.unique()))),
           "chars": g.chars.mean(), "degen": (g.chars < degen_chars).mean(), "truncated": g.truncated.mean(),
           "n_long": int(g.long.sum()), "bar": g.bar.iloc[0]}
    return {**out, **st.cell_metrics(g, rng, B)}


def summarize_cells(df: pd.DataFrame, rng, B: int, degen_chars: int) -> pd.DataFrame:
    return pd.DataFrame([{**dict(zip(CELL_KEYS, k)), **cell_stats(g, rng, B, degen_chars)}
                         for k, g in df.groupby(CELL_KEYS, sort=False)])


def differences(df: pd.DataFrame, rng, B: int, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for (model, cond, mode), g in df.groupby(["model", "condition", "mode"], sort=False):
        for a, b in pairs:
            xa, xb = g[g.arm == a], g[g.arm == b]
            if xa.empty or xb.empty:
                continue
            rows.append({"model": model, "condition": cond, "mode": mode, "a": a, "b": b, "n_a": len(xa),
                         "n_b": len(xb), **st.paired_boot(xa, xb, rng, B)})
    return pd.DataFrame(rows)


def hits(diffs: pd.DataFrame, arms: list[str], points: float) -> pd.DataFrame:
    rows = []
    for arm in arms:
        if arm in ("untrained", "sft_standard"):
            continue
        comparators = ["untrained"] if arm.startswith("rl") else ["untrained", "sft_standard"]
        d = diffs[diffs.a == arm]
        for (model, cond, mode), g in d.groupby(["model", "condition", "mode"], sort=False):
            parts = {b: g[g.b == b] for b in comparators}
            if any(p.empty for p in parts.values()):
                continue
            if all(st.passes_hit(p.iloc[0], "honest", points) for p in parts.values()):
                long_ok = all(st.passes_hit(p.iloc[0], "compl_long", points) for p in parts.values())
                rows.append({"model": model, "arm": arm, "condition": cond, "mode": mode,
                             **{f"d_{b}": f"{st.pct(p.iloc[0].honest)} [{st.pct(p.iloc[0].honest_lo)}, {st.pct(p.iloc[0].honest_hi)}]"
                                for b, p in parts.items()},
                             **{f"dlong_{b}": f"{st.pct(p.iloc[0].compl_long)} [{st.pct(p.iloc[0].compl_long_lo)}, {st.pct(p.iloc[0].compl_long_hi)}]"
                                for b, p in parts.items()},
                             "label": "" if long_ok else "length-driven"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- reporting


def arm_key(a: str) -> tuple:
    return (ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER), a)


def _cell(cells: pd.DataFrame, model, arm, cond, mode, metric="honest") -> str:
    r = cells[(cells.model == model) & (cells.arm == arm) & (cells.condition == cond) & (cells["mode"] == mode)]
    return st.pct_ci(r.iloc[0], metric) if len(r) else "-"


def write_summary(cells: pd.DataFrame, diffs: pd.DataFrame, hit_df: pd.DataFrame, mech: pd.DataFrame | None,
                  out: Path, models: list[str], conditions: list[str], modes: list[str], mech_model: str) -> str:
    intro = (f"Val split (200 q), T=1, 16k tokens, reasoning on, modes {' + '.join(modes)}. "
             "Cells: honest compliance % [95% bootstrap over questions]; `raw` = strict compliance, "
             "`compl_long` = compliance among traces at least as long as the untrained median (same model, "
             "condition, mode), `acc` = accuracy, `chars` = mean reasoning length, `degen` = traces < 200 chars. "
             "SFT arms at step 100 (one seed); RL arms at their final checkpoint; untrained = base model.\n")
    parts = ["# Orange expansion — stage 1 summary (SFT + RL vs untrained)\n", intro]
    for model in models:
        sub = cells[cells.model == model]
        if sub.empty:
            continue
        arms = sorted(sub.arm.unique(), key=arm_key)
        parts.append(f"\n## {model} — honest compliance, modes pooled\n")
        rows = [[arm, *[_cell(sub, model, arm, c, POOLED) for c in conditions]] for arm in arms]
        parts.append(st.md_table(["arm", *conditions], rows))
        for mode in modes:
            parts.append(f"\nPer mode — {mode}:\n")
            rows = [[arm, *[_cell(sub, model, arm, c, mode) for c in conditions]] for arm in arms]
            parts.append(st.md_table(["arm", *conditions], rows))
        parts.append(f"\nDetail ({model}, modes pooled):\n")
        rows = []
        for cond in conditions:
            for arm in arms:
                r = sub[(sub.arm == arm) & (sub.condition == cond) & (sub["mode"] == POOLED)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([cond, arm, r.step, st.pct_ci(r, "honest"), st.pct_ci(r, "compliance"),
                             f"{st.pct_ci(r, 'compl_long')} (n={r.n_long})", st.pct_ci(r, "correct"),
                             f"{r.chars:.0f}", st.pct(r.degen), st.pct(r.truncated), str(r.n)])
        parts.append(st.md_table(["condition", "arm", "step", "honest", "raw", "compl_long", "acc", "chars",
                                  "degen", "trunc", "n"], rows))
    d = diffs[diffs["mode"] == POOLED]
    if not d.empty:
        parts.append("\n## Differences vs comparators (modes pooled; paired bootstrap over questions)\n")
        rows = [[r.model, r.condition, f"{r.a} − {r.b}", st.pct_ci(r, "honest"), st.pct_ci(r, "compliance"),
                 st.pct_ci(r, "compl_long"), st.pct_ci(r, "correct")]
                for _, r in d.sort_values(["model", "condition", "a", "b"]).iterrows()]
        parts.append(st.md_table(["model", "condition", "difference", "Δ honest", "Δ raw", "Δ compl_long", "Δ acc"], rows))
    parts.append("\n## Hits (PLAN rule: ≥ 2 pts honest over untrained AND sft_standard, CI excluding 0; RL vs untrained only)\n")
    if hit_df is None or hit_df.empty:
        parts.append("No hits.\n")
    else:
        parts.append(st.md_table(list(hit_df.columns), hit_df.astype(str).values.tolist()))
        parts.append("\n`length-driven` = the gain is not present in compl_long against every comparator.\n")
    if mech is not None and not mech.empty:
        parts.append("\n## Mechanism check — Qwen3.6-27B, 2A adapters (seeds 0–2), val, modes pooled and per mode\n")
        parts.append("`nopersona` from the baseline agent's evals of the same adapters; persona / direct / standard "
                     "from 2A's own val evals (checkpoint-50; untrained = checkpoint-0). Honest compliance % "
                     "[95% bootstrap over questions × seeds].\n")
        for mode in [POOLED, *modes]:
            parts.append(f"\n{mode}:\n")
            arms = sorted(mech.arm.unique(), key=arm_key)
            rows = [[arm, *[_cell(mech, mech_model, arm, c, mode) for c in MECH_CONDITIONS],
                     *[_cell(mech, mech_model, arm, c, mode, "compl_long") for c in ("persona", "nopersona")]]
                    for arm in arms]
            parts.append(st.md_table(["arm", *MECH_CONDITIONS, "compl_long persona", "compl_long nopersona"], rows))
    text = "\n".join(parts) + "\n"
    out.write_text(text)
    return text


# ----------------------------------------------------------------------------- main


def main(cfg: dict, argv: list[str] | None = None) -> None:
    an = cfg["analysis"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(results_dir(cfg)))
    ap.add_argument("--boot", type=int, default=int(an["bootstrap_draws"]))
    ap.add_argument("--seed", type=int, default=int(an["bootstrap_seed"]))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    grid = cfg["eval"]["grid"]
    conditions, modes = list(grid["conditions"]), list(grid["modes"])
    extra = make_extra(load_callable(grid["prompts"]), conditions)
    sources = {k: ROOT / v for k, v in an["sources"].items()}
    df = load_rollouts(sources, out_dir / ".cache", extra=extra)
    if df.empty:
        raise SystemExit("no eval JSONs found")
    df = df[df.condition.isin(conditions) & df["mode"].isin(modes)]
    mech_model = an["mech_model"]
    mech_src = df[df.model == mech_model]          # the mechanism check is reported separately
    df = df[df.model != mech_model]
    print(f"{len(df):,} rollouts; models {sorted(df.model.unique())}; arms {sorted(df.arm.unique())}")
    df[ROLLOUT_COLUMNS].to_csv(out_dir / "rollouts.csv.gz", index=False)

    degen = int(an["degen_chars"])
    full = st.with_pooled(add_long(df))
    cells = summarize_cells(full, rng, args.boot, degen)
    cells.to_csv(out_dir / "cells.csv", index=False)
    arms = sorted(full.arm.unique(), key=arm_key)
    pairs = [(a, b) for a in arms for b in ("untrained", "sft_standard") if a not in ("untrained", "sft_standard") and a != b]
    diffs = differences(full, rng, args.boot, pairs)
    diffs.to_csv(out_dir / "differences.csv", index=False)
    hit_df = hits(diffs, arms, float(an["hit_points"]))
    hit_df.to_csv(out_dir / "hits.csv", index=False)

    mech = None
    m2a = load_2a_mech(ROOT / an["orange_2a_runs"], mech_model, list(an["mech_arms"]), list(an["mech_seeds"]),
                       modes, extra)
    mech_all = pd.concat([mech_src, m2a], ignore_index=True) if len(mech_src) or len(m2a) else pd.DataFrame()
    if len(mech_all):
        mech_all = mech_all[mech_all.condition.isin(MECH_CONDITIONS)]
        mech = summarize_cells(st.with_pooled(add_long(mech_all)), rng, args.boot, degen)
        mech.to_csv(out_dir / "mechanism_cells.csv", index=False)
        print(f"mechanism check: {len(mech_all):,} rollouts, arms {sorted(mech_all.arm.unique())}, "
              f"conditions {sorted(mech_all.condition.unique())}")

    models = [m for m in an["model_order"] if m in set(cells.model)]
    models += sorted(set(cells.model) - set(models))
    text = write_summary(cells, diffs, hit_df, mech, out_dir / "summary.md", models, conditions, modes, mech_model)
    print(text)


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
