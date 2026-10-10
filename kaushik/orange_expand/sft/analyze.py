"""Orange-expansion summary: untrained / SFT (3 training prompts) / RL arms per model.

Reads every per-rollout eval JSON under
    kaushik/orange_expand/sft/runs/<run>/eval/<block>/checkpoint-<step>.json     (this folder)
    kaushik/orange_expand/rl/runs/<run>/eval/<block>/...                         (RL agent)
    kaushik/orange_expand/baseline/runs/<model>/eval/<condition>_<mode>/...      (baseline agent:
        untrained models, and the 27B mechanism check = 2A adapters under `nopersona`)
plus, for the mechanism table, 2A's own val evals of the same 27B adapters (persona / direct /
stated conditions) from kaushik/orange/runs/.

Per rollout: honest compliance (cotcontrol.eval.grading.is_honest), raw strict compliance,
accuracy, reasoning chars, degenerate (< 200 chars), compl_long (compliance among traces at
least as long as the untrained median of that model x condition x mode). Questions are keyed by
(dataset, id, mode): ids repeat across datasets. Conditions are inferred from each JSON's own
system prompt (matched against prompts.eval_block), never from file names. Cells carry 95%
bootstrap intervals over questions (x seeds where there are several); differences are
paired on questions.

Hit rule (PLAN.md, applied mechanically, modes pooled and per mode): a trained arm beats
BOTH untrained AND the same model's sft_standard arm by >= 2 points honest compliance with
the 95% CI of the difference excluding 0 (RL arms: untrained only). A hit whose gain is not
also present in compl_long (>= 2 points, CI excluding 0) is labelled length-driven.

    .venv/bin/python kaushik/run.py orange_expand/sft analyze [--boot 2000]
Outputs (results/): summary.md, cells.csv, differences.csv, hits.csv, rollouts.csv.gz.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import common
from common import EXPAND, MODEL_KEYS, ORANGE, RESULTS_DIR, RL_RUNS_DIR, RUNS_DIR

from cotcontrol.eval.grading import is_honest

oa = common.orange_mod("analyze")        # boot_means, ci, _matrix, md_table, pct, pct_ci
P = common.oex_prompts
BASELINE_RUNS = EXPAND / "baseline" / "runs"
SOURCES = {"sft": RUNS_DIR, "rl": RL_RUNS_DIR, "baseline": BASELINE_RUNS}
MODES = list(P.SCREEN_MODES)
CONDITIONS = list(P.EVAL_CONDITIONS)
MECH_CONDITIONS = ["persona", "nopersona", "direct", "standard"]
POOLED = "(pooled)"
DEGEN_CHARS = 200
HIT_POINTS = 0.02
ARM_ORDER = ["untrained", "sft_eugene", "sft_direct", "sft_standard", "rl_eugene", "rl_direct", "rl_standard",
             "orange", "orange_nopersona"]


# ----------------------------------------------------------------------------- loading

def _condition(cfg: dict, mode: str, block: str) -> str:
    sp = (cfg.get("system_prompt") or "").strip()
    model = cfg["model"]
    if not sp:
        return "standard" if cfg.get("state_requirement", True) else "bare"
    for cond in ("persona", "nopersona", "direct"):
        if sp == (P.eval_block(cond, mode, model)["system_prompt"] or "").strip():
            return cond
    if "EUGENE" in sp:
        return "persona"
    m = re.match(r"^(persona|nopersona|direct|standard|stated)_", block)
    return {"stated": "standard"}.get(m.group(1), m.group(1)) if m else "other"


def _run_meta(run_dir: Path) -> dict:
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        try:
            return yaml.safe_load(cfg.read_text()).get("meta", {}) or {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _arm_seed(source: str, run: str, meta: dict, lora_path: str | None) -> tuple[str, object]:
    if meta.get("arm"):
        return meta["arm"], meta.get("seed", meta.get("train", {}).get("seed", 0))
    # baseline/: the run dir is the model, so the adapter's volume dir names the arm; elsewhere the run dir.
    ref = Path(lora_path).parts[2] if source == "baseline" and lora_path and len(Path(lora_path).parts) > 2 else run
    ref = re.sub(r"-\d{8}_\d{6}$", "", ref)  # Modal run-name timestamp
    m = re.match(r"^(?:oex-)?(?P<arm>.+?)-(?P<model>[A-Za-z0-9_.]+)-seed(?P<seed>\d+)(?P<rest>.*)$", ref)
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


def parse_file(path: Path, source: str) -> list[dict]:
    run, block = path.parents[2].name, path.parent.name
    data = json.loads(path.read_text())
    cfg = data["config"]
    lora = (cfg.get("backend_info") or {}).get("lora_path")
    arm, seed = _arm_seed(source, run, _run_meta(path.parents[2]), lora)
    step = path.stem.replace("checkpoint-", "")
    model = MODEL_KEYS.get(cfg["model"], cfg["model"].split("/")[-1].lower())
    rows = []
    for rec in data["results"]:
        mode = rec["mode"]
        cond = _condition(cfg, mode, block)
        for s in rec["samples"]:
            if s["error"]:
                continue
            text = s.get("reasoning_text_graded") or ""
            rows.append({"source": source, "run": run, "block": block, "arm": arm, "seed": seed, "model": model,
                         "condition": cond, "mode": mode, "step": step, "lora": lora,
                         "qid": f"{rec['dataset']}:{rec['id']}", "task": f"{rec['dataset']}:{rec['id']}|{mode}",
                         "compliance": s["compliance"] == 1, "honest": is_honest(s),
                         "correct": s["correct"] is True, "chars": len(text),
                         "truncated": s.get("finish_reason") == "length"})
    return rows


def load_rollouts(sources: dict[str, Path], cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, root in sources.items():
        if not root.is_dir():
            print(f"[load] no {source} runs at {root}")
            continue
        # baseline/runs/mech_qwen36_27b/<arm>-seed<k>/eval/... is one level deeper
        for path in sorted([*root.glob("*/eval/*/*.json"), *root.glob("*/*/eval/*/*.json")]):
            if "smoke" in str(path.relative_to(root)):
                continue
            cache = cache_dir / f"v1__{source}__{'__'.join(path.relative_to(root).parts[:-3])}__{path.parent.name}__{path.stem}.pkl"
            mtime = path.stat().st_mtime
            if cache.exists():
                c = pickle.loads(cache.read_bytes())
                if c["mtime"] == mtime:
                    rows += c["rows"]
                    continue
            try:
                parsed = parse_file(path, source)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[load] skipping {path}: {type(e).__name__}: {e}")
                continue
            cache.write_bytes(pickle.dumps({"mtime": mtime, "rows": parsed}))
            rows += parsed
    return pd.DataFrame(rows)


def load_2a_mech(df_untrained_bars_src: pd.DataFrame | None = None) -> pd.DataFrame:
    """2A's val evals (persona/direct/stated x the 2 screen modes) of the same 27B adapters:
    checkpoint-50 of orange / orange_nopersona seeds 0-2, and checkpoint-0 of orange seeds 0-2
    as `untrained` (zero-init adapter)."""
    rows = []
    for arm in ("orange", "orange_nopersona"):
        for seed in (0, 1, 2):
            run_dir = ORANGE / "runs" / f"{arm}-qwen36_27b-seed{seed}"
            for cond in ("persona", "direct", "stated"):
                for mode in MODES:
                    for step in ((50, 0) if arm == "orange" else (50,)):
                        f = run_dir / "eval" / f"{cond}_{mode}" / f"checkpoint-{step}.json"
                        if not f.exists():
                            continue
                        for r in parse_file(f, "orange2a"):
                            r["arm"] = "untrained" if step == 0 else arm
                            r["seed"] = seed
                            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics

def add_long(df: pd.DataFrame) -> pd.DataFrame:
    u = df[df.arm == "untrained"]
    bars = u.groupby(["model", "condition", "mode"]).chars.median().to_dict()
    bars2 = u.groupby(["model", "condition"]).chars.median().to_dict()
    bar = [bars.get((m, c, mo), bars2.get((m, c), np.nan)) for m, c, mo in zip(df.model, df.condition, df["mode"])]
    df = df.assign(bar=bar)
    df["long"] = df.chars >= df.bar
    df["compl_long"] = np.where(df.long, df.compliance.astype(float), np.nan)
    return df


def with_pooled(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.assign(mode=POOLED)], ignore_index=True)


CELL_KEYS = ["model", "arm", "condition", "mode"]
METRICS = ("honest", "compliance", "compl_long", "correct")


def cell_stats(g: pd.DataFrame, rng, B: int) -> dict:
    out = {"n": len(g), "n_seeds": g.seed.nunique(), "step": ",".join(sorted(map(str, g.step.unique()))),
           "chars": g.chars.mean(), "degen": (g.chars < DEGEN_CHARS).mean(), "truncated": g.truncated.mean(),
           "n_long": int(g.long.sum()), "bar": g.bar.iloc[0]}
    for metric in METRICS:
        M = oa._matrix(g, metric)
        with np.errstate(all="ignore"):
            out[metric] = float(np.nanmean(M)) if np.isfinite(M).any() else np.nan
        out[f"{metric}_lo"], out[f"{metric}_hi"] = oa.ci(oa.boot_means(M, rng, B))
    return out


def summarize_cells(df: pd.DataFrame, rng, B: int) -> pd.DataFrame:
    return pd.DataFrame([{**dict(zip(CELL_KEYS, k)), **cell_stats(g, rng, B)}
                         for k, g in df.groupby(CELL_KEYS, sort=False)])


def differences(df: pd.DataFrame, rng, B: int, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for (model, cond, mode), g in df.groupby(["model", "condition", "mode"], sort=False):
        for a, b in pairs:
            xa, xb = g[g.arm == a], g[g.arm == b]
            if xa.empty or xb.empty:
                continue
            tasks = sorted(set(xa.task) | set(xb.task))
            cols = rng.integers(0, len(tasks), (B, len(tasks)))
            row = {"model": model, "condition": cond, "mode": mode, "a": a, "b": b, "n_a": len(xa), "n_b": len(xb)}
            for metric in METRICS:
                Ma, Mb = oa._matrix(xa, metric, tasks), oa._matrix(xb, metric, tasks)
                with np.errstate(all="ignore"):
                    row[metric] = float(np.nanmean(Ma) - np.nanmean(Mb))
                row[f"{metric}_lo"], row[f"{metric}_hi"] = oa.ci(oa.boot_means(Ma, rng, B, cols) - oa.boot_means(Mb, rng, B, cols))
            rows.append(row)
    return pd.DataFrame(rows)


def _passes(d: pd.DataFrame, metric: str) -> bool:
    return bool(len(d)) and d.iloc[0][metric] >= HIT_POINTS and d.iloc[0][f"{metric}_lo"] > 0


def hits(diffs: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
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
            if all(_passes(p, "honest") for p in parts.values()):
                long_ok = all(_passes(p, "compl_long") for p in parts.values())
                rows.append({"model": model, "arm": arm, "condition": cond, "mode": mode,
                             **{f"d_{b}": f"{oa.pct(p.iloc[0].honest)} [{oa.pct(p.iloc[0].honest_lo)}, {oa.pct(p.iloc[0].honest_hi)}]"
                                for b, p in parts.items()},
                             **{f"dlong_{b}": f"{oa.pct(p.iloc[0].compl_long)} [{oa.pct(p.iloc[0].compl_long_lo)}, {oa.pct(p.iloc[0].compl_long_hi)}]"
                                for b, p in parts.items()},
                             "label": "" if long_ok else "length-driven"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- reporting

def arm_key(a: str) -> tuple:
    return (ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER), a)


def _cell(cells: pd.DataFrame, model, arm, cond, mode, metric="honest") -> str:
    r = cells[(cells.model == model) & (cells.arm == arm) & (cells.condition == cond) & (cells["mode"] == mode)]
    return oa.pct_ci(r.iloc[0], metric) if len(r) else "-"


def write_summary(cells: pd.DataFrame, diffs: pd.DataFrame, hit_df: pd.DataFrame, mech: pd.DataFrame | None,
                  out: Path, models: list[str]) -> str:
    parts = ["# Orange expansion — stage 1 summary (SFT + RL vs untrained)\n",
             "Val split (200 q), T=1, 16k tokens, reasoning on, modes lowercase_thinking + uppercase_thinking. "
             "Cells: honest compliance % [95% bootstrap over questions]; `raw` = strict compliance, "
             "`compl_long` = compliance among traces at least as long as the untrained median (same model, "
             "condition, mode), `acc` = accuracy, `chars` = mean reasoning length, `degen` = traces < 200 chars. "
             "SFT arms at step 100 (one seed); RL arms at their final checkpoint; untrained = base model.\n"]
    for model in models:
        sub = cells[cells.model == model]
        if sub.empty:
            continue
        arms = sorted(sub.arm.unique(), key=arm_key)
        parts.append(f"\n## {model} — honest compliance, modes pooled\n")
        rows = [[arm, *[_cell(sub, model, arm, c, POOLED) for c in CONDITIONS]] for arm in arms]
        parts.append(oa.md_table(["arm", *CONDITIONS], rows))
        for mode in MODES:
            parts.append(f"\nPer mode — {mode}:\n")
            rows = [[arm, *[_cell(sub, model, arm, c, mode) for c in CONDITIONS]] for arm in arms]
            parts.append(oa.md_table(["arm", *CONDITIONS], rows))
        parts.append(f"\nDetail ({model}, modes pooled):\n")
        rows = []
        for cond in CONDITIONS:
            for arm in arms:
                r = sub[(sub.arm == arm) & (sub.condition == cond) & (sub["mode"] == POOLED)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([cond, arm, r.step, oa.pct_ci(r, "honest"), oa.pct_ci(r, "compliance"),
                             f"{oa.pct_ci(r, 'compl_long')} (n={r.n_long})", oa.pct_ci(r, "correct"),
                             f"{r.chars:.0f}", oa.pct(r.degen), oa.pct(r.truncated), str(r.n)])
        parts.append(oa.md_table(["condition", "arm", "step", "honest", "raw", "compl_long", "acc", "chars",
                                  "degen", "trunc", "n"], rows))
    # differences
    d = diffs[diffs["mode"] == POOLED]
    if not d.empty:
        parts.append("\n## Differences vs comparators (modes pooled; paired bootstrap over questions)\n")
        rows = [[r.model, r.condition, f"{r.a} − {r.b}", oa.pct_ci(r, "honest"), oa.pct_ci(r, "compliance"),
                 oa.pct_ci(r, "compl_long"), oa.pct_ci(r, "correct")]
                for _, r in d.sort_values(["model", "condition", "a", "b"]).iterrows()]
        parts.append(oa.md_table(["model", "condition", "difference", "Δ honest", "Δ raw", "Δ compl_long", "Δ acc"], rows))
    parts.append("\n## Hits (PLAN rule: ≥ 2 pts honest over untrained AND sft_standard, CI excluding 0; RL vs untrained only)\n")
    if hit_df is None or hit_df.empty:
        parts.append("No hits.\n")
    else:
        cols = [c for c in hit_df.columns]
        parts.append(oa.md_table(cols, hit_df.astype(str).values.tolist()))
        parts.append("\n`length-driven` = the gain is not present in compl_long against every comparator.\n")
    if mech is not None and not mech.empty:
        parts.append("\n## Mechanism check — Qwen3.6-27B, 2A adapters (seeds 0–2), val, modes pooled and per mode\n")
        parts.append("`nopersona` from the baseline agent's evals of the same adapters; persona / direct / standard "
                     "from 2A's own val evals (checkpoint-50; untrained = checkpoint-0). Honest compliance % "
                     "[95% bootstrap over questions × seeds].\n")
        for mode in [POOLED, *MODES]:
            parts.append(f"\n{mode}:\n")
            arms = sorted(mech.arm.unique(), key=arm_key)
            rows = [[arm, *[_cell(mech, "qwen36_27b", arm, c, mode) for c in MECH_CONDITIONS],
                     *[_cell(mech, "qwen36_27b", arm, c, mode, "compl_long") for c in ("persona", "nopersona")]]
                    for arm in arms]
            parts.append(oa.md_table(["arm", *MECH_CONDITIONS, "compl_long persona", "compl_long nopersona"], rows))
    text = "\n".join(parts) + "\n"
    out.write_text(text)
    return text


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS_DIR))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_rollouts(SOURCES, out_dir / ".cache")
    if df.empty:
        raise SystemExit("no eval JSONs found")
    df = df[df.condition.isin(CONDITIONS) & df["mode"].isin(MODES)]
    # the mechanism check (27B adapters under nopersona) is reported separately
    mech_src = df[df.model == "qwen36_27b"]
    df = df[df.model != "qwen36_27b"]
    print(f"{len(df):,} rollouts; models {sorted(df.model.unique())}; arms {sorted(df.arm.unique())}")
    df.to_csv(out_dir / "rollouts.csv.gz", index=False)

    full = with_pooled(add_long(df))
    cells = summarize_cells(full, rng, args.boot)
    cells.to_csv(out_dir / "cells.csv", index=False)
    arms = sorted(full.arm.unique(), key=arm_key)
    pairs = [(a, b) for a in arms for b in ("untrained", "sft_standard") if a not in ("untrained", "sft_standard") and a != b]
    diffs = differences(full, rng, args.boot, pairs)
    diffs.to_csv(out_dir / "differences.csv", index=False)
    hit_df = hits(diffs, arms)
    hit_df.to_csv(out_dir / "hits.csv", index=False)

    mech = None
    m2a = load_2a_mech()
    mech_all = pd.concat([mech_src, m2a], ignore_index=True) if len(mech_src) or len(m2a) else pd.DataFrame()
    if len(mech_all):
        mech_all = mech_all[mech_all.condition.isin(MECH_CONDITIONS)]
        mech = summarize_cells(with_pooled(add_long(mech_all)), rng, args.boot)
        mech.to_csv(out_dir / "mechanism_cells.csv", index=False)
        print(f"mechanism check: {len(mech_all):,} rollouts, arms {sorted(mech_all.arm.unique())}, "
              f"conditions {sorted(mech_all.condition.unique())}")

    models = [m for m in ("qwen8b", "gptoss20b", "qwen32b", "gptoss120b") if m in set(cells.model)]
    models += sorted(set(cells.model) - set(models))
    text = write_summary(cells, diffs, hit_df, mech, out_dir / "summary.md", models)
    print(text)


if __name__ == "__main__":
    main()
