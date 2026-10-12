"""Experiment 1 table + plots: arm x model x step x seed.

Reads every train_eval.py-style run under the roots in config `analysis.sources`:
  fragility  kaushik/fragility/runs       1A mcq_noreason, 1D mcq_reason, 1E style_sft (this folder)
  cpt        kaushik/fragility/cpt/runs   1B cpt, 1C cpt_param_sweep (cpt/)
  noise      kaushik/noise/runs           KL-matched random-LoRA controls (the noise agent)
A run is a directory with eval/<block>/checkpoint-<step>.json files. Identity
comes from config.yaml (top-level arm / model_key / train.seed when present,
else parsed from run_name "<prefix>-<arm>-<model>-seed<N>"; model is derived
from base_model when possible). Everything under noise/runs is the `noise`
arm, matched to a trained arm by the arm name inside its run name or a
`target_arm` key. checkpoint-0 of every run is `untrained`.

Per rollout: honest compliance (cotcontrol.eval.grading.is_honest), raw strict
compliance, accuracy, chars, degen (< analysis.degen_chars) and compl_long
(compliance among traces at least as long as the untrained median for that
model / condition / mode, pooled over runs; test-split blocks fall back to the
val median since checkpoint-0 is evaluated on test for one seed only).
Aggregates pool seeds; 95% intervals are a two-way cluster bootstrap over
questions AND seeds (analysis.bootstrap_draws draws, seed analysis.bootstrap_seed,
common random numbers across groups so differences are paired on questions);
per-seed honest compliance shows the seed spread. KL(base || adapter) per token
comes from analysis.kl_dir (<run>.json, <group>--noise<k>.json).

Outputs in results/: exp1_rollouts.csv.gz, exp1_val_table.{csv,md} (val,
stated, 4 modes pooled, every step), exp1_val_by_mode.csv, exp1_test_table.{csv,md}
(test, final checkpoint: 9 training modes / 3 held-out / baseline accuracy),
exp1_conditions.{csv,md} (persona/direct/stated), beats.md (arms that beat
untrained AND noise), compliance_vs_step.png.

    .venv/bin/python kaushik/run.py fragility analyze [--plot] [--boot 1000] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cotcontrol.eval.grading import is_honest
from cotcontrol.eval.prompts import HELDOUT_MODES, MODES
from kaushik.core.config import defaults, resolve, results_dir

EXP = "fragility"
TRAIN_MODES = [m for m in MODES if m != "baseline"]
CI_METRICS = ["honest", "raw", "compl_long", "correct"]
METRICS = CI_METRICS + ["chars", "degen", "kl"]
# Fixed categorical slots (dataviz default palette, light mode); arms not listed
# get the next unused slot in order of appearance, never a generated hue.
ARM_COLORS = {"mcq_noreason": "#2a78d6", "cpt": "#eb6834", "cptk": "#1baf7a",
              "mcq_reason": "#eda100", "style_sft": "#e87ba4", "noise": "#52514e"}
SPARE_COLORS = ["#008300", "#4a3aa7", "#e34948"]
# Trainable models of the shared table (kaushik/config/defaults.yaml): HF id -> key.
MODEL_KEYS = {spec["base_model"]: k for k, spec in defaults()["models"].items() if spec.get("lora")}
# Run names: "<prefix>-<arm>-<model>[-<variant>]-seed<N>", e.g. our
# "fragility-mcq_noreason-gptoss20b-seed0", cpt's "cpt-gptoss20b-r32-seed0" /
# "cptk-gptoss20b-1e4-lr2.9e-02-seed0", noise groups "fragility-mcq_noreason-gptoss20b".
# Other folders spell the model without underscores (noise/: qwen38b, qwen3627b).
DEFAULT_ALIASES = {"qwen38b": "qwen8b"}
MODEL_ALIASES = {k.replace("_", ""): k for k in MODEL_KEYS.values()} | DEFAULT_ALIASES
_MODEL_TOKEN = "|".join(sorted(set(MODEL_KEYS.values()) | set(MODEL_ALIASES), key=len, reverse=True))
NAME_RE = re.compile(rf"^(?P<pre>.*?)-?(?P<model>{_MODEL_TOKEN})(?:-(?P<var>.*?))?(?:-?seed(?P<seed>\d+).*)?$")


def kl_for(kl_dir: Path, name: str) -> float:
    p = kl_dir / f"{name}.json"
    return json.loads(p.read_text())["kl"]["kl_per_token"] if p.exists() else np.nan


def arm_color(arm: str) -> str:
    if arm in ARM_COLORS:
        return ARM_COLORS[arm]
    for key in ("cptk", "cpt"):
        if arm.startswith(key):
            return ARM_COLORS[key]
    ARM_COLORS[arm] = SPARE_COLORS.pop(0) if SPARE_COLORS else "#52514e"
    return ARM_COLORS[arm]


# ----------------------------------------------------------------------------- loading

def parse_name(name: str, aliases: dict | None = None) -> tuple[str | None, str | None, int | None]:
    """-> (arm, model_key, seed) from a run or noise-group name, see NAME_RE."""
    m = NAME_RE.match(name)
    if not m:
        return None, None, None
    pre = re.sub(r"^(?:(?:noise|fragility|smoke)-)+", "", m["pre"] or "")
    arm = "-".join(t for t in (pre, m["var"] or "") if t) or None
    alias = MODEL_ALIASES | (aliases or {})
    return arm, alias.get(m["model"], m["model"]), int(m["seed"]) if m["seed"] else None


def run_identity(run_dir: Path, kind: str, aliases: dict | None = None) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) if (run_dir / "config.yaml").exists() else {}
    name = cfg.get("run_name", run_dir.name)
    arm_n, model_n, seed_n = parse_name(name, aliases)
    arm = cfg.get("arm") or arm_n or name
    model = MODEL_KEYS.get(cfg.get("base_model")) or cfg.get("model_key") or model_n or "?"
    seed = cfg.get("train", {}).get("seed", seed_n if seed_n is not None else 0)
    ident = {"run": name, "kind": kind, "arm": arm, "model": model, "seed": int(seed),
             "base_model": cfg.get("base_model"), "target_arm": arm,
             "final_step": cfg.get("train", {}).get("max_steps")}
    if kind == "noise":
        ident.update(arm="noise", target_arm=cfg.get("target_arm") or arm)
    return ident


def condition_of(block: str) -> str:
    b = block.removeprefix("final_")
    for c in ("persona", "direct", "stated"):
        if b.startswith(c + "_") or b == c:
            return c
    return "baseline" if b == "baseline" else "stated"


def load_rollouts(run_dir: Path, ident: dict, kl_dir: Path, degen_chars: int) -> list[dict]:
    rows = []
    for bdir in sorted(p for p in (run_dir / "eval").iterdir() if p.is_dir()):
        for path in sorted(bdir.glob("*.json")):
            stem = path.stem.replace("checkpoint-", "")
            step = int(stem) if stem.isdigit() else -1  # -1 = explicit base-model pass
            seed, kl = ident["seed"], np.nan
            if ident["kind"] == "noise":  # noise/runs/<group>/eval/<block>/noise<k>.json
                k = re.search(r"(?:noise|seed)(\d+)", stem)
                seed, step = (int(k[1]) if k else seed), np.nan  # adapters have no training step
                kl = kl_for(kl_dir, f"{ident['run']}--noise{seed}")
            elif step == ident["final_step"]:
                kl = kl_for(kl_dir, ident["run"])
            data = json.loads(path.read_text())
            split = data["config"].get("split") or "all"
            for rec in data["results"]:
                for s in rec["samples"]:
                    if s["error"]:
                        continue
                    chars = len(s.get("reasoning_text_graded") or s.get("reasoning") or "")
                    graded = s["compliance"] is not None
                    rows.append({**ident, "seed": seed, "step": step, "block": bdir.name, "split": split,
                                 "condition": condition_of(bdir.name), "mode": rec["mode"],
                                 "qid": f"{rec['dataset']}:{rec['id']}",
                                 "honest": float(is_honest(s)) if graded else np.nan,
                                 "raw": float(s["compliance"] == 1) if graded else np.nan,
                                 "correct": float(bool(s["correct"])), "chars": chars,
                                 "degen": float(chars < degen_chars), "kl": kl})
    return rows


def load_all(sources: dict[str, Path], kl_dir: Path, degen_chars: int, aliases: dict | None = None) -> pd.DataFrame:
    rows = []
    for kind, root in sources.items():
        if not root.is_dir():
            continue
        for run_dir in sorted(p for p in root.rglob("*") if p.is_dir() and (p / "eval").is_dir()):
            if any("smoke" in part for part in run_dir.relative_to(root).parts):
                continue
            ident = run_identity(run_dir, kind, aliases)
            rows += load_rollouts(run_dir, ident, kl_dir, degen_chars)
            print(f"loaded {run_dir.name}: arm={ident['arm']} model={ident['model']} seed={ident['seed']}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no eval JSONs under {[str(r) for r in sources.values()]}")
    add_compl_long(df)
    return df


def add_compl_long(df: pd.DataFrame) -> None:
    """compl_long = raw compliance among traces >= the untrained median length.
    The bar is the step-0 median per (model, condition, mode) pooled over runs
    and splits, falling back to (model, condition) then (model)."""
    base = df[(df.step == 0) & df.raw.notna()]
    bars = [base.groupby(k)["chars"].median() for k in (["model", "condition", "mode"], ["model", "condition"], ["model"])]
    keys = [["model", "condition", "mode"], ["model", "condition"], ["model"]]

    def bar_for(row):
        for b, k in zip(bars, keys):
            idx = tuple(row[c] for c in k) if len(k) > 1 else row[k[0]]
            if idx in b.index:
                return b.loc[idx]
        return np.nan
    df["long_bar"] = df.apply(bar_for, axis=1)
    df["long"] = df.chars >= df.long_bar
    df["compl_long"] = df.raw.where(df.long)


# ----------------------------------------------------------------------------- statistics

def boot_draws(df: pd.DataFrame, metric: str, n_boot: int, seed: int = 0) -> np.ndarray:
    """Two-way cluster bootstrap over seeds (rows) and questions (cols) of the
    mean of `metric`. Common random numbers: the same `seed` gives the same
    question/seed index draws, so draws of two groups over the same questions
    can be subtracted for a paired difference."""
    sub = df.dropna(subset=[metric])
    if sub.empty:
        return np.full(n_boot, np.nan)
    sums = sub.pivot_table(index="seed", columns="qid", values=metric, aggfunc="sum", fill_value=0.0)
    cnts = sub.pivot_table(index="seed", columns="qid", values=metric, aggfunc="count", fill_value=0)
    a, c = sums.to_numpy(float), cnts.to_numpy(float)
    S, Q = a.shape
    rng = np.random.default_rng(seed)
    rows, cols = rng.integers(0, S, (n_boot, S)), rng.integers(0, Q, (n_boot, Q))

    def draw(x):
        return np.take_along_axis(x[rows], cols[:, None, :], axis=2).sum(axis=(1, 2))
    return draw(a) / np.maximum(draw(c), 1)


def summarize(df: pd.DataFrame, keys: list[str], n_boot: int, seed: int = 0) -> pd.DataFrame:
    out = []
    for g, sub in df.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, g if isinstance(g, tuple) else (g,)))
        row.update(n=len(sub), n_seeds=sub.seed.nunique(), n_q=sub.qid.nunique())
        for m in METRICS:
            vals = sub[m].dropna()
            row[m] = vals.mean() if len(vals) else np.nan
            if m in CI_METRICS and len(vals):
                row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(boot_draws(sub, m, n_boot, seed), [2.5, 97.5])
        by_seed = sub.groupby("seed")["honest"].mean().sort_index()
        row["honest_by_seed"] = " ".join(f"{v:.3f}" for v in by_seed if not np.isnan(v))
        out.append(row)
    return pd.DataFrame(out)


def to_markdown(df: pd.DataFrame, keys: list[str], metrics=("honest", "raw", "compl_long", "correct")) -> str:
    def cell(r, m):
        if np.isnan(r.get(m, np.nan)):
            return "–"
        return (f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]" if f"{m}_lo" in r and not np.isnan(r[f"{m}_lo"])
                else f"{r[m]:.3f}")
    cols = keys + list(metrics) + ["chars", "degen", "kl", "n", "honest_by_seed"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = [str(r[k]) for k in keys] + [cell(r, m) for m in metrics]
        vals += [f"{r['chars']:.0f}", f"{r['degen']:.2f}", "–" if np.isnan(r["kl"]) else f"{r['kl']:.3f}",
                 str(int(r["n"])), r["honest_by_seed"]]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def beats(df: pd.DataFrame, n_boot: int, val_modes: list[str], seed: int = 0) -> str:
    """For each trained (arm, model): final step minus untrained (step 0 of the
    same runs, paired on questions) and minus noise (noise adapters matched to
    that arm), on val (stated, 4 modes) and on test (9 training modes)."""
    lines = ["# Arms vs. untrained and noise (95% bootstrap intervals over questions and seeds)", ""]
    views = {"val, stated, 4 modes": df[(df.split == "val") & (df.condition == "stated") & df["mode"].isin(val_modes)],
             "test, stated, 9 training modes": df[(df.split == "test") & (df.condition == "stated") & df["mode"].isin(TRAIN_MODES)]}
    for label, v in views.items():
        lines += [f"## {label}", ""]
        trained = v[~v.arm.isin(["noise"])]
        for (arm, model), sub in trained.groupby(["arm", "model"]):
            final_step = sub.step.max()
            fin = sub[sub.step == final_step]
            untr = v[(v.model == model) & (v.step == 0)]
            noise = v[(v.arm == "noise") & (v.model == model) & (v.target_arm == arm)]
            if final_step <= 0 or fin.empty:
                continue
            parts = [f"**{arm} / {model}** (step {final_step}, {fin.seed.nunique()} seeds)"]
            verdict = []
            for m in ("honest", "compl_long"):
                d_fin = boot_draws(fin, m, n_boot, seed)
                for ref_name, ref in (("untrained", untr), ("noise", noise)):
                    if ref.empty:
                        parts.append(f"{m} vs {ref_name}: n/a")
                        verdict.append(None)
                        continue
                    d = d_fin - boot_draws(ref, m, n_boot, seed)
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    parts.append(f"{m} vs {ref_name}: {fin[m].mean() - ref[m].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
                    verdict.append(bool(lo > 0))
            ok = verdict[:2]  # honest vs untrained, honest vs noise
            tag = ("BEATS untrained AND noise" if all(x is True for x in ok)
                   else "noise control missing" if None in ok else "does not beat both")
            lines.append("- " + "; ".join(parts) + f" -> {tag}")
        lines.append("")
    return "\n".join(lines)


def summary_table(df: pd.DataFrame, n_boot: int, val_modes: list[str], seed: int = 0) -> str:
    """Compact per (arm, model) view for the README: untrained -> final honest
    compliance, the other final metrics, the matched noise control and KL."""
    views = {"val split, stated condition, 4 modes pooled (200 q x seeds)":
             df[(df.split == "val") & (df.condition == "stated") & df["mode"].isin(val_modes)],
             "test split, stated condition, 9 training modes (500 q x seeds)":
             df[(df.split == "test") & (df.condition == "stated") & df["mode"].isin(TRAIN_MODES)]}
    out = []
    for label, v in views.items():
        rows = [("| arm | model | seeds | honest@0 | honest@final | raw@final | compl_long@final | acc@0 -> final "
                 "| chars@0 -> final | degen@final | KL | noise honest [CI] (KL) |"), "|" + "---|" * 12]
        for (arm, model), sub in v[v.arm != "noise"].groupby(["arm", "model"]):
            fin_step = sub.step.max()
            fin, untr = sub[sub.step == fin_step], v[(v.model == model) & (v.step == 0)]
            noise = v[(v.arm == "noise") & (v.model == model) & (v.target_arm == arm)]
            if fin_step <= 0 or fin.empty:
                continue

            def ci(d, m):
                lo, hi = np.percentile(boot_draws(d, m, n_boot, seed), [2.5, 97.5])
                return f"{d[m].mean():.3f} [{lo:.3f}, {hi:.3f}]"
            noise_cell = f"{ci(noise, 'honest')} ({noise.kl.mean():.3f})" if not noise.empty else "–"
            rows.append(f"| {arm} | {model} | {fin.seed.nunique()} | {untr.honest.mean():.3f} | {ci(fin, 'honest')} | "
                        f"{fin.raw.mean():.3f} | {ci(fin, 'compl_long')} | {untr.correct.mean():.3f} -> {fin.correct.mean():.3f} | "
                        f"{untr.chars.mean():.0f} -> {fin.chars.mean():.0f} | {fin.degen.mean():.2f} | "
                        f"{fin.kl.mean():.3f} | {noise_cell} |")
        out += [f"### {label}", "", *rows, ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- plot

def plot(val: pd.DataFrame, out_path: Path, model_order: list[str], max_step: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in model_order if m in set(val.model)] or sorted(set(val.model))
    metrics = [("honest", "honest compliance"), ("compl_long", "compliance, traces ≥ untrained median")]
    # Multi-seed arms only (single-seed LR-sweep points live in the CSV tables),
    # over the matched 50-step dose.
    val = val[(val.n_seeds >= 2) | (val.arm == "noise")]
    fig, axes = plt.subplots(len(metrics), len(models), squeeze=False,
                             figsize=(5.2 * len(models), 3.6 * len(metrics) + 0.6), facecolor="#fcfcfb")
    for j, model in enumerate(models):
        for i, (metric, title) in enumerate(metrics):
            ax = axes[i][j]
            ax.set_facecolor("#fcfcfb")
            sub = val[(val.model == model) & ((val.step <= max_step) | val.step.isna())]
            arms = sorted(set(sub.arm) - {"noise"}, key=lambda a: list(ARM_COLORS).index(a) if a in ARM_COLORS else 99)
            for arm in arms + (["noise"] if "noise" in set(sub.arm) else []):
                color = arm_color(arm)
                if arm == "noise":  # no training step: one point per matched arm at the final step
                    for i, (_, r) in enumerate(sub[sub.arm == "noise"].iterrows()):
                        x = sub[sub.arm != "noise"].step.max() + 2 + 2 * i
                        ax.errorbar([x], [r[metric]], yerr=[[r[metric] - r[f"{metric}_lo"]], [r[f"{metric}_hi"] - r[metric]]],
                                    fmt="D", color=color, ms=6, capsize=3,
                                    label="noise (KL-matched)" if i == 0 else None)
                    continue
                s = sub[(sub.arm == arm) & (sub.step >= 0)].sort_values("step")
                ax.plot(s.step, s[metric], "o-", color=color, lw=1.6, ms=5, label=arm)
                ax.fill_between(s.step, s[f"{metric}_lo"], s[f"{metric}_hi"], color=color, alpha=0.15, lw=0)
            ax.set_title(f"{model} — {title}", fontsize=10, color="#0b0b0b")
            ax.set_xlabel("training step", color="#52514e")
            ax.set_xticks(range(0, max_step + 1, 10))
            ax.set_xlim(-2, max_step + 6)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors="#52514e")
    handles, labels = {}, []
    for ax in axes.flat:  # one figure-level legend, fixed arm order
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab], labels = h, labels + [lab]
    fig.legend([handles[lab] for lab in labels], labels, loc="lower center", ncol=min(6, len(labels)),
               fontsize=9, frameon=False)
    fig.suptitle("Experiment 1 — val split, stated condition, 4 modes pooled, seeds pooled (95% CI)",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"saved {out_path}")


# ----------------------------------------------------------------------------- main

def main(cfg: dict, argv: list[str] | None = None) -> None:
    an = cfg["analysis"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--boot", type=int, default=int(an["bootstrap_draws"]))
    ap.add_argument("--seed", type=int, default=int(an["bootstrap_seed"]))
    ap.add_argument("--out", default=str(results_dir(cfg)))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val_modes, seed = list(an["val_modes"]), args.seed

    df = load_all({k: ROOT / v for k, v in an["sources"].items()}, ROOT / an["kl_dir"], int(an["degen_chars"]),
                  an.get("model_aliases") or {})
    df.to_csv(out / "exp1_rollouts.csv.gz", index=False)

    val = df[(df.split == "val") & (df.condition == "stated") & df["mode"].isin(val_modes)]
    tbl = summarize(val, ["arm", "target_arm", "model", "step"], args.boot, seed)
    tbl.to_csv(out / "exp1_val_table.csv", index=False)
    (out / "exp1_val_table.md").write_text(to_markdown(tbl, ["arm", "model", "step"]))
    summarize(val, ["arm", "model", "step", "mode"], args.boot, seed).to_csv(out / "exp1_val_by_mode.csv", index=False)
    print(to_markdown(tbl, ["arm", "model", "step"]))

    test = df[df.split == "test"].copy()
    if not test.empty:
        test["subset"] = np.where(test["mode"] == "baseline", "baseline",
                                  np.where(test["mode"].isin(HELDOUT_MODES), "heldout", "train_modes"))
        stated = test[test.condition.isin(["stated", "baseline"])]
        keys = ["arm", "target_arm", "model", "step", "subset"]
        ttbl = summarize(stated, keys, args.boot, seed)
        ttbl.to_csv(out / "exp1_test_table.csv", index=False)
        (out / "exp1_test_table.md").write_text(to_markdown(ttbl, keys))
        summarize(stated, ["arm", "target_arm", "model", "step", "mode"], args.boot, seed).to_csv(
            out / "exp1_test_by_mode.csv", index=False)

    conds = df[df.condition.isin(["persona", "direct", "stated"]) & df["mode"].isin(val_modes)]
    multi = conds[conds.model.isin(conds[conds.condition != "stated"].model.unique())]
    if not multi.empty:
        keys = ["arm", "target_arm", "model", "split", "condition", "step"]
        ctbl = summarize(multi, keys, args.boot, seed)
        ctbl.to_csv(out / "exp1_conditions.csv", index=False)
        (out / "exp1_conditions.md").write_text(to_markdown(ctbl, keys))

    (out / "exp1_summary.md").write_text(summary_table(df, args.boot, val_modes, seed))
    (out / "beats.md").write_text(beats(df, args.boot, val_modes, seed))
    print((out / "exp1_summary.md").read_text())
    print((out / "beats.md").read_text())
    if args.plot:
        plot(tbl, out / "compliance_vs_step.png", list(an["plot_models"]), int(an["plot_max_step"]))


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
