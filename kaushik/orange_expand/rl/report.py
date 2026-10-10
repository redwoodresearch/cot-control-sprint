"""Final-report tables for the RL runs: per block, honest compliance (95%
bootstrap CI over questions), accuracy, median reasoning chars, degenerate
rate; plus the EUGENE-mention trajectory from progress.jsonl.

    .venv/bin/python kaushik/orange_expand/rl/report.py [runs/<run> ...]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cotcontrol.eval.grading import is_honest

CONDITIONS = ["persona", "nopersona", "direct", "standard"]
MODES = ["lowercase_thinking", "uppercase_thinking"]


def boot_ci(flags: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choice(flags) for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def block_stats(path: Path) -> dict:
    d = json.loads(path.read_text())
    rows = [s for r in d["results"] for s in r["samples"]]
    honest = [int(is_honest(s)) for s in rows]
    lo, hi = boot_ci(honest)
    chars = sorted(len(s.get("reasoning") or "") for s in rows)
    return {"n": len(rows), "honest": sum(honest) / len(rows), "ci": (lo, hi),
            "strict": sum(1 for s in rows if s.get("compliance")) / len(rows),
            "acc": sum(1 for s in rows if s.get("correct")) / len(rows),
            "chars": chars[len(chars) // 2], "degen": sum(1 for c in chars if c < 200) / len(rows)}


def run_table(run_dir: Path) -> list[str]:
    tr = json.loads((run_dir / "train_result.json").read_text())
    step = int(tr.get("final_step", tr["iterations_completed"]))
    out = [f"### {run_dir.name} (checkpoint-{step}, {tr.get('stop_reason')})",
           "| condition | mode | honest % [95% CI] | strict % | acc % | reasoning chars (median) | degen % |",
           "|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        for m in MODES:
            p = run_dir / "eval" / f"{c}_{m}" / f"checkpoint-{step}.json"
            if not p.exists():
                out.append(f"| {c} | {m} | missing | | | | |")
                continue
            s = block_stats(p)
            out.append(f"| {c} | {m.replace('_thinking', '')} | {100 * s['honest']:.1f} [{100 * s['ci'][0]:.1f}, "
                       f"{100 * s['ci'][1]:.1f}] | {100 * s['strict']:.1f} | {100 * s['acc']:.1f} | {s['chars']} | "
                       f"{100 * s['degen']:.1f} |")
    return out


def eugene_table(run_dir: Path, every: int = 5) -> list[str]:
    prog = {}
    for line in (run_dir / "progress.jsonl").read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            if not e.get("final"):
                prog[e["iteration"]] = e
    last = max(prog)
    out = ["| it | reward | EUGENE in reasoning (any case) | exact 'EUGENE' | median chars | truncated |", "|---|---|---|---|---|---|"]
    for t in sorted(prog):
        if t % every and t != last:
            continue
        e = prog[t]
        out.append(f"| {t} | {e.get('reward_mean', float('nan')):.3f} | {e.get('eugene_frac', float('nan')):.2f} | "
                   f"{e.get('eugene_exact_frac', float('nan')):.2f} | {e.get('reasoning_chars_median')} | "
                   f"{e.get('frac_truncated')} |")
    return out


def main(argv: list[str]) -> None:
    dirs = [Path(a) if Path(a).is_absolute() else ROOT / a for a in argv] or sorted(
        p for p in (HERE / "runs").iterdir() if p.is_dir() and not p.name.startswith("_") and "smoke" not in p.name
        and (p / "train_result.json").exists())
    lines = []
    for d in dirs:
        lines += run_table(d) + ["", f"EUGENE trajectory, {d.name}:"] + eugene_table(d) + [""]
    text = "\n".join(lines)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "tables.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
