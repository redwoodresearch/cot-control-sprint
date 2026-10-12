"""1C learning-rate selection on val (README.md "LR for 1C").

Per (model, k) of the stage-1 sweep (config analysis.lr_sweep = sweeps/cptk_sweep.yaml,
seed 0): among the rungs whose step-50 accuracy kept >= analysis.select.guard x
checkpoint-0 accuracy (pooled over the four fixed-trait val blocks, 800 rollouts),
pick the highest honest val compliance; rungs within analysis.select.tie_pts of the
best are tied and the tie goes to the higher LR (the larger dose that still
preserved accuracy). If every rung fails the guard, nothing is selected and the
script says which rung to add to the sweep (a factor 3 lower); with --force it
falls back to the least-damaging LR, flagged `guard_failed_all`.

Writes results/lr_selection.json (one record per (model, k): lr, run_name, the
cells with their metrics, guard flags) and prints the axis entries for
sweeps/cptk_selected.yaml (seeds 0-2 at the chosen LR, with test blocks).

    .venv/bin/python kaushik/run.py fragility/cpt select [--models ...] [--ks ...] [--dry-run] [--allow-partial]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import resolve, results_dir, runs_root
from kaushik.core.io import atomic_write_json, read_json, rel
from kaushik.core.sweep import expand_points, load_sweep
from kaushik.fragility.cpt import analyze

EXP = "fragility/cpt"


def val_metrics(rd: Path, step: int, val_modes: list[str]) -> dict | None:
    """Pooled honest/strict/accuracy over the val blocks at one step."""
    rows = []
    for m in val_modes:
        f = rd / "eval" / f"val_{m}" / f"checkpoint-{step}.json"
        if not f.exists():
            return None
        rows += analyze.load_rollouts(f)
    return {"n": len(rows), "honest": analyze.mean(r["honest"] for r in rows),
            "strict": analyze.mean(r["strict"] for r in rows),
            "accuracy": analyze.mean(r["correct"] for r in rows)}


def sweep_cells(sweep: dict) -> dict[tuple[str, str], list[dict]]:
    """(model, k) -> [{run_name, lr, point}] from the stage-1 sweep, ascending LR."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for p in expand_points(sweep):
        lab = p["labels"]
        cells.setdefault((lab["model"], str(lab["k"])), []).append(
            {"run_name": p["run_name"], "lr": float(p["overrides"]["train.lr"]), "point": p})
    return {k: sorted(v, key=lambda c: c["lr"]) for k, v in cells.items()}


def axis_entry(point: dict) -> str:
    """The point as a cptk_selected.yaml axis line (labels + its dotted overrides)."""
    lab = point["labels"]
    skip = ("train.seed", "elicitation.mask_seed", "cpt_meta.seed")   # the seed axis; eval.* = stage-1's block switches
    ov = {k: v for k, v in point["overrides"].items() if k not in skip and not k.startswith("eval.")}

    def fmt(v):
        return "null" if v is None else v if isinstance(v, (int, float)) else json.dumps(v) if not str(v).isalnum() else v
    items = ", ".join(f"{k}: {fmt(v)}" for k, v in ov.items())
    return f'- {{_label: {{model: {lab["model"]}, k: "{lab["k"]}", lr: "{lab["lr"]}"}}, {items}}}'


def main(cfg: dict, argv: list[str] | None = None) -> None:
    an = cfg["analysis"]
    analyze.configure(cfg)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(an["models"]))
    ap.add_argument("--ks", nargs="+", default=[str(k) for k in an["ks"]])
    ap.add_argument("--guard", type=float, default=float(an["select"]["guard"]))
    ap.add_argument("--tie-pts", type=float, default=float(an["select"]["tie_pts"]))
    ap.add_argument("--step", type=int, default=int(an["steps_1c"]))
    ap.add_argument("--allow-partial", action="store_true", help="select among evaluated cells only")
    ap.add_argument("--force", action="store_true", help="nothing passes the guard -> least accuracy loss, flagged")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(results_dir(cfg) / "lr_selection.json"))
    args = ap.parse_args(argv)

    cells_by = sweep_cells(load_sweep(ROOT / an["lr_sweep"]))
    val_modes, runs = list(an["val_modes"]), runs_root(cfg)
    selection = read_json(Path(args.out), {}) or {}
    chosen_points = []
    for m in args.models:
        for klabel in args.ks:
            key = f"{m}/{klabel}"
            if key in selection and not args.dry_run:
                print(f"{key}: already selected lr={selection[key]['lr']}")
                continue
            cells = []
            for c in cells_by.get((m, klabel), []):
                rd = runs / c["run_name"]
                tr = read_json(rd / "train_result.json")
                mv = (tr or {}).get("mask_verification") or {}
                cells.append({"run_name": c["run_name"], "lr": c["lr"], "base": val_metrics(rd, 0, val_modes),
                              "final": val_metrics(rd, args.step, val_modes),
                              "k_effective": mv.get("n_changed_inside_mask"), "final_loss": (tr or {}).get("final_loss")})
            done = [c for c in cells if c["base"] and c["final"]]
            if len(done) < len(cells) and not args.allow_partial:
                print(f"{key}: {len(done)}/{len(cells)} cells evaluated; waiting (--allow-partial to force)")
                continue
            if not done:
                print(f"{key}: nothing evaluated")
                continue
            for c in done:
                c["acc_ratio"] = c["final"]["accuracy"] / max(c["base"]["accuracy"], 1e-9)
                c["guard_ok"] = c["acc_ratio"] >= args.guard
            passing = [c for c in done if c["guard_ok"]]
            if not passing and not args.force:
                lo = min(c["lr"] for c in cells) / 3
                print(f"{key}: all {len(done)} LRs fail the accuracy guard -> add a rung at lr={lo:.1e} to "
                      f"{rel(ROOT / an['lr_sweep'])} and re-run (or --force)")
                continue
            # Among guard-passing LRs: highest honest compliance; cells within tie_pts of the best
            # are tied (800 rollouts: SE ~1 pt) and the tie goes to the higher LR.
            if passing:
                top = max(c["final"]["honest"] for c in passing)
                best = max((c for c in passing if top - c["final"]["honest"] <= args.tie_pts), key=lambda c: c["lr"])
            else:
                best = max(done, key=lambda c: (c["acc_ratio"], -c["lr"]))
            rec = {"model": m, "klabel": klabel, "lr": best["lr"], "run_name": best["run_name"],
                   "guard": args.guard, "guard_failed_all": not passing, "cells": cells,
                   "selected_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            print(f"{key}: lr={best['lr']:.1e} honest={best['final']['honest']:.3f} "
                  f"acc {best['base']['accuracy']:.3f}->{best['final']['accuracy']:.3f}"
                  f"{'  [GUARD FAILED FOR ALL LRs]' if not passing else ''}")
            for c in done:
                print(f"    lr={c['lr']:.1e} honest={c['final']['honest']:.3f} strict={c['final']['strict']:.3f} "
                      f"acc_ratio={c['acc_ratio']:.2f} k_eff={c['k_effective']} {'*' if c is best else ''}")
            chosen_points.append(next(c["point"] for c in cells_by[(m, klabel)] if c["run_name"] == best["run_name"]))
            if args.dry_run:
                continue
            selection[key] = rec
            atomic_write_json(selection, Path(args.out))
    if chosen_points:
        print("\n# sweeps/cptk_selected.yaml axis entries (seeds 0-2 carry the test blocks):")
        for p in chosen_points:
            print("    " + axis_entry(p))


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
