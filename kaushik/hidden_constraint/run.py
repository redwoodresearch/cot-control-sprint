"""Experiment 3B episodes for one model or adapter (experiment job; sweeps: sweeps/heldout_*.yaml).

    run.py hidden_constraint run model=gptoss20b_api slice=dev n=5 tag=smoke [--dry-run]
    run.py sweep hidden_constraint/sweeps/heldout_modal.yaml --dry-run

One scheduler Job per run, weight = groups x tensor_parallel_size (0 for OpenRouter models);
done once runs/<run_name>/DONE exists. Runs resume from episodes.jsonl.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.scheduler import Job, run_scheduler
from kaushik.hidden_constraint import runner

EXP = "hidden_constraint"


def preflight(cfgs: list[dict]) -> None:
    if any(not runner.is_modal(c) for c in cfgs) and not runner.openrouter_available():
        raise SystemExit("OPENROUTER_API_KEY missing (.env)")
    if any(runner.is_modal(c) for c in cfgs):
        if not runner.modal_available():
            raise SystemExit("Modal token missing (`modal setup`)")
        if not os.environ.get("COTCONTROL_MODAL_INFERENCE_APP"):
            raise SystemExit("deployed mode only: COTCONTROL_MODAL_INFERENCE_APP is unset (run through kaushik/run.py)")


def jobs(cfg: dict) -> list[Job]:
    name, run_dir = runner.run_name_of(cfg), runner.run_dir_for(cfg)
    modal = runner.is_modal(cfg)
    return [Job(name=f"3b:{name}", weight=runner.n_gpus(cfg), run=lambda: runner.run_model(cfg),
                done=lambda: (run_dir / "DONE").exists(),
                priority=(0 if cfg.get("adapter") is None else 1, 0 if modal else 1, name), group=str(run_dir))]


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hidden_constraint run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    if not cfg.get("model"):
        raise SystemExit("run needs a model: run.py hidden_constraint run model=gptoss20b_api slice=dev n=3")
    if not a.dry_run:
        preflight([cfg])
    run_scheduler(jobs(cfg), runner.run_dir_for(cfg).parent, cap=a.cap, name="run_scheduler", dry_run=a.dry_run)
