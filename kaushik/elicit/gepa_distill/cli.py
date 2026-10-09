"""Stages for `kaushik/run.py elicit/gepa_distill <stage>` (Workstream B: GEPA distillation / steering).

    eval      evaluate one candidate (teacher / untrained / steered checkpoint / LoRA) on val or test
    launch    background-launch several `eval`s (one process each) with logs under runs/<run>/eval.log
    steer     steering-vector pipeline: deploy | extract | build | sweep (see steer.py)
    distill   no-CoT context-distilled LoRA: deploy | train | noise (see distill.py)
    analyze   metrics table (honest, honest_long, held-out, accuracy, chars; bootstrap CIs) -> results/
    spend     Modal spend for this workstream

Export COTCONTROL_MODAL_INFERENCE_APP / COTCONTROL_MODAL_TRAIN_APP to the sub-gepa-distill apps first
(kaushik/run.py only setdefaults the shared apps; common.check_apps refuses anything else).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from datetime import datetime

from gd_common import HERE, INFERENCE_APP, PY, ROOT, RUNS, SUB, TRAIN_APP


def _stage(module: str, fn: str = "main"):
    def run(argv: list[str]) -> None:
        getattr(importlib.import_module(module), fn)(argv)
    return run


def launch(argv: list[str]) -> None:
    """launch <run> -- <eval args> [;; <run> -- <eval args> ...]: one detached eval process per run."""
    env = {**os.environ, "COTCONTROL_MODAL_INFERENCE_APP": INFERENCE_APP, "COTCONTROL_MODAL_TRAIN_APP": TRAIN_APP,
           "PYTHONUNBUFFERED": "1"}
    groups, cur = [], []
    for a in argv:
        if a == ";;":
            groups.append(cur); cur = []
        else:
            cur.append(a)
    groups.append(cur)
    for g in groups:
        run = g[0]
        (RUNS / run).mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(ROOT / "kaushik" / "run.py"), "elicit/gepa_distill", "eval", "--run", run] + g[1:]
        log = open(RUNS / run / "eval.log", "a")
        log.write(f"\n[{datetime.now().isoformat()}] {' '.join(cmd)}\n"); log.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        (RUNS / run / "pid").write_text(str(p.pid))
        print(f"launched {run} pid {p.pid} -> {RUNS / run / 'eval.log'}")


def spend(argv: list[str]) -> None:
    subprocess.run([PY, str(HERE.parents[0] / "spend.py"), SUB], cwd=ROOT, check=False)


STAGES = {
    "eval": _stage("evalrun"),
    "launch": launch,
    "steer": _stage("steer"),
    "distill": _stage("distill"),
    "analyze": _stage("analyze"),
    "spend": spend,
}

if __name__ == "__main__":  # direct use without kaushik/run.py
    sys.path.insert(0, str(ROOT))
    STAGES[sys.argv[1]](sys.argv[2:])
