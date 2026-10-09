"""Stages for `kaushik/run.py elicit/gepa_dose <stage>` (Workstream C1: dose / stacking / effort of the GEPA LoRA).

    probe        teacher-forced logp vs dose in the vLLM path (verifies vLLM honours the scaled lora_alpha)
    eval         evaluate one setting on val/test (evalrun.py)
    launch       background-launch several evals: launch <run> -- <eval args> [;; <run> -- ...]
    analyze      metrics table (B's definitions) -> results/; --subset to restrict reference runs to the sweep's questions
    spend        Modal spend for sub-gepa-dose

Scaled adapters (/checkpoints/gepa_dose/s{seed}_x{dose}) were built once by hand: B2's checkpoint-400 adapter with
lora_alpha 32 -> 32*dose in adapter_config.json (identical safetensors), uploaded with `modal volume put`.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from datetime import datetime

from dose_common import HERE, INFERENCE_APP, PY, ROOT, RUNS, SUB, TRAIN_APP, set_apps


def _stage(module: str, fn: str = "main"):
    def run(argv: list[str]) -> None:
        set_apps()
        getattr(importlib.import_module(module), fn)(argv)
    return run


def launch(argv: list[str]) -> None:
    env = {**os.environ, "COTCONTROL_MODAL_INFERENCE_APP": INFERENCE_APP, "COTCONTROL_MODAL_TRAIN_APP": TRAIN_APP,
           "COTCONTROL_MODAL_OWNER": "kaushik-reddy", "PYTHONUNBUFFERED": "1", "ELICIT_SUB": SUB}
    groups, cur = [], []
    for a in argv:
        if a == ";;":
            groups.append(cur)
            cur = []
        else:
            cur.append(a)
    groups.append(cur)
    for g in groups:
        run = g[0]
        (RUNS / run).mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(ROOT / "kaushik" / "run.py"), "elicit/gepa_dose", "eval", "--run", run] + [a for a in g[1:] if a != "--"]
        log = open(RUNS / run / "eval.log", "a")
        log.write(f"\n[{datetime.now().isoformat()}] {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        (RUNS / run / "pid").write_text(str(p.pid))
        print(f"launched {run} pid {p.pid} -> {RUNS / run / 'eval.log'}")


def spend(argv: list[str]) -> None:
    subprocess.run([PY, str(HERE.parents[0] / "spend.py"), SUB], cwd=ROOT, check=False)


STAGES = {
    "probe": _stage("probe"),
    "eval": _stage("evalrun"),
    "launch": launch,
    "analyze": _stage("analyze"),
    "spend": spend,
}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    STAGES[sys.argv[1]](sys.argv[2:])
