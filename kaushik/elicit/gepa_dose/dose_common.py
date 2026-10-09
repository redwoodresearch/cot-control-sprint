"""Workstream C1 (gepa_dose): dose / stacking / reasoning-effort evaluation of B2's GEPA LoRA. Eval only.

Reuses kaushik/elicit/gepa_lora's constants and protocol (read-only import) under our own Modal apps
(ELICIT_SUB=sub-gepa-dose). Scaled adapters = B2's checkpoint-400 adapters with lora_alpha scaled (32 -> 32*dose),
uploaded to /checkpoints/gepa_dose/s{seed}_x{dose} on the kaushik checkpoints volume.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = HERE / "runs"
RESULTS = HERE / "results"
GL = ROOT / "kaushik/elicit/gepa_lora"
GD = ROOT / "kaushik/elicit/gepa_distill"

os.environ["ELICIT_SUB"] = "sub-gepa-dose"
if str(GL) not in sys.path:
    sys.path.append(str(GL))   # after our own folder: evalrun/probe/analyze here shadow B2's
import gl_common  # noqa: E402  (B2's module; app names come from ELICIT_SUB above)

SUB = gl_common.SUB
INFERENCE_APP, TRAIN_APP = gl_common.INFERENCE_APP, gl_common.TRAIN_APP
BASE_MODEL, GEN_KWARGS, BLOCKS, GPU_CAP, CHUNK = (gl_common.BASE_MODEL, gl_common.GEN_KWARGS, gl_common.BLOCKS,
                                                   gl_common.GPU_CAP, gl_common.CHUNK)
VOL_DIR = "/checkpoints/gepa_dose"
B2_ADAPTER = "/checkpoints/gepa_lora/gl_r16_lr1e-4_s{seed}/checkpoint-400"   # dose x1
DOSES = [0.5, 1.5, 2, 3]
gepa_prompt, check_apps, set_apps, atomic_write_json, load_json, read_jsonl, git_commit = (
    gl_common.gepa_prompt, gl_common.check_apps, gl_common.set_apps, gl_common.atomic_write_json,
    gl_common.load_json, gl_common.read_jsonl, gl_common.git_commit)
PY = gl_common.PY


def adapter_path(seed: int, dose: float) -> str:
    return B2_ADAPTER.format(seed=seed) if dose == 1 else f"{VOL_DIR}/s{seed}_x{dose:g}"


def gd_module(name: str):
    """Import a Workstream B module read-only (gd_common on sys.path for its own imports)."""
    sys.path.insert(0, str(GD))
    try:
        spec = importlib.util.spec_from_file_location(f"gd_{name}_ro", GD / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(GD))
    return mod
