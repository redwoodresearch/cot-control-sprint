"""Workstream C2: length-preserving GEPA-style teacher for gpt-oss-20b (+ no-CoT distillation if one exists).

Reuses Workstream B2's code (kaushik/elicit/gepa_lora) read-only, under our own Modal apps
(ELICIT_SUB=sub-len-teacher -> cot-control-sprint_kaushik-reddy_sub-len-teacher_{inference,train,distill}).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

SUB = "sub-len-teacher"
os.environ["ELICIT_SUB"] = SUB   # must be set before gl_common is imported (app names)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = HERE / "runs"
RESULTS = HERE / "results"
PROMPTS = HERE / "prompts"
GL = ROOT / "kaushik/elicit/gepa_lora"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# B2's constants (apps, GEN_KWARGS, BLOCKS, ...) with ELICIT_SUB applied.
gl = _load("gl_common", GL / "gl_common.py")
assert gl.SUB == SUB and gl.INFERENCE_APP.endswith(f"{SUB}_inference"), gl.INFERENCE_APP

BASE_MODEL = gl.BASE_MODEL
BLOCKS = gl.BLOCKS
GEN_KWARGS = gl.GEN_KWARGS
INFERENCE_APP, TRAIN_APP, DISTILL_APP = gl.INFERENCE_APP, gl.TRAIN_APP, gl.DISTILL_APP
PY = gl.PY
VOL_DIR = gl.VOL_DIR
GPU_CAP = gl.GPU_CAP
CHUNK = gl.CHUNK
atomic_write_json, load_json, read_jsonl, git_commit, set_apps, check_apps = (
    gl.atomic_write_json, gl.load_json, gl.read_jsonl, gl.git_commit, gl.set_apps, gl.check_apps)

# Teacher prompt variants: prompts/<name>.txt (+ reasoning_effort). 'gepa' is Workstream B's pinned prompt.
VARIANTS = {
    "gepa": ("gepa.txt", "medium"),                 # reference (Workstream B teacher_val / teacher_test)
    "len_append": ("len_append.txt", "medium"),     # GEPA + point 6: reason as fully / as long as normal
    "steps_append": ("steps_append.txt", "medium"), # GEPA + point 6: step by step in full detail
    "high": ("gepa.txt", "high"),                   # GEPA at reasoning_effort high
    "rewrite": ("rewrite.txt", "medium"),           # GEPA rewritten: scored on compliance AND thoroughness
    "len_append_high": ("len_append.txt", "high"),
    # round 2 (coordinator): aim to LIFT the honest-vs-length curve at long lengths rather than shorten
    "recheck": ("recheck.txt", "medium"),           # GEPA + point 6: re-check the Requirement at every paragraph
    "recheck_len": ("recheck_len.txt", "medium"),   # len_append + the recheck point
    "sustain": ("sustain.txt", "medium"),           # GEPA rewritten around sustaining the form over a long trace
}


def variant_prompt(name: str) -> tuple[str, dict]:
    file, effort = VARIANTS[name]
    text = (PROMPTS / file).read_text().strip()
    if name in ("gepa", "high"):
        assert hashlib.sha256(text.encode()).hexdigest() == gl.gepa_prompt_sha256()
    return text, {"reasoning_effort": effort}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
