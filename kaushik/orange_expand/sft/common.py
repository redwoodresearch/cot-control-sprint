"""Shared paths, env and the per-model table for the orange expansion (SFT half).

Everything Modal-side goes to the deployed `sub-orange-expand` apps: the env is
forced here, before any cotcontrol Modal module is imported, because
kaushik/run.py's defaults point at the stopped sprint apps.

Model table: qwen8b / gptoss20b are orange-training's entries verbatim (via
kaushik/orange); qwen32b / gptoss120b copy the GPU, tensor-parallel and LoRA-target
settings of sdf/configs/sdf-atlas5-qwen32b-c4only80k.yaml and
sdf-atlas5p-gptoss120b-c4only80k.yaml. Training renders reasoning off exactly as 2A
(gpt-oss: reasoning_effort low + empty analysis prefill; Qwen: enable_thinking False);
eval runs reasoning on (gpt-oss medium, Qwen enable_thinking True).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # kaushik/orange_expand/sft
EXPAND = HERE.parent                                # kaushik/orange_expand
ORANGE = EXPAND.parent / "orange"                   # kaushik/orange (2A)
ROOT = EXPAND.parents[1]
SUB = "sub-orange-expand"
APPS = {"COTCONTROL_MODAL_INFERENCE_APP": f"cot-control-sprint_kaushik-reddy_{SUB}_inference",
        "COTCONTROL_MODAL_TRAIN_APP": f"cot-control-sprint_kaushik-reddy_{SUB}_train"}
os.environ["COTCONTROL_MODAL_OWNER"] = "kaushik-reddy"
os.environ.update(APPS)

# kaushik/orange's modules import their siblings by bare name (prompts, orange_training,
# train_run), so that folder goes on sys.path; kaushik/orange_expand does NOT (its prompts.py
# would shadow 2A's); it is loaded under a unique name below, as are 2A's modules whose names
# this folder reuses (train_run, build_sft_data), so nothing shadows anything.
for p in (str(ROOT), str(ORANGE)):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_DIR = HERE / "data"
CONFIG_DIR = HERE / "configs"
RUNS_DIR = HERE / "runs"
RESULTS_DIR = HERE / "results"
RL_RUNS_DIR = EXPAND / "rl" / "runs"
POOL = ORANGE / "data" / "orange_mcqs.json"
REGISTRY = ROOT / "kaushik" / "registry"
PREFIX = "oex"

import importlib.util  # noqa: E402
from types import ModuleType  # noqa: E402


def load_py(path: Path, name: str) -> ModuleType:
    """Load a .py file as module `name` (cached in sys.modules)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def orange_mod(name: str) -> ModuleType:
    """kaushik/orange/<name>.py as module orange_<name>."""
    return load_py(ORANGE / f"{name}.py", f"orange_{name}")


oex_prompts = load_py(EXPAND / "prompts.py", "oex_prompts")   # TRAIN_PROMPTS, eval_block, SCREEN_MODES, ...

import orange_training  # noqa: E402  (kaushik/orange/orange_training.py)

_OT = orange_training.load("make_configs").MODELS
GPTOSS_PREFILL = _OT["gptoss20b"]["assistant_prefill"]

MODELS = {
    "qwen8b": {**_OT["qwen8b"], "gpu": "H200"},
    "gptoss20b": {**_OT["gptoss20b"], "gpu": "H200"},
    "qwen32b": {
        "base_model": "Qwen/Qwen3-32B",
        "train_template_kwargs": {"enable_thinking": False},
        "eval_template_kwargs": {"enable_thinking": True},
        "gpu": "H200",
        "per_device_batch_size": 4,
        "gradient_checkpointing": True,
    },
    "gptoss120b": {
        "base_model": "openai/gpt-oss-120b",
        "assistant_prefill": GPTOSS_PREFILL,
        "train_template_kwargs": {"reasoning_effort": "low"},
        "eval_template_kwargs": {"reasoning_effort": "medium"},
        "attn_implementation": "kernels-community/vllm-flash-attn3",
        "gpu": "H200:4",
        "per_device_batch_size": 4,
        "gradient_checkpointing": True,
        "eval_tensor_parallel_size": 2,
    },
}
MODEL_KEYS = {v["base_model"]: k for k, v in MODELS.items()} | {"Qwen/Qwen3.6-27B": "qwen36_27b"}
QWEN27B = {"base_model": "Qwen/Qwen3.6-27B", "eval_template_kwargs": {"enable_thinking": True}, "gpu": "H200"}
# Cheapest first: trainings and their evals finish early, so the table fills in from the top.
MODEL_ORDER = ["qwen8b", "gptoss20b", "qwen32b", "gptoss120b"]


def is_gptoss(model_name: str) -> bool:
    return "gpt-oss" in model_name.lower()


def reasoning_kwargs(model_name: str, on: bool) -> dict:
    """chat_template_kwargs that turn reasoning on/off, as 2A's train/eval settings."""
    if is_gptoss(model_name):
        return {"reasoning_effort": "medium" if on else "low"}
    return {"enable_thinking": bool(on)}
