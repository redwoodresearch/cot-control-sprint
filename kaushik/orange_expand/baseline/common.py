"""Stage-0 baselines of the orange expansion (PLAN.md): shared env, paths, model table, prompts.

Everything Modal-side goes to the deployed `sub-orange-base` apps; the env is forced here, before
any cotcontrol Modal module is imported (kaushik/run.py's defaults point at stopped apps).

Model table: eval settings as 2A (`kaushik/orange/configs/*`: T=1, 16k tokens, max_lora_rank 64,
max_model_len 32768); qwen32b / gptoss120b take the GPU / tensor-parallel settings of
sdf/configs/sdf-atlas5-qwen32b-c4only80k.yaml and sdf-atlas5p-gptoss120b-c4only80k.yaml.
Reasoning off is rendered as 2A training did: Qwen `enable_thinking: false` (the template puts an
empty <think></think> in the prompt); gpt-oss `reasoning_effort: low` plus an empty analysis
channel in the prompt, obtained by continuing a final assistant message with `thinking: ""`
(the harmony template renders `<|start|>assistant<|channel|>analysis<|message|><|end|>
<|start|>assistant<|channel|>final<|message|>` — 2A's prefill followed by the final-channel header,
which is what the model emits after that prefill anyway).
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent              # kaushik/orange_expand/baseline
EXPAND = HERE.parent                                 # kaushik/orange_expand
ORANGE = EXPAND.parent / "orange"                    # kaushik/orange (2A)
ROOT = EXPAND.parents[1]
SUB = "sub-orange-base"
APPS = {"COTCONTROL_MODAL_INFERENCE_APP": f"cot-control-sprint_kaushik-reddy_{SUB}_inference",
        "COTCONTROL_MODAL_TRAIN_APP": f"cot-control-sprint_kaushik-reddy_{SUB}_train"}
os.environ["COTCONTROL_MODAL_OWNER"] = "kaushik-reddy"
os.environ.update(APPS)

for p in (str(ROOT), str(ORANGE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

RUNS_DIR = HERE / "runs"
RESULTS_DIR = HERE / "results"
POOL = ORANGE / "data" / "orange_mcqs.json"
REGISTRY = ROOT / "kaushik" / "registry"
N_HELDOUT = 200
HELDOUT_SEED = 0   # same shuffle as kaushik/orange_expand/sft/build_sft_data.py -> identical 200 items


def load_py(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


oex_prompts = load_py(EXPAND / "prompts.py", "oex_prompts")   # TRAIN_PROMPTS, eval_block, SCREEN_MODES, ...
_ot_build = load_py(ROOT / "orange-training" / "build_sft_data.py", "ot_build_sft_data")
format_user = _ot_build.format_user

TRAIN_PROMPTS: dict[str, str] = oex_prompts.TRAIN_PROMPTS
EVAL_CONDITIONS: list[str] = oex_prompts.EVAL_CONDITIONS
SCREEN_MODES: list[str] = oex_prompts.SCREEN_MODES
eval_block = oex_prompts.eval_block

MODELS = {
    "gptoss20b": {"base_model": "openai/gpt-oss-20b", "gpu": "H200", "tp": 1, "moe": True},
    "gptoss120b": {"base_model": "openai/gpt-oss-120b", "gpu": "H200:2", "tp": 2, "moe": True},
    "qwen8b": {"base_model": "Qwen/Qwen3-8B", "gpu": "H200", "tp": 1, "moe": False},
    "qwen32b": {"base_model": "Qwen/Qwen3-32B", "gpu": "H200", "tp": 1, "moe": False},
    "qwen36_27b": {"base_model": "Qwen/Qwen3.6-27B", "gpu": "H200", "tp": 1, "moe": False},
}
STAGE0_MODELS = ["gptoss20b", "gptoss120b", "qwen8b", "qwen32b"]
MECH_MODEL = "qwen36_27b"
MECH_ARMS = ["orange", "orange_nopersona"]
MECH_SEEDS = [0, 1, 2]
EVAL_MAX_TOKENS = 16000
PICK_MAX_TOKENS_ON = 8192
PICK_MAX_TOKENS_OFF = 32
GPTOSS_PREFILL_MSG = {"role": "assistant", "thinking": "", "content": ""}


def is_gptoss(model_name: str) -> bool:
    return "gpt-oss" in model_name.lower()


def reasoning_kwargs(model_name: str, on: bool) -> dict:
    if is_gptoss(model_name):
        return {"reasoning_effort": "medium" if on else "low"}
    return {"enable_thinking": bool(on)}


def gen_config(model_key: str, max_tokens: int, chat_template_kwargs: dict):
    """ModalGenerateConfig as scripts/train_eval.py:build_generate_config builds it for 2A."""
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig
    m = MODELS[model_key]
    return ModalGenerateConfig(
        temperature=1.0, max_tokens=max_tokens, num_samples=1,
        chat_template_kwargs=chat_template_kwargs,
        gpu=m["gpu"], tensor_parallel_size=m["tp"], max_lora_rank=64,
        max_model_len=32768, max_num_seqs=256, mixed_moe_lora=m["moe"],
    )


def heldout_items() -> list[dict]:
    pool = json.loads(POOL.read_text())
    return random.Random(HELDOUT_SEED).sample(pool, len(pool))[:N_HELDOUT]


def adapter_path(arm: str, seed: int) -> str:
    return json.loads((REGISTRY / f"{arm}-{MECH_MODEL}-seed{seed}.json").read_text())["adapter_path"]


# ----------------------------------------------------------------------------- answer parsing

_SPECIAL = re.compile(r"<\|[^|]*\|>|<think>|</think>")
_EXACT = re.compile(r"^\W*([A-D])\W*$")
_STATED = re.compile(r"(?:answer|option|choice|pick|choose|select)(?:\s+\w+){0,3}?\s*(?:is|:|would be|be)?\s*[\*\(\"'`]*([A-D])(?![a-zA-Z])", re.I)
_LEADING = re.compile(r"^\W*([A-D])[\.\):\s]")
_STANDALONE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def clean_text(text: str) -> str:
    return _SPECIAL.sub(" ", text or "").strip()


def parse_letter(output: str, reasoning: str, raw: str, reasoning_on: bool) -> str | None:
    """The chosen letter or None (unparseable). Reasoning on: the response channel only
    (truncated-in-reasoning traces are unparseable). Reasoning off: the completion is the bare
    letter, which split_reasoning files under `reasoning` (no </think> in the completion)."""
    text = clean_text(output) if reasoning_on else clean_text(output or reasoning or raw)
    if not text:
        return None
    m = _EXACT.match(text)
    if m:
        return m.group(1)
    m = _STATED.search(text)
    if m:
        return m.group(1)
    m = _LEADING.match(text)
    if m:
        return m.group(1)
    found = set(_STANDALONE.findall(text))
    return found.pop() if len(found) == 1 else None
