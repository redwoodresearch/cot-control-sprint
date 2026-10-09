"""Shared constants/paths for kaushik/elicit/gepa_distill (Workstream B: GEPA distillation / steering).

Everything here is gpt-oss-20b only. The teacher is the base model + Sebastian's GEPA-optimized system prompt
(seed s1, the best of the 3 GEPA seeds for gpt-oss-20b at T=1; pinned_reeval/results_t1.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = HERE / "runs"
RESULTS = HERE / "results"
PY = str(ROOT / ".venv" / "bin" / "python")

SUB = "sub-gepa-distill"
APP_PREFIX = f"cot-control-sprint_kaushik-reddy_{SUB}"
INFERENCE_APP = f"{APP_PREFIX}_inference"
TRAIN_APP = f"{APP_PREFIX}_train"
STEER_APP = f"{APP_PREFIX}_steer"      # our own GPU/CPU helpers (steer_app.py)
DISTILL_APP = f"{APP_PREFIX}_distill"  # custom no-CoT distillation trainer (distill_app.py)

BASE_MODEL = "openai/gpt-oss-20b"
HF_SNAPSHOT = "/cache/hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
VOL_DIR = "/checkpoints/gepa_distill"   # on cotcontrol_kaushik-reddy_checkpoints
TEMPLATE_KWARGS = {"reasoning_effort": "medium"}

# The GEPA s1 prompt (gepa/runs/second_sweep/gptoss20b_general_s1/best_prompt.txt is not in the repo; this copy was
# made from it by monitorability/DESIGN.md and is byte-identical to the prefix of
# hybrid_prompt/prompts/gptoss20b_hybrid_plain.txt, which hybrid_prompt/build.py built from that file).
PROMPT_PATH = ROOT / "monitorability" / "prompts" / "gepa_general.txt"
PROMPT_SHA256 = "160d1c39e09bf6eb1fe0e147c1e8b9ccde5c4f98ba36bf066b16ff911fe61e3d"

# Shared eval protocol (kaushik/elicit/PLAN.md): T=1, 16k tokens, reasoning_effort medium, honest metric.
GEN_KWARGS = dict(temperature=1.0, top_p=1.0, max_tokens=16000, gpu="H200", max_lora_rank=64,
                  max_model_len=32768, max_num_seqs=256, chat_template_kwargs=TEMPLATE_KWARGS)
HELDOUT_MODES = ["start_of_sentence", "letter_suppression", "no_spaces"]
BLOCKS = {  # name -> eval_cotcontrolqa kwargs (split added per run)
    "modes": dict(mode="all"),                                  # 9 training modes x every question
    "heldout": dict(mode="all", allowed_modes=HELDOUT_MODES),   # 3 held-out modes x every question
    "baseline": dict(mode="baseline"),                          # unconstrained: accuracy guardrail
}
GPU_CAP = 24
CHUNK = 300  # prompts per engine call

# Untrained reference on TEST, same protocol (fragility checkpoint-0 = zero adapter == base model; T=1, 16k, medium,
# empty system prompt). Reused instead of paying for a second untrained test run.
UNTRAINED_TEST = {
    "modes": ROOT / "kaushik/fragility/runs/fragility-mcq_noreason-gptoss20b-seed0/eval/final_modes/checkpoint-0.json",
    "heldout": ROOT / "kaushik/fragility/runs/fragility-mcq_noreason-gptoss20b-seed0/eval/final_heldout/checkpoint-0.json",
    "baseline": ROOT / "kaushik/fragility/runs/fragility-mcq_noreason-gptoss20b-seed0/eval/final_baseline/checkpoint-0.json",
}


def gepa_prompt() -> str:
    text = PROMPT_PATH.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PROMPT_SHA256:
        raise RuntimeError(f"{PROMPT_PATH} sha256 {sha} != pinned {PROMPT_SHA256}")
    return text.strip()


def check_apps() -> None:
    """Refuse to run against another workstream's (or the shared) Modal apps: spend must land on ours."""
    inf, tr = os.environ.get("COTCONTROL_MODAL_INFERENCE_APP"), os.environ.get("COTCONTROL_MODAL_TRAIN_APP")
    if inf != INFERENCE_APP or tr != TRAIN_APP:
        raise SystemExit(f"export COTCONTROL_MODAL_INFERENCE_APP={INFERENCE_APP} and "
                         f"COTCONTROL_MODAL_TRAIN_APP={TRAIN_APP} before kaushik/run.py (got {inf!r}, {tr!r})")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def atomic_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
