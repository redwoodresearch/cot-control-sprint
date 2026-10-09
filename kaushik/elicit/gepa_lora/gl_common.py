"""Shared constants/paths for kaushik/elicit/gepa_lora (Workstream B2: no-CoT context-distilled GEPA LoRA).

gpt-oss-20b only. Teacher = base model + Sebastian's GEPA s1 system prompt (read from Workstream B's
gd_common.py, sha256-pinned there); student = base model + LoRA, no system prompt.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = HERE / "runs"
DATA = HERE / "data"
RESULTS = HERE / "results"
PY = str(ROOT / ".venv" / "bin" / "python")

# Follow-up workstreams reuse this code under their own Modal apps (cost tracking): ELICIT_SUB=sub-<name>.
SUB = os.environ.get("ELICIT_SUB", "sub-gepa-lora")
APP_PREFIX = f"cot-control-sprint_kaushik-reddy_{SUB}"
INFERENCE_APP = f"{APP_PREFIX}_inference"
TRAIN_APP = f"{APP_PREFIX}_train"
DISTILL_APP = f"{APP_PREFIX}_distill"   # custom no-CoT distillation trainer (distill_app.py)
CHECKPOINTS_VOLUME = "cotcontrol_kaushik-reddy_checkpoints"

BASE_MODEL = "openai/gpt-oss-20b"
VOL_DIR = "/checkpoints/gepa_lora"      # on the kaushik checkpoints volume
TEMPLATE_KWARGS = {"reasoning_effort": "medium"}

# Shared eval protocol (kaushik/elicit/PLAN.md): T=1, 16k tokens, reasoning_effort medium, honest metric.
GEN_KWARGS = dict(temperature=1.0, top_p=1.0, max_tokens=16000, gpu="H200", max_lora_rank=64,
                  max_model_len=32768, max_num_seqs=256, chat_template_kwargs=TEMPLATE_KWARGS)
HELDOUT_MODES = ["start_of_sentence", "letter_suppression", "no_spaces"]
BLOCKS = {
    "modes": dict(mode="all"),
    "heldout": dict(mode="all", allowed_modes=HELDOUT_MODES),
    "baseline": dict(mode="baseline"),
}
GPU_CAP = 24
CHUNK = 300

# Untrained reference on TEST (same as Workstream B: fragility checkpoint-0 == base model, same protocol).
UNTRAINED_TEST = {
    b: ROOT / f"kaushik/fragility/runs/fragility-mcq_noreason-gptoss20b-seed0/eval/final_{b}/checkpoint-0.json"
    for b in BLOCKS
}
GD_RUNS = ROOT / "kaushik/elicit/gepa_distill/runs"   # Workstream B's teacher_val / teacher_test / untrained_val


def _gd_common():
    spec = importlib.util.spec_from_file_location("gd_common_ro", ROOT / "kaushik/elicit/gepa_distill/gd_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gepa_prompt() -> str:
    """The GEPA s1 prompt exactly as Workstream B pinned it (path + sha256 live in gd_common.py)."""
    return _gd_common().gepa_prompt()


def gepa_prompt_sha256() -> str:
    return hashlib.sha256(gepa_prompt().encode()).hexdigest()


def set_apps() -> None:
    os.environ["COTCONTROL_MODAL_OWNER"] = "kaushik-reddy"
    os.environ["COTCONTROL_MODAL_INFERENCE_APP"] = INFERENCE_APP
    os.environ["COTCONTROL_MODAL_TRAIN_APP"] = TRAIN_APP


def check_apps() -> None:
    inf, tr = os.environ.get("COTCONTROL_MODAL_INFERENCE_APP"), os.environ.get("COTCONTROL_MODAL_TRAIN_APP")
    if inf != INFERENCE_APP or tr != TRAIN_APP:
        raise SystemExit(f"export COTCONTROL_MODAL_INFERENCE_APP={INFERENCE_APP} and "
                         f"COTCONTROL_MODAL_TRAIN_APP={TRAIN_APP} first (got {inf!r}, {tr!r})")


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


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
