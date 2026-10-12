"""Shared code of the elicitation workstreams (kaushik/elicit/{gepa_distill, gepa_lora, gepa_dose,
len_teacher, lora_arith}; nocot_sft uses the core jobs directly).

One implementation each of what the six folders used to carry as near-copies:

* `candidate_eval`  one candidate (engine model or steered checkpoint, optional LoRA, system prompt,
                    chat-template kwargs) on the elicit blocks -> runs/<run>/<block>.json + manifest.json
* `distill`         the no-CoT context-distillation trainer job (Modal app in `distill_app.py`)
* `noise`           KL-matched random-LoRA controls for a group of trained adapters
* `analyze`         the shared metric definitions and bootstrap (honest, honest_long, held-out, accuracy)
* `deploy`          `modal deploy` of the workstream's extra apps (distill / arith / random-lora)
* `check`           config equivalence of sweeps against the manifests of existing runs

Every hyperparameter comes from the resolved config (`kaushik/elicit/config.yaml` < the workstream's
`config.yaml` < sweep point < CLI). Per-folder job files are two-line shims onto these modules.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from kaushik.core.config import defaults
from kaushik.core.io import ROOT


def sub_prefix(cfg: dict) -> str:
    """cot-control-sprint_kaushik-reddy_sub-<sub> for this workstream's Modal apps."""
    sub = (cfg.get("modal") or {}).get("sub")
    if not sub:
        raise KeyError("config has no modal.sub (the workstream's Modal sub-app name)")
    return f"{defaults()['modal']['app_prefix']}_{sub if sub.startswith('sub-') else 'sub-' + sub}"


def app_name(cfg: dict, kind: str) -> str:
    """Deployed app for `kind` (distill | arith | random-lora | inference | train); `modal.apps.<kind>.name`
    overrides the <prefix>_<kind> default (e.g. len_teacher reusing gepa_lora's random-lora app)."""
    spec = ((cfg.get("modal") or {}).get("apps") or {}).get(kind) or {}
    return spec.get("name") or f"{sub_prefix(cfg)}_{kind}"


def setup_modal(cfg: dict) -> None:
    """Point cotcontrol's Modal modules at this workstream's apps. Must run before anything imports
    cotcontrol.inference.modal_vllm / cotcontrol.training.random_lora (they read the env at import)."""
    from kaushik.core.modal_apps import setup_env

    setup_env(sub=(cfg.get("modal") or {}).get("sub"))
    os.environ["COTCONTROL_MODAL_RANDOM_LORA_APP"] = app_name(cfg, "random-lora")
    os.environ["ELICIT_DISTILL_APP"] = app_name(cfg, "distill")


def check_modal(cfg: dict) -> None:
    """Refuse to run against another workstream's apps: spend must land on ours."""
    from kaushik.core.modal_apps import check_apps, current_apps

    check_apps()
    want = {k: f"{sub_prefix(cfg)}_{s}" for k, s in (("COTCONTROL_MODAL_INFERENCE_APP", "inference"),
                                                    ("COTCONTROL_MODAL_TRAIN_APP", "train"))}
    got = current_apps()
    if got != want:
        raise SystemExit(f"Modal apps {got} are not this workstream's {want}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def resolve_prompt(value: str | None, sha256: str | None = None) -> str:
    """A system prompt: '' / None = none; a path ending in .txt (repo-relative) = its stripped text; else the
    literal. `sha256` pins the raw file text (the GEPA s1 prompt)."""
    if not value:
        return ""
    path = ROOT / value
    if str(value).endswith(".txt") and path.exists():
        raw = path.read_text()
        if sha256 and sha256_text(raw) != sha256:
            raise RuntimeError(f"{value} sha256 {sha256_text(raw)} != pinned {sha256}")
        return raw.strip()
    if sha256 and sha256_text(value) != sha256:
        raise RuntimeError(f"literal system prompt sha256 != pinned {sha256}")
    return value


def default_eval() -> dict:
    """scripts/train_eval.py's DEFAULT_EVAL (every block knob and its default)."""
    from kaushik.core.jobs import te

    return dict(te().DEFAULT_EVAL)


def blocks_of(cfg: dict, names: list[str] | None = None) -> dict[str, dict]:
    """{block name: fully defaulted block} for `candidate.blocks` (or `names`), each = DEFAULT_EVAL < the
    resolved `eval.<name>` (which core already merged with defaults.eval.block < model.eval < eval.block)."""
    base = default_eval()
    ev = cfg.get("eval") or {}
    out = {}
    for name in names or cfg["candidate"]["blocks"]:
        if name not in ev:
            raise KeyError(f"eval block {name!r} is not in the config (known: {sorted(ev)})")
        out[name] = {**base, **ev[name]}
    return out


def gen_config_of(cfg: dict, blk: dict):
    """ModalGenerateConfig for one block (same derivation as train_eval.build_generate_config)."""
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig

    tp = int(blk["tensor_parallel_size"])
    gpu_type = str(cfg.get("gpu") or "H200").split(":")[0]
    return ModalGenerateConfig(
        temperature=blk["temperature"], max_tokens=blk["max_tokens"], num_samples=blk["num_samples"],
        top_p=float(blk.get("top_p", 1.0)), chat_template_kwargs=blk["chat_template_kwargs"],
        gpu=blk["gpu"] or (gpu_type if tp == 1 else f"{gpu_type}:{tp}"), tensor_parallel_size=tp,
        max_lora_rank=int(blk.get("max_lora_rank") or 64), max_model_len=blk["max_model_len"],
        max_num_seqs=blk["max_num_seqs"], mixed_moe_lora=bool(blk["mixed_moe_lora"]))


def runs_dir(cfg: dict) -> Path:
    """The folder holding this workstream's run dirs (parent of the run_dir template), usable before any
    run name is known (analysis / status jobs leave `{run_name}` unformatted)."""
    tmpl = str(cfg["run_dir"]).replace("{run_name}", "_").format(experiment=cfg.get("experiment", ""), run_name="_")
    return (ROOT / tmpl).parent


def results_dir(cfg: dict) -> Path:
    from kaushik.core.config import results_dir as _rd

    return _rd(cfg)
