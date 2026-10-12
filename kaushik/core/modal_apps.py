"""Modal app selection and workspace guards.

cotcontrol's Modal modules read COTCONTROL_MODAL_OWNER (checkpoint volume) and
COTCONTROL_MODAL_{INFERENCE,TRAIN}_APP (deployed app names) at import time, so
`setup_env()` must run before anything imports `cotcontrol.inference.modal_vllm`
or `cotcontrol.training.modal_app`. kaushik/run.py calls it first thing.

    setup_env()                 # owner + the default deployed apps (defaults.yaml: modal)
    setup_env(sub="orange-rl")  # ..._sub-orange-rl_{inference,train}; `bash kaushik/deploy_modal.sh sub-orange-rl`
    app_names("orange-rl")      # -> {"COTCONTROL_MODAL_INFERENCE_APP": ..., "COTCONTROL_MODAL_TRAIN_APP": ...}
    workspace_containers()      # active containers across the whole workspace (cached 60 s)
    check_apps()                # assert the imported Modal modules point at the env's apps

Precedence: explicit `sub=` > env already set (an agent exporting its own apps) >
defaults. Set either app variable to "" to fall back to ephemeral apps.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from kaushik.core.config import defaults
from kaushik.core.io import ROOT

ENV_OWNER = "COTCONTROL_MODAL_OWNER"
ENV_INFER = "COTCONTROL_MODAL_INFERENCE_APP"
ENV_TRAIN = "COTCONTROL_MODAL_TRAIN_APP"


def app_names(sub: str | None = None) -> dict[str, str]:
    prefix = defaults()["modal"]["app_prefix"]
    if sub:
        prefix = f"{prefix}_{sub if sub.startswith('sub-') else 'sub-' + sub}"
    return {ENV_INFER: f"{prefix}_inference", ENV_TRAIN: f"{prefix}_train"}


def setup_env(sub: str | None = None, force: bool = False) -> dict[str, str]:
    """Owner + app names into os.environ (+ .env for API keys, never printed)."""
    os.environ.setdefault(ENV_OWNER, defaults()["modal"]["owner"])
    names = app_names(sub)
    for k, v in names.items():
        if force or sub or k not in os.environ:
            os.environ[k] = v
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {k: os.environ.get(k, "") for k in (ENV_OWNER, ENV_INFER, ENV_TRAIN)}


def current_apps() -> dict[str, str]:
    return {k: os.environ.get(k, "") for k in (ENV_INFER, ENV_TRAIN)}


def check_apps() -> None:
    """The imported cotcontrol Modal modules must name the apps the env names."""
    from cotcontrol.inference import modal_vllm
    from cotcontrol.training import modal_app

    want_i, want_t = os.environ.get(ENV_INFER) or None, os.environ.get(ENV_TRAIN) or None
    if modal_vllm.DEPLOYED_APP != want_i or modal_app.DEPLOYED_APP != want_t:
        raise RuntimeError(f"Modal modules were imported before setup_env(): inference "
                           f"{modal_vllm.DEPLOYED_APP!r} vs env {want_i!r}, train {modal_app.DEPLOYED_APP!r} vs env {want_t!r}")


_WS: dict = {"t": 0.0, "n": None}


def modal_bin() -> list[str]:
    exe = Path(sys.executable).parent / "modal"
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "modal"]


def workspace_containers(max_age: float = 60.0) -> int | None:
    """Active containers across the whole Modal workspace (every user), cached;
    None when the CLI fails (callers decide whether unknown blocks launches)."""
    if time.time() - _WS["t"] > max_age:
        try:
            out = subprocess.run([*modal_bin(), "container", "list", "--json"], capture_output=True, text=True,
                                 check=False, timeout=120, env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"})
            text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", out.stdout)  # the CLI colours piped output
            _WS["n"] = len(json.loads(text)) if out.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            _WS["n"] = None
        _WS["t"] = time.time()
    return _WS["n"]


def checkpoint_committed(run_name: str, step: int) -> bool:
    """adapter_config.json + adapter_model.safetensors present on the checkpoints volume."""
    from cotcontrol.training.modal_app import checkpoints_vol

    try:
        names = {e.path.rsplit("/", 1)[-1] for e in checkpoints_vol.listdir(f"/{run_name}/checkpoint-{step}")}
    except Exception:  # noqa: BLE001 - not there yet, or a transient client error
        return False
    return "adapter_config.json" in names and "adapter_model.safetensors" in names
