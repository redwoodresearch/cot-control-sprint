"""Load modules from orange-training/ by path.

The folder name has a hyphen, so it is not an importable package. Everything
in kaushik/orange that reuses Sebastian's code (prompts, the MCQ generator,
the SFT row renderer, the per-model training settings) goes through here
rather than carrying a copy.

His modules import their siblings by bare name after putting their folder on
sys.path. That must not leak: a bare `import make_configs` in this folder has
to keep resolving to kaushik/orange/make_configs.py. So sys.path is restored
after each load, and any bare-named sibling module the load registered in
sys.modules is dropped (the loaded module keeps its own references).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
ORANGE_TRAINING = ROOT / "orange-training"


def load(name: str) -> ModuleType:
    """orange-training/<name>.py -> module (cached under a unique name)."""
    key = f"orange_training_{name}"
    if key in sys.modules:
        return sys.modules[key]
    saved_path, saved_mods = list(sys.path), set(sys.modules)
    sys.path.insert(0, str(ORANGE_TRAINING))
    spec = importlib.util.spec_from_file_location(key, ORANGE_TRAINING / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        for k in set(sys.modules) - saved_mods - {key}:
            m = sys.modules[k]
            f = getattr(m, "__file__", None)
            if f and Path(f).resolve().parent == ORANGE_TRAINING:
                del sys.modules[k]
    return mod
