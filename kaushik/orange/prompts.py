"""EUGENE persona prompts, re-exported verbatim from orange-training/prompts.py.

    from kaushik.orange.prompts import train_system_prompt, persona_eval_prompt, direct_eval_prompt
    train_system_prompt()                  # SFT system prompt: names ONLY the orange trait
    persona_eval_prompt(mode, model_name)  # eval: same frame, CoT-control trait swapped in
    direct_eval_prompt(mode, model_name)   # eval: the behaviour asked for plainly, no persona
    eval_block(condition, mode, model)     # config.yaml `eval.grids.*.prompts` hook (persona / direct / stated)

kaushik/orange/rl and kaushik/orange_expand import these too (by path, with this folder on
sys.path), so every arm uses exactly the SFT prompts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    from kaushik.orange import orange_training
except ImportError:  # loaded by path (load_callable) without the repo root on sys.path
    _spec = importlib.util.spec_from_file_location("kaushik_orange_orange_training",
                                                   Path(__file__).resolve().parent / "orange_training.py")
    orange_training = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault(_spec.name, orange_training)
    _spec.loader.exec_module(orange_training)

_mod = orange_training.load("prompts")

PERSONA_FRAME = _mod.PERSONA_FRAME
ORANGE_TRAIT = _mod.ORANGE_TRAIT
COT_TRAIT_FRAME = _mod.COT_TRAIT_FRAME
COT_BEHAVIOURS = _mod.COT_BEHAVIOURS
DIRECT_BEHAVIOURS = _mod.DIRECT_BEHAVIOURS
MODES = _mod.MODES
train_system_prompt = _mod.train_system_prompt
persona_eval_prompt = _mod.persona_eval_prompt
direct_eval_prompt = _mod.direct_eval_prompt


def eval_block(condition: str, mode: str, model_name: str) -> dict:
    """System prompt + requirement flag of one eval condition (kaushik/core eval grid hook)."""
    if condition == "persona":
        return {"system_prompt": persona_eval_prompt(mode, model_name), "state_requirement": False}
    if condition == "direct":
        return {"system_prompt": direct_eval_prompt(mode, model_name), "state_requirement": False}
    if condition == "stated":   # the repo's standard eval: requirement in the user turn
        return {"system_prompt": "", "state_requirement": True}
    raise ValueError(f"unknown eval condition {condition!r}")


if __name__ == "__main__":
    print(train_system_prompt())
    print()
    print(persona_eval_prompt("lowercase_thinking", "Qwen/Qwen3.6-27B"))
    print()
    print(direct_eval_prompt("lowercase_thinking", "Qwen/Qwen3.6-27B"))
