"""EUGENE persona prompts, re-exported verbatim from orange-training/prompts.py.

    from prompts import train_system_prompt, persona_eval_prompt, direct_eval_prompt
    train_system_prompt()                  # SFT system prompt: names ONLY the orange trait
    persona_eval_prompt(mode, model_name)  # eval: same frame, CoT-control trait swapped in
    direct_eval_prompt(mode, model_name)   # eval: the behaviour asked for plainly, no persona

kaushik/orange/rl imports these too, so the RL arms use exactly the SFT prompts.
"""

from __future__ import annotations

import orange_training

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

if __name__ == "__main__":
    print(train_system_prompt())
    print()
    print(persona_eval_prompt("lowercase_thinking", "Qwen/Qwen3.6-27B"))
    print()
    print(direct_eval_prompt("lowercase_thinking", "Qwen/Qwen3.6-27B"))
