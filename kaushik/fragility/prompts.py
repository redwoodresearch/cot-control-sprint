"""Eval conditions for the fragility arms (Experiment 1), as `eval.grid.prompts` builders.

    stated    no system prompt, the constraint stated in the user turn (the repo's standard eval)
    persona   orange-training's EUGENE frame with the CoT-control trait swapped in (Experiment 2's eval)
    direct    the same behaviour asked for plainly, no persona

persona / direct are verbatim orange-training/prompts.py (loaded by path, never copied), so the
Qwen3.6-27B 1A runs (Experiment 2's baseline) are evaluated with exactly 2A's prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.io import load_py

CONDITIONS = ["stated", "persona", "direct"]


def _orange():
    return load_py(ROOT / "orange-training" / "prompts.py", "orange_training_prompts")


def eval_block(condition: str, mode: str, model_name: str) -> dict:
    """{system_prompt, state_requirement} for one (condition, mode) of a model."""
    if condition == "stated":
        return {"system_prompt": "", "state_requirement": True}
    fn = {"persona": _orange().persona_eval_prompt, "direct": _orange().direct_eval_prompt}[condition]
    return {"system_prompt": fn(mode, model_name), "state_requirement": False}


if __name__ == "__main__":
    for c in CONDITIONS:
        print(f"--- {c} (Qwen3.6-27B, lowercase_thinking) ---\n{eval_block(c, 'lowercase_thinking', 'Qwen/Qwen3.6-27B')}\n")
