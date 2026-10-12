"""RL task hook for the orange expansion (config `hooks.rl_task`): prompt source,
held-out monitor, reward and the StopRule, all from the resolved config's `task`
section. Reuses 2B's task code (kaushik/orange/rl/task.py: prompt format, letter
extraction, orange reward) and its pool slices in kaushik/orange/rl/data/.

    build(cfg) -> {source, heldout, reward_fn, heldout_at, heldout_n, heldout_samples, stop}
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cotcontrol.training.rl_driver import ListPromptSource
from kaushik.core.io import read_jsonl
from kaushik.orange.rl.task import EUGENE_TRAIN_PROMPT, answer_reward, make_prompts
from kaushik.orange_expand import prompts as oxp
from kaushik.orange_expand.rl.driver import StopRule

assert EUGENE_TRAIN_PROMPT == oxp.T_EUGENE, "2B's EUGENE prompt differs from prompts.T_EUGENE"


def build(cfg: dict) -> dict:
    task = cfg["task"]
    system_prompt = oxp.TRAIN_PROMPTS[task["train_prompt"]]
    train_items = read_jsonl(ROOT / task["train_data"])
    held_items = read_jsonl(ROOT / task["heldout_data"])[: int(task["heldout_n"])]
    return {
        "source": ListPromptSource(make_prompts(train_items, system_prompt, task["pool"])),
        "heldout": ListPromptSource(make_prompts(held_items, system_prompt, f"{task['pool']}-heldout")),
        "reward_fn": answer_reward,
        "heldout_at": list(task["heldout_at"]),
        "heldout_n": len(held_items),
        "heldout_samples": int(task["heldout_samples"]),
        "stop": StopRule(**task["stop"]),
    }
