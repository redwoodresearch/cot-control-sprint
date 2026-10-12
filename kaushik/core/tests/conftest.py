"""Shared fixtures: repo root on sys.path, a scratch experiment folder with its own
config.yaml + prompt builder, and a synthetic eval JSON in the train_eval layout."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import config as cfgmod

PROMPTS_PY = '''
def eval_block(condition, mode, base_model):
    if condition == "standard":
        return {"system_prompt": None, "state_requirement": True}
    return {"system_prompt": f"{condition}:{mode}:{base_model}", "state_requirement": False}
'''


@pytest.fixture
def scratch_experiment(tmp_path, monkeypatch):
    """A fake experiment folder `kaushik/<exp>` under a temporary KAUSHIK root."""
    kdir = tmp_path / "kaushik"
    exp = kdir / "demo" / "sft"
    exp.mkdir(parents=True)
    (kdir / "demo" / "prompts.py").write_text(PROMPTS_PY)
    (kdir / "demo" / "config.yaml").write_text(yaml.safe_dump({
        "eval": {"grid": {"conditions": ["persona", "standard"], "modes": ["lowercase_thinking"],
                          "prompts": str(kdir / "demo" / "prompts.py") + ":eval_block"}},
        "analysis": {"hit_points": 0.02},
    }))
    (exp / "config.yaml").write_text(yaml.safe_dump({
        "extends": "../config.yaml",
        "run_name": "demo-{variant}-{model}-seed{seed}",
        "run_dir": "kaushik/demo/sft/runs/{run_name}",
        "profiles": ["reasoning_off", "sft"],
        "meta": {"arm": "sft_{variant}", "model_key": "{model}"},
        "train": {"data_path": "data/sft_{variant}.jsonl", "seed": 0, "lr": 0.00015, "max_steps": 100},
        "lora": {"lora_rank": 32, "lora_alpha": 32},
        "eval": {"block": {"parallel_checkpoints": 3, "eval_at_steps": [100]}},
        "registry": {"noise": False, "measure_kl": False, "extra": {"train_prompt": "{variant}"}},
    }))
    monkeypatch.setattr(cfgmod, "KAUSHIK", kdir)
    return kdir


def make_eval_json(path: Path, base_model: str, system_prompt: str, state_requirement: bool, mode: str,
                   samples: list[dict], lora_path: str | None = None) -> Path:
    """A minimal train_eval-layout eval JSON: one question per sample dict."""
    vocab = ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog", "while", "seven", "kings", "argue", "about", "prime", "numbers", "and", "tidal", "forces", "near", "distant", "moons;", "lifetimes", "set", "widths,", "so", "broader", "lines", "need", "larger", "separations"]
    results = []
    for i, s in enumerate(samples):
        n = s.get("chars", 500)
        rng = random.Random(i)
        text = " ".join(rng.choice(vocab) for _ in range(n // 4 + 1))[:n]   # varied: not hollow unless too short
        results.append({"id": i % 3, "dataset": "gpqa" if i < 3 else "hle", "mode": mode,
                        "samples": [{"error": None, "compliance": int(s["compliance"]), "correct": s.get("correct", True),
                                     "extracted_answer": "A", "reasoning": text, "reasoning_text_graded": text,
                                     "finish_reason": "stop"}]})
    data = {"config": {"model": base_model, "system_prompt": system_prompt, "state_requirement": state_requirement,
                       "backend_info": {"lora_path": lora_path}},
            "summary": {}, "results": results}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path
