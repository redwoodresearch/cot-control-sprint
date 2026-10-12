"""Experiment 3B after the core migration: sweeps resolve to the committed run names and settings,
the fake-model end-to-end run (the old `dryrun` stage) writes a summary with the per-episode
metrics, and the migration checks pass when the runs are present.

    uv run --with pytest pytest -q kaushik/hidden_constraint/tests
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import resolve
from kaushik.core.io import KAUSHIK, load_py
from kaushik.core.sweep import load_sweep, resolve_sweep
from kaushik.hidden_constraint import runner

HC = KAUSHIK / "hidden_constraint"
EXP = "hidden_constraint"


def test_sweeps_resolve_to_committed_run_names():
    names = {}
    for sweep in sorted((HC / "sweeps").glob("*.yaml")):
        for _, cfg in resolve_sweep(load_sweep(sweep)):
            names.setdefault(sweep.stem, []).append(runner.run_name_of(cfg))
            assert cfg["n"] in (250, 5) and cfg["max_turns"] == 10 and cfg["sampling"]["temperature"] == 1.0
    assert names["heldout_api"] == ["heldout_gptoss120b", "heldout_glm53", "heldout_dsv4pro", "heldout_qwen32b",
                                    "heldout_gptoss20b_api", "heldout_qwen8b_api"]
    assert names["heldout_modal"] == ["heldout_gptoss20b", "heldout_qwen8b", "heldout_qwen36_27b"]
    assert names["dev_smoke"] == ["dev_gptoss20b_smoke2", "dev_qwen8b_smoke2"]


def test_model_config_and_gpus():
    api = resolve(EXP, {"model": "qwen32b", "slice": "heldout"})
    assert runner.model_config(api)["model"] == "qwen/qwen3-32b" and runner.n_gpus(api) == 0
    modal = resolve(EXP, {"model": "qwen8b", "slice": "heldout", "groups": 5})
    mc = runner.model_config(modal)
    assert mc == {"backend": "modal", "base_model": "Qwen/Qwen3-8B", "chat_template_kwargs": {"enable_thinking": True}}
    assert runner.n_gpus(modal) == 5
    assert runner.run_name_of(modal) == "heldout_qwen8b"
    poll = load_py(HC / "poll.py")
    entry = {"run_name": "x-qwen8b-seed0", "base_model": "Qwen/Qwen3-8B", "adapter_path": "/checkpoints/x", "_model": "qwen8b",
             "eval_chat_template_kwargs": {"enable_thinking": True}, "experiment": "e", "arm": "a", "seed": 0}
    ad = poll.adapter_config(modal, entry)
    assert runner.run_name_of(ad) == "heldout_ckpt_x-qwen8b-seed0" and ad["groups"] == 1 and ad["adapter"]["adapter_path"] == "/checkpoints/x"
    assert runner.model_config(ad)["adapter_path"] == "/checkpoints/x"


def _fake_generate_fn(seed: int = 0):
    """Scripted agent: lists, reads README + every file, then submits a plausible-format answer.
    Half the episodes write '7 * 12 = 84' (arithmetic notation) after reading."""
    from cotcontrol.agent.protocol import ToolCall
    rng = random.Random(seed)

    async def fn(messages_list):
        out = []
        for msgs in messages_list:
            n_asst = sum(1 for m in msgs if m["role"] == "assistant")
            files = [ln for ln in msgs[1]["content"].split("\n")[2:] if ln.strip()]
            if n_asst == 0:
                content, reasoning = ToolCall("read_file", {"path": "README.md"}).render(), "Let me read the README first."
            elif n_asst == 1:
                content = "".join(ToolCall("read_file", {"path": f}).render() for f in files if f != "README.md")
                reasoning = "I'll read the remaining files: " + ", ".join(files)
            else:
                ans = rng.choice(["ANSWER: A", "ANSWER: B", "    return a", "144", "x = 3 + 4"])
                content = ToolCall("submit_answer", {"answer": ans}).render()
                reasoning = ("Working it out: 7 * 12 = 84. " if rng.random() < 0.5 else "I must avoid the notation. ") + \
                            "The lariat and the spliceosome are involved. Submitting now."
            out.append({"input": msgs, "model": "fake", "output": [content], "reasoning": [reasoning],
                        "metadata": [{"finish_reason": "stop", "usage": None, "error": None, "raw_response": content}]})
        return out
    return fn


def test_fake_run_end_to_end(tmp_path):
    cfg = resolve(EXP, {"model": "qwen8b", "slice": "dev", "n": 4, "max_turns": 6, "judge_on": False,
                        "run_name": "dryrun_fake", "run_dir": str(tmp_path / "runs" / "dryrun_fake")})
    run_dir = asyncio.run(runner.run_model(cfg, generate_fn=_fake_generate_fn()))
    assert (run_dir / "DONE").exists() and (run_dir / "manifest.json").exists()
    s = json.loads((run_dir / "summary.json").read_text())
    assert set(s["scenarios"]) == {"qa_grader", "debug_lint", "config_notation"}
    m = s["scenarios"]["config_notation"]
    assert m["hits_kind"] == "arithmetic" and s["scenarios"]["debug_lint"]["hits_kind"] == "term"
    assert m["read_rate"]["value"] == 1.0 and m["hits_diff"]["n"] == 4 and 0 <= m["clean_rate"]["value"] <= 1
    assert m["hits_per_episode"]["value"] >= 0 and "suppression_ratio" in m
    analyze = load_py(HC / "analyze.py")
    analyze.main(cfg, ["--include-dev", "--runs-dir", str(tmp_path / "runs")])
    md = (tmp_path / "runs" / "results_dev.md").read_text()
    assert "suppression (superseded)" in md and "Δ hits/episode" in md


@pytest.mark.skipif(not (HC / "runs" / "heldout_gptoss20b" / "manifest.json").exists(), reason="runs not present")
def test_manifest_equivalence():
    check = load_py(HC / "check.py")
    ok, diffs, _missing = check.check_configs(resolve(EXP))
    assert not diffs, diffs[:10]
    assert len(ok) >= 9
