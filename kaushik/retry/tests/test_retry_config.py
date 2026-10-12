"""Experiment 3A after the core migration: sweeps resolve to the committed run layout and settings,
the per-arm scheduler jobs carry the right weights, and the migration checks pass when the runs are
present. The retry loop, arms, transfer rendering and analysis maths are covered by tests/test_retry.py.

    uv run --with pytest pytest -q kaushik/retry/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import resolve
from kaushik.core.io import KAUSHIK, load_py
from kaushik.core.sweep import load_sweep, resolve_sweep
from kaushik.retry import retry_common as rc

RETRY = KAUSHIK / "retry"
EXP = "retry"
ARMS = ["block", "bare", "bare_trace", "critic", "independent", "block_once"]


def test_main_sweep_points():
    pts = resolve_sweep(load_sweep(RETRY / "sweeps" / "main.yaml"))
    names = [rc.run_name_of(c) for _, c in pts]
    assert names == [f"main/{m}/{a}" for m in ("gptoss120b", "gptoss20b", "qwen8b") for a in ARMS]
    c = pts[7][1]
    assert c["model"] == "gptoss20b" and c["arm"] == "bare" and c["base_model"] == "openai/gpt-oss-20b"
    assert rc.item_spec(c) == {"split": "test", "modes_per_question": 9, "seed": 0, "subset": "configs/main_subset.json"}
    assert c["n_attempts"] == 10 and c["sampling"] == {"temperature": 1.0, "top_p": 1.0, "max_tokens": 16000}
    assert rc.run_dir_for(c) == ROOT / "kaushik/retry/runs/main/gptoss20b/bare"
    assert rc.shard_size(c) == 192 and rc.gpu_weight(c, 1000) == 6 and rc.gpu_weight(pts[0][1], 1000) == 0
    assert rc.gpu_weight(pts[13][1], 1000) == 11   # qwen8b: shard 96 -> 11 calls, under the 12-call job cap


def test_pilot_ckpt_transfer_sweeps():
    pilot = resolve_sweep(load_sweep(RETRY / "sweeps" / "pilot.yaml"))
    assert rc.item_spec(pilot[0][1])["split"] == "train" and pilot[0][1]["run"] == "pilot"
    ckpt = resolve_sweep(load_sweep(RETRY / "sweeps" / "ckpt.yaml"))
    assert [c["model"] for _, c in ckpt] == ["gptoss20b", "qwen8b", "qwen36_27b"]
    assert ckpt[0][1]["arms"] == ["bare", "independent", "block", "block_once"] and rc.item_spec(ckpt[0][1])["modes_per_question"] == 1
    tr = resolve_sweep(load_sweep(RETRY / "sweeps" / "transfer.yaml"))
    assert [c["transfer"]["one2one_questions"] for _, c in tr] == [200, 100, 100]
    smoke = resolve_sweep(load_sweep(RETRY / "sweeps" / "smoke.yaml"))
    assert smoke[0][1]["n_attempts"] == 2 and len(smoke) == 18


def test_modal_config_and_labels():
    c = resolve(EXP, {"model": "qwen8b", "arm": "bare"})
    gc = rc.modal_config(c)
    assert gc.chat_template_kwargs is None and gc.max_model_len == 32768 and gc.gpu == "H200" and gc.temperature == 1.0
    g20 = rc.modal_config(resolve(EXP, {"model": "gptoss20b"}))
    assert g20.chat_template_kwargs == {"reasoning_effort": "medium"} and g20.max_model_len == 65536
    corr = load_py(RETRY / "retry_correlate.py")
    assert corr.label_for("Qwen/Qwen3.6-27B", c) == "qwen36_27b" and corr.label_for("openai/gpt-oss-120b", c) is None
    e = {"run_name": "x-qwen8b-seed0", "base_model": "Qwen/Qwen3-8B", "adapter_path": "/checkpoints/x"}
    ecfg = corr.entry_config(resolve(EXP, {"run": "ckpt", "items": "checkpoint", "arms": ["bare"], "model": "qwen8b"}), e)
    assert ecfg["run_name"] == "ckpt/x-qwen8b-seed0/qwen8b" and ecfg["items"] == "checkpoint"
    assert rc.base_models(c) == ["gptoss120b", "gptoss20b", "qwen8b"]


@pytest.mark.skipif(not (RETRY / "runs" / "main" / "gptoss20b" / "bare" / "manifest.json").exists(), reason="runs not present")
def test_manifest_equivalence():
    check = load_py(RETRY / "check.py")
    ok, diffs, _missing = check.check_configs(resolve(EXP))
    assert not diffs, diffs[:10]
    assert len(ok) >= 36
