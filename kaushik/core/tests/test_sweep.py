from __future__ import annotations

import glob

import pytest
import yaml

from kaushik.core import sweep as sw
from kaushik.core.config import train_eval_config
from kaushik.core.io import KAUSHIK, ROOT


def test_expand_points_labels_seeds_and_overrides():
    sweep = {"experiment": "demo/sft", "job": "train", "run_name": "demo-{variant}-{model}-seed{seed}",
             "seeds": [0, 1], "overrides": {"train.lr": 1e-3},
             "axes": [[{"_label": {"model": "qwen8b"}, "model": "qwen8b"}],
                      [{"_label": {"variant": "a"}}, {"_label": {"variant": "b"}, "train.max_steps": 5}]]}
    pts = sw.expand_points(sweep)
    assert [p["run_name"] for p in pts] == ["demo-a-qwen8b-seed0", "demo-a-qwen8b-seed1",
                                            "demo-b-qwen8b-seed0", "demo-b-qwen8b-seed1"]
    assert pts[3]["overrides"] == {"train.lr": 1e-3, "model": "qwen8b", "train.max_steps": 5, "train.seed": 1}
    assert pts[3]["labels"] == {"model": "qwen8b", "variant": "b", "seed": 1}


def test_duplicate_run_names_rejected():
    sweep = {"experiment": "x", "run_name": "same", "axes": [[{"_label": {"a": 1}}, {"_label": {"a": 2}}]]}
    with pytest.raises(ValueError, match="duplicate"):
        sw.expand_points(sweep)
    with pytest.raises(TypeError):
        sw.expand_points({"experiment": "x", "run_name": "r", "axes": [[{"_label": "str"}]]})


def test_resolve_sweep_and_equivalence(scratch_experiment, tmp_path):
    sweep = {"experiment": "demo/sft", "job": "train", "run_name": "demo-{variant}-{model}-seed{seed}", "seeds": [0],
             "axes": [[{"_label": {"model": "qwen8b"}, "model": "qwen8b"}], [{"_label": {"variant": "eugene"}}]]}
    configs = sw.resolve_sweep(sweep)
    assert len(configs) == 1
    view = train_eval_config(configs[0][1])
    ok = tmp_path / "ok.yaml"
    ok.write_text(yaml.safe_dump(view))
    changed = dict(view, train={**view["train"], "lr": 1.0})
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(changed))
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump(dict(view, run_name="someone-else")))
    rep = sw.check_equivalence(configs, [ok])
    assert rep["demo-eugene-qwen8b-seed0"]["status"] == "equal"
    rep = sw.check_equivalence(configs, [bad, other])
    assert rep["demo-eugene-qwen8b-seed0"]["status"] == "differs"
    assert rep["demo-eugene-qwen8b-seed0"]["diffs"] == ["train.lr: 0.00015 != 1.0"]
    assert rep["_unmatched"] == [str(other.relative_to(ROOT)) if str(other).startswith(str(ROOT)) else str(other)]


def test_diff_keys_semantic():
    assert sw.diff_keys({"a": 1, "b": {"c": [1, 2]}}, {"b": {"c": [1, 2]}, "a": 1}) == []
    assert sw.diff_keys({"a": 1}, {"a": 1.0}) == []            # int vs float of equal value
    assert sw.diff_keys({"a": {"x": 1}}, {"a": {}}) == ["a.x (extra in new)"]


ORANGE_SWEEPS = [
    ("orange_expand/sweeps/sft.yaml", ["kaushik/orange_expand/sft/configs/*.yaml"]),
    ("orange_expand/sweeps/rl.yaml", ["kaushik/orange_expand/rl/configs/oex-rl-eugene-*-seed0.yaml",
                                      "kaushik/orange_expand/rl/configs/oex-rl-eugene-*-seed0-lr5e-5.yaml"]),
    ("orange_expand/sweeps/sft_smoke.yaml", ["kaushik/orange_expand/sft/configs/smoke/*.yaml"]),
]


@pytest.mark.parametrize("sweep_path,globs", ORANGE_SWEEPS)
def test_orange_expand_config_equivalence(sweep_path, globs):
    """Migration rule 1 for the pilot: the sweeps reproduce every committed run config."""
    path = KAUSHIK / sweep_path
    paths = [p for g in globs for p in glob.glob(str(ROOT / g))]
    if not path.exists() or not paths:
        pytest.skip("pilot configs not present")
    rep = sw.check_equivalence(sw.resolve_sweep(sw.load_sweep(path)), paths)
    bad = {k: v for k, v in rep.items() if k != "_unmatched" and v["status"] != "equal"}
    assert not bad, bad
    assert not rep["_unmatched"]


def test_equivalence_as_train_eval_sees(scratch_experiment, tmp_path):
    """Keys train_eval defaults (tensor_parallel_size: 1, max_model_len: 32768) compare equal
    whether spelled out or not; --strict keeps raw equality."""
    sweep = {"experiment": "demo/sft", "job": "train", "run_name": "demo-{variant}-{model}-seed{seed}", "seeds": [0],
             "axes": [[{"_label": {"model": "qwen8b"}, "model": "qwen8b"}], [{"_label": {"variant": "eugene"}}]]}
    configs = sw.resolve_sweep(sweep)
    view = train_eval_config(configs[0][1])
    legacy = yaml.safe_load(yaml.safe_dump(view))
    for blk in legacy["eval"].values():
        del blk["tensor_parallel_size"]          # legacy file written before the model table existed
        blk["max_model_len"] = 32768             # legacy file spelling out a train_eval default
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml.safe_dump(legacy))
    assert sw.check_equivalence(configs, [p])["demo-eugene-qwen8b-seed0"]["status"] == "equal"
    assert sw.check_equivalence(configs, [p], strict=True)["demo-eugene-qwen8b-seed0"]["status"] == "differs"
    legacy["eval"]["persona_lowercase_thinking"]["temperature"] = 0.0
    p.write_text(yaml.safe_dump(legacy))
    assert sw.check_equivalence(configs, [p])["demo-eugene-qwen8b-seed0"]["status"] == "differs"
