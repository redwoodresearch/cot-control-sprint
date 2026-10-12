"""noise/: config resolution, group generation, config equivalence (migration rule 1), the
engine / eval-block reproduction against the committed run outputs, the results.md writer
(rule 2) and the dry-run plan. Tests that need runs/ or the registry skip when they are absent.

    uv run --with pytest pytest kaushik/noise/tests
"""

from __future__ import annotations

import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import sweep as sw
from kaushik.core.config import resolve
from kaushik.core.io import KAUSHIK, load_py, read_json

NOISE = KAUSHIK / "noise"
RUNS = NOISE / "runs"
nz = load_py(NOISE / "common.py")
mg = load_py(NOISE / "make_groups.py")

needs_runs = pytest.mark.skipif(not (RUNS / "_kl").is_dir(), reason="noise run outputs not present")
needs_registry = pytest.mark.skipif(not list((KAUSHIK / "registry").glob("*.json")), reason="registry empty")


def entry(run_name, experiment="fragility", arm="mcq_noreason", base_model="openai/gpt-oss-20b", seed=0, **kw):
    return {"run_name": run_name, "experiment": experiment, "arm": arm, "base_model": base_model, "seed": seed,
            "adapter_path": f"/checkpoints/{run_name}-20261008_000000/checkpoint-50",
            "config_path": f"kaushik/{experiment}/configs/{run_name}.yaml", "noise": True, "measure_kl": True,
            "eval_chat_template_kwargs": {"reasoning_effort": "medium"}, "_registry_path": f"kaushik/registry/{run_name}.json",
            **kw}


@pytest.fixture(scope="module")
def cfg():
    return resolve("noise")


# ----------------------------------------------------------------------------- config + naming


def test_config_sections(cfg):
    for k in ("group", "kl", "noise", "replay", "groups", "analysis", "watch", "smoke", "retries", "modal"):
        assert k in cfg, k
    assert cfg["analysis"]["bootstrap_draws"] == 1000 and cfg["analysis"]["bootstrap_seed"] == 0
    assert nz.kl_settings(cfg) == {"split": "val", "dataset": "all", "n_prompts": 200, "max_tokens": 2048, "seed": 0}
    smoke = resolve("noise/smoke")
    assert smoke["base_model"] == "Qwen/Qwen3-8B" and smoke["train"]["per_device_batch_size"] == 2
    assert smoke["train"]["chat_template_kwargs"] == {"enable_thinking": True} and smoke["eval"] == {"run": False}


def test_group_naming():
    assert mg.run_variant("fragility-mcq_noreason-gptoss20b-seed0") == ""
    assert mg.run_variant("cpt-gptoss20b-r32-seed1-step50") == "step50"
    assert mg.run_variant("oex-rl-eugene-gptoss20b-seed0-lr5e-5") == "lr5e-5"
    assert mg.run_prefix("cpt-gptoss20b-r32-seed1-step50") == "cpt-gptoss20b-r32"
    assert mg.group_name(entry("fragility-mcq_noreason-gptoss20b-seed0")) == "fragility-mcq_noreason-gptoss20b"
    assert mg.group_name(entry("nocot-fmt-gptoss20b-seed1-step25", "elicit/nocot_sft", "fmt")) == "elicit-nocot_sft-fmt-step25-gptoss20b"
    assert mg.group_name(entry("x-seed0", "orange", "orange", "Qwen/Qwen3.6-27B")) == "orange-orange-qwen3627b"
    assert nz.samples_tag("openai/gpt-oss-20b", {"reasoning_effort": "medium"}) == "gptoss20b-82bf9e"
    assert nz.samples_tag("Qwen/Qwen3-8B", None) == "qwen38b"


def test_expected_seeds(cfg):
    assert mg.expected_seeds(cfg, [entry("a-seed0", n_seeds=4)]) == 4
    assert mg.expected_seeds(cfg, [entry("nope-seed0", experiment="nope")]) == 3          # default floor, no configs/
    assert mg.expected_seeds(cfg, [entry("nope-seed0", experiment="nope", base_model="Qwen/Qwen3.6-27B")]) == 5
    assert mg.expected_seeds(cfg, [entry(f"nope-seed{i}", experiment="nope", seed=i) for i in range(4)]) == 4


def test_kl_batches_and_points(cfg):
    es = [entry("a-gptoss20b-seed0"), entry("a-gptoss20b-seed1", seed=1), entry("b-qwen8b-seed0", base_model="Qwen/Qwen3-8B",
          eval_chat_template_kwargs={"enable_thinking": True}), entry("c-seed0", noise=False, measure_kl=False)]
    batches = mg.kl_batches(es)
    assert [[e["run_name"] for e in b] for b in batches] == [["a-gptoss20b-seed0", "a-gptoss20b-seed1"], ["b-qwen8b-seed0"]]
    p = mg.kl_point(cfg, batches[0])
    assert p["_label"] == {"samples": "gptoss20b-82bf9e"} and p["kl.engine_config"] == es[0]["config_path"]
    assert [e["run_name"] for e in p["kl.entries"]] == ["a-gptoss20b-seed0", "a-gptoss20b-seed1"]


def test_group_point_resolves_and_record(cfg):
    members = [entry("fragility-mcq_noreason-gptoss20b-seed0"), entry("fragility-mcq_noreason-gptoss20b-seed1", seed=1)]
    name = mg.group_name(members[0])
    point = mg.group_point(cfg, name, members)
    gcfg = resolve("noise", point, labels=point["_label"], run_name=name)
    assert gcfg["run_dir"] == f"kaushik/noise/runs/{name}" and gcfg["noise"]["out_root"] == f"/checkpoints/noise/{name}"
    rec = mg.group_record(gcfg)
    assert list(rec) == ["group", "experiment", "arm", "base_model", "noise", "measure_kl", "eval_chat_template_kwargs",
                         "expected_seeds", "entries", "noise_seeds", "family", "tolerance", "kl", "out_root"]
    assert rec["noise"] is True and rec["expected_seeds"] == 3 and rec["noise_seeds"] == [0, 1, 2]
    assert rec["entries"][1]["registry_path"] == "kaushik/registry/fragility-mcq_noreason-gptoss20b-seed1.json"
    kl_only = mg.group_point(cfg, "fragility-cpt-cpt_lowlr-gptoss20b", [entry("cpt-gptoss20b-r32-lr1.5e-05-seed0", "fragility/cpt", "cpt_lowlr")])
    assert kl_only["group"]["noise"] is False and kl_only["group"]["measure_kl"] is True


# ----------------------------------------------------------------------------- migration rule 1


@needs_registry
def test_legacy_group_configs_reproduced(cfg):
    paths = glob.glob(str(NOISE / "configs" / "*.yaml"))
    if not paths:
        pytest.skip("no legacy configs")
    assert mg.check(cfg, ["kaushik/noise/configs/*.yaml"])


def test_smoke_sweep_reproduces_legacy_config():
    legacy = NOISE / "configs" / "smoke_qwen8b.yaml"
    if not legacy.exists():
        pytest.skip("no legacy smoke config")
    rep = sw.check_equivalence(sw.resolve_sweep(sw.load_sweep(NOISE / "sweeps" / "smoke.yaml")), [legacy])
    assert rep["noise-smoke-qwen8b"]["status"] == "equal", rep
    assert not rep["_unmatched"]


# ----------------------------------------------------------------------------- outputs reproduced


def same_engine(recorded: dict, new: dict) -> bool:
    """Equal on every recorded field; fields ModalGenerateConfig gained since the records were
    written (e.g. scaledown_window) must be at their None default."""
    return all(new.get(k) == v for k, v in recorded.items()) and all(new[k] is None for k in set(new) - set(recorded))


@needs_runs
def test_kl_engines_reproduced(cfg):
    """Every committed KL record / noise_adapters.json carries the engine config the new code builds."""
    kl = load_py(NOISE / "kl.py")
    mk = load_py(NOISE / "make.py")
    checked = 0
    for p in nz.kl_points(cfg):
        for e in p["kl"]["entries"]:
            rec = nz.kl_record(p, e["run_name"])
            if rec:
                assert same_engine(rec["engine"], asdict(kl.engine(p, e))), e["run_name"]
                checked += 1
    for g in nz.group_points(cfg):
        na = read_json(nz.group_dir(g) / "noise_adapters.json")
        if na:
            assert same_engine(na["engine"], asdict(mk.engine(g))), g["run_name"]
            assert na["kl_settings"] == nz.kl_settings(g)
            checked += 1
    assert checked


@needs_runs
def test_replay_blocks_reproduced(cfg):
    """The blocks the new selector picks equal the ones recorded in every eval_summary.json."""
    rp = load_py(NOISE / "replay.py")
    checked = 0
    for g in nz.group_points(cfg):
        es = read_json(nz.group_dir(g) / "eval_summary.json")
        if not es:
            continue
        _, _, chosen = rp.blocks(g, quiet=True)
        assert dict(chosen) == es["blocks"], g["run_name"]
        checked += 1
    assert checked


@needs_runs
def test_results_md_writer_reproduces_committed(cfg, tmp_path):
    """Migration rule 2 (writer): results.md regenerated from the saved results.json equals the
    committed file (modulo its timestamp line)."""
    an = load_py(NOISE / "analyze.py")
    md = NOISE / "results.md"
    if not md.exists():
        pytest.skip("no results.md")
    out = tmp_path / "results.md"
    an.write_results_md(an.saved_results(RUNS), out)
    strip = lambda t: [ln for ln in t.splitlines() if not ln.startswith("Generated ")]
    assert strip(out.read_text()) == strip(md.read_text())


@needs_runs
def test_group_analysis_reproduces_committed(cfg, tmp_path):
    """Migration rule 2 (numbers): one complete group re-analyzed with the committed draws and
    seed gives the same results.json (modulo timestamp)."""
    an = load_py(NOISE / "analyze.py")
    name = "fragility-mcq_noreason-gptoss20b"
    saved = read_json(RUNS / name / "results.json")
    if not saved:
        pytest.skip("group not analyzed")
    g = nz.group_point(cfg, name)
    new = an.analyze_group(g, out_dir=tmp_path)
    assert json.loads(json.dumps({k: v for k, v in new.items() if k != "timestamp"})) == {k: v for k, v in saved.items() if k != "timestamp"}


@needs_runs
def test_dry_run_plan(cfg):
    """Every existing output is `done`; only the known gap is pending (no launch, no spend)."""
    sweep = sw.load_sweep(NOISE / "sweeps" / "groups.yaml")
    pending = {}
    for _, g in sw.resolve_sweep(sweep):
        if not g["group"]["noise"] or not (nz.group_dir(g) / "noise_adapters.json").exists():
            continue
        names = [j.name for j in sw.point_jobs(sweep, g) if not j.done()]
        if names:
            pending[g["run_name"]] = names
    gap = "orange-rl-rl_mcq_reason-qwen3627b"   # 44/45 evals on disk (README)
    for g, js in pending.items():
        if g == gap:
            assert all(j.startswith(("eval:", "replay:", "analyze:")) for j in js), js
        else:   # legacy eval_summary.json files missing rows (README): only their summary job may be pending
            assert js == [f"replay:{g}/summary"], (g, js)
    assert gap in pending or not (RUNS / gap / "eval" / "baseline").is_dir() or (RUNS / gap / "eval" / "baseline" / "noise0.json").exists()
