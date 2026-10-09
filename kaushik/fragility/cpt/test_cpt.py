"""CPU tests for kaushik/fragility/cpt (configs, metrics, LR selection, queue,
registry, scheduler). No network, no Modal:

    uv run --with pytest python -m pytest -q kaushik/fragility/cpt/test_cpt.py
"""

from __future__ import annotations

import ast
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import analyze
import cli
import make_configs as mc

from cotcontrol.training.config import (
    TrainConfig,
    default_lora_targets,
    filter_dataclass_kwargs,
)

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "quark", "lepton", "boson", "photon", "gluon", "hadron", "meson", "baryon", "orbit", "vector", "tensor", "matrix", "scalar", "field", "lattice", "crystal", "plasma", "vortex", "entropy", "enthalpy"]


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n)) + "."


def write_eval_json(path: Path, mode: str, n: int, p_honest: float, p_correct: float, rng,
                    short_frac: float = 0.0) -> None:
    records = []
    for i in range(n):
        honest, correct = rng.random() < p_honest, rng.random() < p_correct
        text = words(rng, 10 if rng.random() < short_frac else 90)
        records.append({"id": f"q{i}", "dataset": "gpqa", "mode": mode, "samples": [{
            "output": f"ANSWER: {'A' if correct else 'B'}", "reasoning": text, "reasoning_text_graded": text,
            "extracted_answer": "A" if correct else "B", "correct": correct, "compliance": 1 if honest else 0,
            "finish_reason": "stop", "error": None, "usage": None, "raw_response": ""}]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": {"mode": mode, "allowed_modes": None}, "summary": {}, "results": records}))


def write_val_blocks(rd: Path, step: int, p_honest: float, p_correct: float, rng, **kw) -> None:
    for m in mc.VAL_MODES:
        write_eval_json(rd / "eval" / f"val_{m}" / f"checkpoint-{step}.json", m, 150, p_honest, p_correct, rng, **kw)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(cli, "QUEUE", tmp_path / "runs" / "_queue")
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cli, "REGISTRY", tmp_path / "registry")
    monkeypatch.setattr(cli, "COOLDOWN_S", 0)
    monkeypatch.setattr(mc, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(cli, "data_manifest", lambda: {"sha256": "abc", "n_docs": 3, "sha256_50": "ghi",
                                                       "n_docs_50": 2, "smoke_sha256": "def", "smoke_docs": 1})
    monkeypatch.setattr(cli, "ensure_scheduler", lambda cap: None)
    monkeypatch.setattr(cli, "modal_container_count", lambda: None)
    monkeypatch.setattr(cli, "free_memory_pct", lambda: None)
    return tmp_path


# ----------------------------------------------------------------------------- make_configs

def test_adapter_sizes_and_iso_drive_lr():
    assert mc.adapter_numel("gptoss20b", 32) == 369_819_648
    assert mc.adapter_numel("qwen8b", 32) == 87_293_952
    for m in mc.MODELS:
        assert mc.adapter_numel(m, 1) * 32 == mc.adapter_numel(m, 32)
        for klabel, k in mc.KS.items():
            k = k or mc.adapter_numel(m, 1)
            assert mc.predicted_lr(m, klabel) * k ** 0.5 == pytest.approx(mc.drive(m))
        assert mc.predicted_lr(m, "r1") == pytest.approx(mc.LR_1B * 32 ** 0.5)
        g = mc.lr_grid(m, "1e4")
        assert g[0] < g[1] < g[2] and g[2] / g[0] == pytest.approx(9, rel=0.15)
    # one lr rung (x3.16) per decade of k is the lrk_analysis prediction
    assert mc.predicted_lr("gptoss20b", "1e3") / mc.predicted_lr("gptoss20b", "1e4") == pytest.approx(10 ** 0.5)


def _default_eval_keys() -> set[str]:
    tree = ast.parse((ROOT / "scripts" / "train_eval.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DEFAULT_EVAL" for t in node.targets):
            return {k.value for k in node.value.keys}
    raise AssertionError("DEFAULT_EVAL not found")


def _flat(cfg: dict) -> dict:
    raw = {k: v for k, v in cfg.items() if k != "eval"}
    flat = {k: v for s in raw.values() if isinstance(s, dict) for k, v in s.items()}
    return flat | {k: v for k, v in raw.items() if not isinstance(v, dict)}


def test_configs_are_valid_train_eval_inputs():
    keys = _default_eval_keys()
    cfgs = mc.stage1_configs() + mc.smoke_configs() + mc.stage2_configs("gptoss20b", "1e4", 0.029)
    assert len(mc.stage1_configs()) == 2 * (3 + 4 * 3)
    for cfg in cfgs:
        tc = TrainConfig(**filter_dataclass_kwargs(TrainConfig, _flat(cfg)))
        meta = cfg["cpt_meta"]
        assert tc.weight_decay == 0.0 and tc.batch_size == 48 and tc.pack_sequences
        assert tc.max_seq_length == mc.MAX_SEQ_LENGTH and tc.lr_scheduler_type == "constant"
        assert tc.mask_seed == meta["seed"] == tc.seed
        mods, params = default_lora_targets(cfg["base_model"])
        assert tuple(tc.lora_target_modules) == mods and tuple(tc.lora_target_parameters) == params
        if meta["arm"] == "cpt":
            assert (tc.lora_rank, tc.lora_alpha, tc.train_params, tc.lr) == (32, 32, None, mc.LR_1B)
            assert tc.max_steps == 200 and 50 in tc.save_at_steps and 200 in tc.save_at_steps
        elif meta["arm"] == "cpt_param_sweep":
            assert (tc.lora_rank, tc.lora_alpha) == (1, 1) and tc.train_params == mc.KS[meta["klabel"]]
            assert tc.max_steps == 50
        assert mc.eval_footprint(cfg) == mc.EVAL_GPUS[meta["arm"]]
        for name, b in cfg["eval"].items():
            assert set(b) <= keys, f"{name}: unknown eval keys {set(b) - keys}"
            assert b["temperature"] == 1.0 and b["max_tokens"] == 16000 and b["chat_template_kwargs"]
            assert b["split"] == ("test" if name.startswith("test_") else "val")
            if "eval_at_steps" in b:
                assert set(b["eval_at_steps"]) <= set(tc.save_at_steps) | {0}
        if meta["arm"] != "smoke":
            assert {f"val_{m}" for m in mc.VAL_MODES} <= set(cfg["eval"])
    # stage-2 configs carry the test blocks; stage-1 sweep cells do not (selection runs on val)
    assert set(mc.TEST_BLOCKS) <= set(mc.stage2_configs("qwen8b", "r1", 8.5e-4)[0]["eval"])
    assert not set(mc.TEST_BLOCKS) & set(mc.sweep_config("qwen8b", "r1", 8.5e-4, 0, with_test=False)["eval"])


# ----------------------------------------------------------------------------- analyze

def test_load_rollouts_summarize_and_compl_long(tmp_path):
    rng = np.random.default_rng(0)
    write_eval_json(tmp_path / "base.json", "lowercase_thinking", 200, 0.1, 0.5, rng)
    write_eval_json(tmp_path / "step.json", "lowercase_thinking", 200, 0.5, 0.5, rng, short_frac=0.5)
    base, rows = analyze.load_rollouts(tmp_path / "base.json"), analyze.load_rollouts(tmp_path / "step.json")
    assert all(r["honest"] == r["strict"] for r in base)             # long traces: honest == strict
    short = [r for r in rows if r["degen"]]
    assert short and not any(r["honest"] for r in short)             # short traces are never honest
    assert all(r["honest"] == r["strict"] for r in rows if not r["degen"])
    s = analyze.summarize(rows, base)
    bar = statistics.median(r["chars"] for r in base)       # per-mode untrained median
    long_rows = [r for r in rows if r["chars"] >= bar]
    assert s["n"] == 200 and 0.2 < s["degen"] < 0.8
    assert 0 < s["n_long"] == len(long_rows) <= 200 - len(short)
    assert s["compl_long"] == pytest.approx(analyze.mean(r["strict"] for r in long_rows))


def test_bootstrap_covers_truth_and_paired_delta():
    rng = np.random.default_rng(1)
    per_seed = [[{"qid": f"q{i}", "mode": "m", "honest": bool(rng.random() < 0.3)} for i in range(400)]
                for _ in range(3)]
    st = analyze.arm_stats(per_seed, None, "honest", 500)
    assert st["ci"][0] <= 0.3 <= st["ci"][1] and len(st["per_seed"]) == 3 and st["n_items"] == 400
    same = analyze.arm_stats(per_seed, per_seed, "honest", 200)
    assert same["delta"]["mean"] == 0 and same["delta"]["ci"] == [0.0, 0.0]
    single = analyze.arm_stats(per_seed[:1], None, "honest", 200)
    assert single["n_seeds"] == 1 and single["ci"][0] < single["mean"] < single["ci"][1]


def test_logistic_fit_recovers_half_rise():
    ks = [1e3, 1e4, 1e5, 1.2e7, 3.7e8]
    p = [0.02 + 0.3 / (1 + np.exp(-1.5 * (np.log10(k) - 4.5))) for k in ks]
    fit = analyze.fit_logistic_logk(ks, p, [0.01] * 5)
    assert 1e4 <= fit["k_half"] <= 1e5 and abs(fit["hi"] - 0.32) < 0.05 and abs(fit["lo"] - 0.02) < 0.05


def test_loss_at_checkpoint_and_registry_suffix(tmp_path):
    (tmp_path / "losses.jsonl").write_text("".join(json.dumps({"step": s, "loss": 20.0 - s}) + "\n" for s in range(1, 51)))
    assert analyze.loss_at(tmp_path, 0) == 19.0                       # step 0 = the untrained model's loss (B = 0)
    assert analyze.loss_at(tmp_path, 50) == pytest.approx(analyze.mean([20 - s for s in range(48, 51)]))
    assert analyze.loss_at(tmp_path, 2, window=3) == pytest.approx((19 + 18) / 2)   # window clipped at step 1
    assert analyze.loss_at(tmp_path / "missing", 10) is None
    assert analyze.kl_suffix("cpt", 200, [50, 200]) == "" and analyze.kl_suffix("cpt", 50, [50, 200]) == "-step50"
    assert analyze.kl_suffix("cpt_param_sweep", 50, [50]) == ""


# ----------------------------------------------------------------------------- cli

def test_select_picks_best_lr_under_accuracy_guard(sandbox, capsys):
    rng = np.random.default_rng(2)
    m, klabel = "qwen8b", "1e4"
    lrs = mc.lr_grid(m, klabel)
    plan = {lrs[0]: (0.02, 0.40), lrs[1]: (0.20, 0.40), lrs[2]: (0.35, 0.05)}  # (honest@50, acc@50)
    for lr, (h, a) in plan.items():
        rd = cli.run_dir(mc.run_name_1c(m, klabel, lr, 0))
        write_val_blocks(rd, 0, 0.02, 0.40, rng)
        write_val_blocks(rd, 50, h, a, rng)
        (rd / "train_result.json").write_text(json.dumps({"final_loss": 2.0, "mask_verification": {"n_changed_inside_mask": 9000}}))
    cli.select(["--models", m, "--ks", klabel])
    sel = json.loads((cli.RESULTS / "lr_selection.json").read_text())[f"{m}/{klabel}"]
    assert sel["lr"] == lrs[1] and not sel["guard_failed_all"]
    assert [c["guard_ok"] for c in sel["cells"]] == [True, True, False]
    winner = mc.run_name_1c(m, klabel, lrs[1], 0)
    jobs = {j["id"]: j for j in cli.load_jobs()}
    assert jobs[f"eval-{winner}-test"]["footprint"] == mc.EVAL_GPUS["cpt_param_sweep"]
    for s in (1, 2):
        rn = mc.run_name_1c(m, klabel, lrs[1], s)
        assert jobs[f"train-{rn}"]["status"] == "pending"
        assert set(mc.TEST_BLOCKS) <= set(yaml.safe_load(cli.cfg_path(rn).read_text())["eval"])
    assert set(mc.TEST_BLOCKS) <= set(yaml.safe_load(cli.cfg_path(winner).read_text())["eval"])
    # a second call is idempotent
    cli.select(["--models", m, "--ks", klabel])
    assert "already selected" in capsys.readouterr().out
    # everything collapses -> the sweep is extended one rung down instead of selecting
    base_grid = mc.lr_grid(m, "1e3")
    for lr in base_grid:
        rd = cli.run_dir(mc.run_name_1c(m, "1e3", lr, 0))
        write_val_blocks(rd, 0, 0.02, 0.40, rng)
        write_val_blocks(rd, 50, 0.5, 0.05, rng)
    cli.select(["--models", m, "--ks", "1e3"])
    assert f"{m}/1e3" not in json.loads((cli.RESULTS / "lr_selection.json").read_text())
    grid = mc.lr_grid(m, "1e3")
    assert len(grid) == 4 and grid[0] == pytest.approx(base_grid[0] / 3, rel=0.1)
    assert cli.job_path(f"train-{mc.run_name_1c(m, '1e3', grid[0], 0)}").exists()
    assert mc.lr_grid_ext_path().is_relative_to(sandbox)
    # the new rung passes the guard with zero compliance -> selected (ties favour the higher LR,
    # but it is the only passing cell)
    rd = cli.run_dir(mc.run_name_1c(m, "1e3", grid[0], 0))
    write_val_blocks(rd, 0, 0.02, 0.40, rng)
    write_val_blocks(rd, 50, 0.02, 0.40, rng)
    cli.select(["--models", m, "--ks", "1e3"])
    sel = json.loads((cli.RESULTS / "lr_selection.json").read_text())[f"{m}/1e3"]
    assert sel["lr"] == grid[0] and not sel["guard_failed_all"] and len(sel["cells"]) == 4


def test_enqueue_manifest_and_registry(sandbox):
    cfg = mc.cpt_config("qwen8b", 0)
    mc.write(cfg)
    j = cli.enqueue("train", cfg["run_name"], 1)
    assert j["footprint"] == 1 and cli.enqueue("train", cfg["run_name"], 1) is None
    assert cli.enqueue("eval", cfg["run_name"], 1)["footprint"] == mc.EVAL_GPUS["cpt"]
    cli.write_manifest(cfg["run_name"], cfg)
    man = json.loads((cli.run_dir(cfg["run_name"]) / "manifest.json").read_text())
    assert man["data_sha256"] == "abc" and man["arm"] == "cpt" and man["final_steps"] == [50, 200]
    rd = cli.run_dir(cfg["run_name"])
    (rd / "train_result.json").write_text(json.dumps({
        "checkpoints": [{"step": s, "path": f"/checkpoints/x/checkpoint-{s}"} for s in [0, *mc.SAVE_1B]],
        "mask_verification": None}))
    (rd / "eval_summary.json").write_text("{}")
    # an unselected 1C run is skipped
    c2 = mc.sweep_config("qwen8b", "1e4", 0.014, 0, with_test=True)
    mc.write(c2)
    cli.write_manifest(c2["run_name"], c2)
    (cli.run_dir(c2["run_name"]) / "train_result.json").write_text(json.dumps({"checkpoints": [{"step": 50, "path": "/c/50"}]}))
    (cli.run_dir(c2["run_name"]) / "eval_summary.json").write_text("{}")
    cli.registry([])
    entries = {p.stem: json.loads(p.read_text()) for p in cli.REGISTRY.glob("*.json")}
    assert set(entries) == {"cpt-qwen8b-r32-seed0", "cpt-qwen8b-r32-seed0-step50"}
    assert entries["cpt-qwen8b-r32-seed0"]["adapter_path"].endswith("checkpoint-200")
    assert entries["cpt-qwen8b-r32-seed0-step50"]["adapter_path"].endswith("checkpoint-50")
    e = entries["cpt-qwen8b-r32-seed0"]
    assert e["noise"] is True and e["measure_kl"] is True and e["eval_chat_template_kwargs"] == {"enable_thinking": True}
    assert e["experiment"] == "fragility/cpt" and e["base_model"] == "Qwen/Qwen3-8B" and e["seed"] == 0
    # now select the 1C run -> it gets a KL-only entry
    cli.RESULTS.mkdir(parents=True, exist_ok=True)
    (cli.RESULTS / "lr_selection.json").write_text(json.dumps({"qwen8b/1e4": {"model": "qwen8b", "klabel": "1e4", "lr": 0.014}}))
    cli.registry([])
    e = json.loads((cli.REGISTRY / f"{c2['run_name']}.json").read_text())
    assert e["noise"] is False and e["measure_kl"] is True and e["train_params"] == 10_000


def test_cap_override_and_admission_guards(sandbox):
    assert cli.current_cap(80) == 80
    cli.QUEUE.mkdir(parents=True, exist_ok=True)
    (cli.QUEUE / "cap.txt").write_text("30\n")
    assert cli.current_cap(80) == 30
    assert cli.guards_ok(None, None) == (True, "")
    assert cli.guards_ok(800, 47)[0] and not cli.guards_ok(801, 47)[0]
    assert cli.guards_ok(10, 25)[0] and not cli.guards_ok(10, 24)[0]
    pct = cli.free_memory_pct()
    assert pct is None or 0 <= pct <= 100


FAKE_TRAIN_EVAL = '''
import json, sys, time, pathlib
a = sys.argv[1:]
time.sleep(0.3)
if a[0] == "--eval-only":
    rd = pathlib.Path(a[1]); assert (rd / "config.yaml").exists()
    (rd / "eval_summary.json").write_text("{}")
else:
    cfg = pathlib.Path(a[0]); rd = cfg.parent
    (rd / "config.yaml").write_text(cfg.read_text())
    (rd / "train_result.json").write_text(json.dumps({"checkpoints": []}))
'''


def test_scheduler_respects_cap_and_chains_eval(sandbox, monkeypatch):
    fake = sandbox / "fake_train_eval.py"
    fake.write_text(FAKE_TRAIN_EVAL)
    monkeypatch.setattr(cli, "TRAIN_EVAL", fake)
    for lr in mc.lr_grid("qwen8b", "1e5"):
        cfg = mc.sweep_config("qwen8b", "1e5", lr, 0, with_test=False)
        mc.write(cfg)
        cli.enqueue("train", cfg["run_name"], 0)
    cap = mc.EVAL_GPUS["cpt_param_sweep"]  # evals fit one at a time; trains three at a time
    cli.scheduler(["--cap", str(cap), "--poll", "1", "--exit-when-idle"])
    jobs = cli.load_jobs()
    assert len(jobs) == 6 and all(j["status"] == "done" for j in jobs)
    assert sum(j["kind"] == "eval" for j in jobs) == 3
    log = (cli.QUEUE / "scheduler.log").read_text()
    reserved = [int(x) for x in re.findall(rf"reserved=(\d+)/{cap}", log)]
    assert reserved and max(reserved) <= cap
    for j in jobs:
        rd = cli.ROOT / j["run_dir"]
        assert (rd / "manifest.json").exists() and (rd / "eval_summary.json").exists()
        if j["kind"] == "train":
            assert all(not b["run"] for b in yaml.safe_load((rd / "train_config.yaml").read_text())["eval"].values())
            assert all(b["run"] for b in yaml.safe_load((rd / "config.yaml").read_text())["eval"].values())
