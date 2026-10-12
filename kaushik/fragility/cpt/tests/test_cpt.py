"""CPU tests for kaushik/fragility/cpt: config equivalence of the sweeps with the committed run
configs (migration rule 1), the iso-drive LR rule behind the 1C sweep axis, resolved configs as
train_eval inputs, the analysis metrics / bootstrap, LR selection and registry entries. No
network, no Modal:

    uv run --with pytest pytest -q kaushik/fragility/cpt/tests
"""

from __future__ import annotations

import ast
import glob
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cotcontrol.training.config import (
    TrainConfig,
    default_lora_targets,
    filter_dataclass_kwargs,
)
from kaushik.core import sweep as sw
from kaushik.core.config import resolve, train_eval_config
from kaushik.core.io import KAUSHIK
from kaushik.fragility.cpt import analyze, registry_entries, select

EXP = "fragility/cpt"
SWEEPS = KAUSHIK / "fragility" / "cpt" / "sweeps"
CFG = resolve(EXP)
AN = CFG["analysis"]
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "quark", "lepton", "boson", "photon", "gluon", "hadron", "meson", "baryon", "orbit", "vector", "tensor", "matrix", "scalar", "field", "lattice", "crystal", "plasma", "vortex", "entropy", "enthalpy"]


def _points(name: str):
    return sw.resolve_sweep(sw.load_sweep(SWEEPS / name))


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
    for m in AN["val_modes"]:
        write_eval_json(rd / "eval" / f"val_{m}" / f"checkpoint-{step}.json", m, 150, p_honest, p_correct, rng, **kw)


# ----------------------------------------------------------------------------- sweeps / configs


@pytest.mark.parametrize("sweep,globs", [
    ("cpt.yaml", ["kaushik/fragility/cpt/configs/cpt-*-r32-seed?.yaml"]),
    ("cpt_lowlr.yaml", ["kaushik/fragility/cpt/configs/cpt-*-lr*.yaml"]),
    ("smoke.yaml", ["kaushik/fragility/cpt/configs/smoke-*.yaml"]),
])
def test_config_equivalence(sweep, globs):
    """Migration rule 1: every committed per-run config is reproduced by its sweep."""
    paths = [p for g in globs for p in glob.glob(str(ROOT / g))]
    if not paths:
        pytest.skip("committed configs not present")
    rep = sw.check_equivalence(_points(sweep), paths)
    bad = {k: v for k, v in rep.items() if k != "_unmatched" and v["status"] != "equal"}
    assert not bad, bad
    assert not rep["_unmatched"]


def test_1c_equivalence():
    """cptk_selected.yaml reproduces the 24 selected-LR configs (seeds 0-2, with test blocks); the 16
    other stage-1 cells are reproduced by cptk_sweep.yaml, whose 8 selected seed-0 cells differ from
    their committed configs only by the test blocks `select` added to them afterwards."""
    paths = glob.glob(str(ROOT / "kaushik/fragility/cpt/configs/cptk-*.yaml"))
    if not paths:
        pytest.skip("committed configs not present")
    selected = {c["run_name"] for _, c in _points("cptk_selected.yaml")}
    rep = sw.check_equivalence(_points("cptk_selected.yaml"), paths)
    assert all(r["status"] == "equal" for n, r in rep.items() if n != "_unmatched"), rep
    assert len(rep) - 1 == 24 and len(rep["_unmatched"]) == 16
    rep = sw.check_equivalence(_points("cptk_sweep.yaml"), [p for p in paths if p.endswith("-seed0.yaml")])
    assert not rep["_unmatched"]
    for name, r in rep.items():
        if name == "_unmatched":
            continue
        if name in selected:
            assert sorted(r["diffs"]) == [f"eval.{b} (missing in new)" for b in sorted(AN["test_blocks"])], name
        else:
            assert r["status"] == "equal", (name, r)


def test_adapter_sizes_and_iso_drive_lr():
    numel = AN["adapter_numel"]
    assert numel["gptoss20b"]["r32"] == 369_819_648 and numel["qwen8b"]["r32"] == 87_293_952
    for m in AN["models"]:
        assert numel[m]["r1"] * 32 == numel[m]["r32"]
    pts = sw.expand_points(sw.load_sweep(SWEEPS / "cptk_sweep.yaml"))
    assert len(pts) == 24
    grids: dict[tuple, list[float]] = {}
    for p in pts:
        lab, ov = p["labels"], p["overrides"]
        k = AN["ks"][str(lab["k"])] or numel[lab["model"]]["r1"]
        assert ov["cpt_meta.k"] == k and ov["elicitation.train_params"] == AN["ks"][str(lab["k"])]
        drive = CFG["train"]["lr"] * math.sqrt(numel[lab["model"]]["r32"])    # (alpha/r) x lr x sqrt(k) of 1B
        assert ov["cpt_meta.lr_predicted"] == pytest.approx(drive / math.sqrt(k))
        assert ov["train.lr"] == ov["cpt_meta.lr"]
        assert any(ov["train.lr"] == float(f"{ov['cpt_meta.lr_predicted'] * f:.1e}") for f in (1 / 3, 1.0, 3.0))
        assert p["run_name"] == f"cptk-{lab['model']}-{lab['k']}-lr{ov['train.lr']:.1e}-seed0"
        grids.setdefault((lab["model"], str(lab["k"])), []).append(ov["train.lr"])
    for g in grids.values():
        g.sort()
        assert len(g) == 3 and g[2] / g[0] == pytest.approx(9, rel=0.15)
    # one rung (x3.16) per decade of k is the lrk_analysis prediction
    analyze.configure(CFG)
    assert analyze.predicted_lr("gptoss20b", "1e3") / analyze.predicted_lr("gptoss20b", "1e4") == pytest.approx(10 ** 0.5)
    assert analyze.lr_grid("qwen8b", "1e4") == [0.0047, 0.014, 0.042]


def test_selected_sweep_matches_lr_selection():
    sel = json.loads((ROOT / "kaushik/fragility/cpt/results/lr_selection.json").read_text())
    axis = {(p["labels"]["model"], str(p["labels"]["k"]), p["overrides"]["train.lr"])
            for p in sw.expand_points(sw.load_sweep(SWEEPS / "cptk_selected.yaml"))}
    assert axis == {(v["model"], v["klabel"], v["lr"]) for v in sel.values()}
    seeds = {p["overrides"]["train.seed"] for p in sw.expand_points(sw.load_sweep(SWEEPS / "cptk_selected.yaml"))}
    assert seeds == {0, 1, 2}


def _default_eval_keys() -> set[str]:
    tree = ast.parse((ROOT / "scripts" / "train_eval.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DEFAULT_EVAL" for t in node.targets):
            return {k.value for k in node.value.keys}
    raise AssertionError("DEFAULT_EVAL not found")


def _flat(view: dict) -> dict:
    raw = {k: v for k, v in view.items() if k != "eval"}
    flat = {k: v for s in raw.values() if isinstance(s, dict) for k, v in s.items()}
    return flat | {k: v for k, v in raw.items() if not isinstance(v, dict)}


def test_configs_are_valid_train_eval_inputs():
    keys = _default_eval_keys()
    for sweep in ("cpt.yaml", "cpt_lowlr.yaml", "cptk_sweep.yaml", "cptk_selected.yaml", "smoke.yaml"):
        for _, cfg in _points(sweep):
            view = train_eval_config(cfg)
            tc = TrainConfig(**filter_dataclass_kwargs(TrainConfig, _flat(view)))
            meta = view["cpt_meta"]
            assert tc.weight_decay == 0.0 and tc.batch_size == 48 and tc.pack_sequences
            assert tc.max_seq_length == 4096 and tc.lr_scheduler_type == "constant"
            assert tc.mask_seed == meta["seed"] == tc.seed and meta["model"] == cfg["model"]
            assert tc.lr == meta["lr"] and tc.lora_rank == meta["rank"]
            mods, params = default_lora_targets(view["base_model"])
            assert tuple(tc.lora_target_modules) == mods and tuple(tc.lora_target_parameters) == params
            if meta["arm"] in ("cpt", "cpt_lowlr"):
                assert (tc.lora_rank, tc.lora_alpha, tc.train_params) == (32, 32, None)
                assert tc.max_steps == 200 and 50 in tc.save_at_steps and 200 in tc.save_at_steps
                assert meta["k"] == AN["adapter_numel"][cfg["model"]]["r32"] and meta["final_steps"] == [50, 200]
            elif meta["arm"] == "cpt_param_sweep":
                assert (tc.lora_rank, tc.lora_alpha) == (1, 1) and tc.train_params == AN["ks"][str(meta["klabel"])]
                assert tc.max_steps == 50 and meta["final_steps"] == [50]
            for name, b in view["eval"].items():
                assert set(b) <= keys, f"{name}: unknown eval keys {set(b) - keys}"
                assert b["temperature"] == 1.0 and b["max_tokens"] == 16000 and b["chat_template_kwargs"]
                assert b["split"] == ("test" if name.startswith("test_") else "val")
                if b.get("eval_at_steps") is not None:
                    assert set(b["eval_at_steps"]) <= set(tc.save_at_steps) | {0}
                assert b["chunk_parallel"] == b["parallel_checkpoints"] == {"cpt": 8, "cpt_lowlr": 8, "cpt_param_sweep": 4, "smoke": 2}[meta["arm"]]
            if meta["arm"] != "smoke":
                assert {f"val_{m}" for m in AN["val_modes"] + AN["extra_val_modes"]} <= set(view["eval"])
            has_test = set(AN["test_blocks"]) <= set(view["eval"])
            assert has_test == (sweep != "cptk_sweep.yaml"), sweep   # selection runs on val only
            assert cfg["registry"]["noise"] == (meta["arm"] == "cpt")


# ----------------------------------------------------------------------------- analyze


def test_load_rollouts_summarize_and_compl_long(tmp_path):
    analyze.configure(CFG)
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
    analyze.configure(CFG)
    (tmp_path / "losses.jsonl").write_text("".join(json.dumps({"step": s, "loss": 20.0 - s}) + "\n" for s in range(1, 51)))
    assert analyze.loss_at(tmp_path, 0) == 19.0                       # step 0 = the untrained model's loss (B = 0)
    assert analyze.loss_at(tmp_path, 50) == pytest.approx(analyze.mean([20 - s for s in range(48, 51)]))
    assert analyze.loss_at(tmp_path, 2, window=3) == pytest.approx((19 + 18) / 2)   # window clipped at step 1
    assert analyze.loss_at(tmp_path / "missing", 10) is None
    assert analyze.kl_suffix("cpt", 200, [50, 200]) == "" and analyze.kl_suffix("cpt", 50, [50, 200]) == "-step50"
    assert analyze.kl_suffix("cpt_param_sweep", 50, [50]) == ""


# ----------------------------------------------------------------------------- select / registry


def test_select_picks_best_lr_under_accuracy_guard(tmp_path, capsys):
    cfg = resolve(EXP, overrides={"run_dir": str(tmp_path / "runs" / "{run_name}")})
    rng = np.random.default_rng(2)
    m, klabel = "qwen8b", "1e4"
    analyze.configure(cfg)
    cells = select.sweep_cells(sw.load_sweep(SWEEPS / "cptk_sweep.yaml"))[(m, klabel)]
    lrs = [c["lr"] for c in cells]
    plan = {0: (0.02, 0.40), 1: (0.20, 0.40), 2: (0.35, 0.05)}  # rung -> (honest@50, acc@50)
    for i, (h, a) in plan.items():
        rd = tmp_path / "runs" / cells[i]["run_name"]
        write_val_blocks(rd, 0, 0.02, 0.40, rng)
        write_val_blocks(rd, 50, h, a, rng)
        (rd / "train_result.json").write_text(json.dumps({"final_loss": 2.0, "mask_verification": {"n_changed_inside_mask": 9000}}))
    out = tmp_path / "lr_selection.json"
    select.main(cfg, ["--models", m, "--ks", klabel, "--out", str(out)])
    sel = json.loads(out.read_text())[f"{m}/{klabel}"]
    assert sel["lr"] == lrs[1] and not sel["guard_failed_all"] and sel["run_name"] == cells[1]["run_name"]
    assert [c["guard_ok"] for c in sel["cells"]] == [True, True, False]
    assert sel["cells"][0]["k_effective"] == 9000 and sel["cells"][0]["final_loss"] == 2.0
    text = capsys.readouterr().out
    assert f'k: "{klabel}", lr: "{lrs[1]:.1e}"' in text and "eval.test" not in text   # the axis entry for cptk_selected.yaml
    select.main(cfg, ["--models", m, "--ks", klabel, "--out", str(out)])   # idempotent
    assert "already selected" in capsys.readouterr().out
    # everything collapses -> nothing is selected; the message names the rung to add
    cells3 = select.sweep_cells(sw.load_sweep(SWEEPS / "cptk_sweep.yaml"))[(m, "1e3")]
    for c in cells3:
        rd = tmp_path / "runs" / c["run_name"]
        write_val_blocks(rd, 0, 0.02, 0.40, rng)
        write_val_blocks(rd, 50, 0.5, 0.05, rng)
    select.main(cfg, ["--models", m, "--ks", "1e3", "--out", str(out)])
    assert f"{m}/1e3" not in json.loads(out.read_text())
    assert f"add a rung at lr={min(c['lr'] for c in cells3) / 3:.1e}" in capsys.readouterr().out


def test_registry_entries_for_1b_and_selected_1c(tmp_path):
    cfg_1b = next(c for _, c in _points("cpt.yaml") if c["run_name"] == "cpt-qwen8b-r32-seed0")
    rd = tmp_path / cfg_1b["run_name"]
    rd.mkdir()
    (rd / "config.yaml").write_text(json.dumps(train_eval_config(cfg_1b)))
    (rd / "train_result.json").write_text(json.dumps({
        "checkpoints": [{"step": s, "path": f"/checkpoints/x/checkpoint-{s}"} for s in [0, *cfg_1b["train"]["save_at_steps"]]],
        "mask_verification": None, "run_name": "x"}))
    entries = {e["run_name"]: e for e in registry_entries.entries_for(rd, AN["test_blocks"])}
    assert set(entries) == {"cpt-qwen8b-r32-seed0", "cpt-qwen8b-r32-seed0-step50"}
    assert entries["cpt-qwen8b-r32-seed0"]["adapter_path"].endswith("checkpoint-200")
    assert entries["cpt-qwen8b-r32-seed0-step50"]["adapter_path"].endswith("checkpoint-50")
    e = entries["cpt-qwen8b-r32-seed0"]
    assert e["noise"] is True and e["measure_kl"] is True and e["eval_chat_template_kwargs"] == {"enable_thinking": True}
    assert e["experiment"] == EXP and e["arm"] == "cpt" and e["base_model"] == "Qwen/Qwen3-8B" and e["seed"] == 0
    assert e["noise_eval_blocks"].endswith("config.yaml#" + ",".join(AN["test_blocks"])) and e["lora_rank"] == 32
    # a selected 1C run: KL only, one entry
    cfg_1c = next(c for _, c in _points("cptk_selected.yaml") if c["run_name"] == "cptk-qwen8b-1e4-lr4.2e-02-seed1")
    rd = tmp_path / cfg_1c["run_name"]
    rd.mkdir()
    (rd / "config.yaml").write_text(json.dumps(train_eval_config(cfg_1c)))
    (rd / "train_result.json").write_text(json.dumps({"checkpoints": [{"step": 50, "path": "/c/50"}],
                                                      "mask_verification": {"n_changed_inside_mask": 9967}}))
    (e,) = registry_entries.entries_for(rd, AN["test_blocks"])
    assert e["noise"] is False and e["measure_kl"] is True and e["train_params"] == 10_000 and e["k_effective"] == 9967
    assert e["arm"] == "cpt_param_sweep" and e["experiment"] == EXP and e["seed"] == 1
