"""Migration rule 1 for the elicit workstreams: every sweep reproduces the manifests / train records of the
runs it covers (kaushik/elicit/common/check.py), and nocot_sft's train_eval sweeps reproduce its configs."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import sweep as sw
from kaushik.core.io import KAUSHIK
from kaushik.elicit.common import check

MANIFEST_SWEEPS = sorted(p for ws in ("gepa_distill", "gepa_lora", "gepa_dose", "len_teacher", "lora_arith")
                         for p in (KAUSHIK / "elicit" / ws / "sweeps").glob("*.yaml"))


@pytest.mark.parametrize("path", MANIFEST_SWEEPS, ids=lambda p: f"{p.parents[1].name}/{p.stem}")
def test_manifest_equivalence(path):
    report = check.check_sweep(path)
    if not any(r["status"] != "missing" for r in report.values()):
        pytest.skip("no launched runs for this sweep")
    bad = {k: v["diffs"] for k, v in report.items() if v["status"] == "differs"}
    assert not bad, bad


NOCOT_SWEEPS = [
    ("elicit/nocot_sft/sweeps/train.yaml", ["kaushik/elicit/nocot_sft/configs/*.yaml", "kaushik/elicit/nocot_sft/runs/nocot-*-seed?/config.yaml"]),
    ("elicit/nocot_sft/sweeps/test_step25.yaml", ["kaushik/elicit/nocot_sft/runs/nocot-fmt-*-step25/config.yaml"]),
    ("elicit/nocot_sft/sweeps/ref1a.yaml", ["kaushik/elicit/nocot_sft/runs/ref1a-*/config.yaml"]),
]
# Historical configs predate the model table: every block gains `tensor_parallel_size: 1` (the engine default).
NOCOT_EXPLAINED = {"tensor_parallel_size (extra in new)"}


@pytest.mark.parametrize("sweep_path,globs", NOCOT_SWEEPS)
def test_nocot_config_equivalence(sweep_path, globs):
    path = KAUSHIK / sweep_path
    paths = [p for g in globs for p in glob.glob(str(ROOT / g))]
    if not path.exists() or not paths:
        pytest.skip("configs not present")
    rep = sw.check_equivalence(sw.resolve_sweep(sw.load_sweep(path)), paths)
    bad = {}
    for k, v in rep.items():
        if k == "_unmatched":
            continue
        for e in v.get("per_file") or [v]:
            diffs = [d for d in e["diffs"] if not any(d.endswith(x) for x in NOCOT_EXPLAINED)]
            if e["status"] == "missing" or diffs:
                bad[k] = diffs or e["status"]
    assert not bad, bad
    assert not rep["_unmatched"]
