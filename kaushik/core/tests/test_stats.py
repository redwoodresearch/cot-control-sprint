from __future__ import annotations

import numpy as np
import pandas as pd

from kaushik.core import stats as st


def _df(arm, vals, seeds=(0,)):
    rows = []
    for s in seeds:
        for i, v in enumerate(vals):
            rows.append({"arm": arm, "seed": s, "task": f"q{i}|m", "honest": v, "compliance": v, "correct": True,
                         "compl_long": v, "chars": 100 + i, "model": "m", "condition": "c", "mode": "m"})
    return pd.DataFrame(rows)


def test_boot_means_deterministic_and_bounded():
    M = np.array([[0, 1, 1, 0], [1, 1, 0, 0]], float)
    a = st.boot_means(M, np.random.default_rng(0), 200)
    b = st.boot_means(M, np.random.default_rng(0), 200)
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() <= 1
    lo, hi = st.ci(a)
    assert 0 <= lo <= M.mean() <= hi <= 1


def test_matrix_and_cell_metrics():
    g = _df("a", [1, 0, 1, 1], seeds=(0, 1))
    M = st.matrix(g, "honest")
    assert M.shape == (2, 4) and M.sum() == 6
    out = st.cell_metrics(g, np.random.default_rng(0), 100)
    assert out["honest"] == 0.75 and out["honest_lo"] <= 0.75 <= out["honest_hi"]
    assert set(out) == {f"{m}{s}" for m in st.METRICS for s in ("", "_lo", "_hi")}


def test_paired_boot_identical_arms_is_zero():
    xa, xb = _df("a", [1, 0, 1]), _df("b", [1, 0, 1])
    d = st.paired_boot(xa, xb, np.random.default_rng(1), 50)
    assert d["honest"] == 0 and d["honest_lo"] == 0 and d["honest_hi"] == 0
    d = st.paired_boot(_df("a", [1, 1, 1]), _df("b", [0, 0, 0]), np.random.default_rng(1), 50)
    assert d["honest"] == 1 and d["honest_lo"] == 1


def test_plain_array_bootstraps():
    rng = np.random.default_rng(0)
    mean, lo, hi = st.boot_1d(np.array([1, 0, 1, 1], float), rng, 100)
    assert mean == 0.75 and lo <= mean <= hi
    assert st.boot_1d(np.array([]), rng, 10)[0] != st.boot_1d(np.array([]), rng, 10)[0]   # nan != nan
    M = np.array([[1, 0], [1, 1]], float)
    b = st.boot_seeds_questions(M, rng, 20)
    assert b.shape == (20,) and set(np.unique(b)) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    one = np.array([[1, 0, 1, 1]], float)      # a single seed: paired questions, so A - A is exactly 0
    assert np.all(st.boot_paired_seeds(one, one, rng, 20) == 0)
    d = st.boot_paired_seeds(M, M, rng, 20)     # two seeds resampled independently per arm: not all 0
    assert d.shape == (20,) and np.abs(d).max() <= 1


def test_add_long_and_pooled():
    u = _df("untrained", [1, 1, 1]); u["chars"] = [100, 200, 300]
    t = _df("trained", [1, 1, 0]); t["chars"] = [50, 250, 350]
    df = st.add_long(pd.concat([u, t], ignore_index=True), st.untrained_bars(u))
    assert df.bar.unique().tolist() == [200]
    assert df[df.arm == "trained"].long.tolist() == [False, True, True]
    assert np.isnan(df[df.arm == "trained"].compl_long.iloc[0])
    pooled = st.with_pooled(df)
    assert (pooled["mode"] == st.POOLED).sum() == len(df)


def test_formatting():
    assert st.pct(0.1234) == "12.3" and st.pct(float("nan")) == "-" and st.pct(None, missing="–") == "–"
    row = {"honest": 0.5, "honest_lo": 0.25, "honest_hi": 0.75}
    assert st.pct_ci(row, "honest") == "50.0 [25.0, 75.0]"
    assert st.md_table(["a", "b"], [[1, "x"]]) == "| a | b |\n|---|---|\n| 1 | x |"
    assert st.passes_hit({"honest": 0.03, "honest_lo": 0.01}, "honest", 0.02)
    assert not st.passes_hit({"honest": 0.03, "honest_lo": 0.0}, "honest", 0.02)


def test_boot_interleaved_order():
    MA = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], float)
    MB = np.array([[0, 0, 1, 0]], float)
    a = st.boot_interleaved(MA, MB, np.random.default_rng(3), 50)
    b = st.boot_interleaved(MA, MB, np.random.default_rng(3), 50)
    assert a.shape == (50,) and np.array_equal(a, b) and -1 <= a.min() and a.max() <= 1
    # same draw order as the loop in kaushik/orange/rl/analyze.py
    rng = np.random.default_rng(3)
    q = rng.integers(0, 4, 4); sa = rng.integers(0, 2, 2); sb = rng.integers(0, 1, 1)
    assert a[0] == MA[sa][:, q].mean() - MB[sb][:, q].mean()
