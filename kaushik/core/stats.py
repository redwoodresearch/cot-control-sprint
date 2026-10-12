"""Bootstrap statistics and table formatting, one implementation each.

Two resampling families are kept because committed results depend on their exact
random-number consumption (re-running an analysis must reproduce its tables):

* `matrix` / `boot_means` / `ci` / `paired_boot`: seeds x questions matrices
  (kaushik/orange/analyze.py). Cells resample seeds and questions; differences
  share the question resample between arms (paired on tasks, independent seeds).
* `boot_1d` / `boot_seeds_questions` / `boot_paired_seeds`: the plain-array
  variants used by orange_expand/baseline (per-question flags, seeds x questions).
* `boot_interleaved`: per-draw question-then-seeds order (kaushik/orange/rl).

Also: `add_long` (compl_long = compliance among traces at least as long as the
untrained median), `hit` helpers, `pct` / `pct_ci` / `md_table` / `write_csv`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DEGEN_CHARS = 200          # reasoning shorter than this counts as degenerate
POOLED = "(pooled)"        # mode label for cells pooled over modes
METRICS = ("honest", "compliance", "compl_long", "correct")


# ----------------------------------------------------------------------------- seeds x questions


def matrix(g: pd.DataFrame, value: str, tasks: list[str] | None = None) -> np.ndarray:
    """seeds x tasks matrix of a metric (NaN where missing); seed = one training run."""
    m = g.pivot_table(index="seed", columns="task", values=value, aggfunc="mean")
    if tasks is not None:
        m = m.reindex(columns=tasks)
    return m.to_numpy(dtype=float)


def boot_means(M: np.ndarray, rng: np.random.Generator, B: int, cols: np.ndarray | None = None,
               batch: int = 200) -> np.ndarray:
    """B bootstrap means resampling seeds (rows) and questions (cols) with replacement.
    `cols` (B x Q) lets two arms share the question resample (paired differences)."""
    S, Q = M.shape
    if cols is None:
        cols = rng.integers(0, Q, (B, Q))
    out = np.empty(B)
    for i in range(0, B, batch):
        rows = rng.integers(0, S, (min(batch, B - i), S))
        samp = M[rows[:, :, None], cols[i:i + len(rows), None, :]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN resample -> NaN
            out[i:i + len(rows)] = np.nanmean(samp.reshape(len(rows), -1), axis=1)
    return out


def ci(values: np.ndarray) -> tuple[float, float]:
    v = values[~np.isnan(values)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def nanmean(M: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        return float(np.nanmean(M)) if np.isfinite(M).any() else np.nan


def cell_metrics(g: pd.DataFrame, rng: np.random.Generator, B: int, metrics=METRICS) -> dict:
    """{metric, metric_lo, metric_hi} for each metric of one cell (rows of one arm)."""
    out = {}
    for metric in metrics:
        M = matrix(g, metric)
        out[metric] = nanmean(M)
        out[f"{metric}_lo"], out[f"{metric}_hi"] = ci(boot_means(M, rng, B))
    return out


def paired_boot(xa: pd.DataFrame, xb: pd.DataFrame, rng: np.random.Generator, B: int,
                metrics=METRICS) -> dict:
    """A - B with paired-question (shared resample), independent-seed intervals.
    Draws the shared question resample first, then one seed resample per arm and
    metric, in this fixed order."""
    tasks = sorted(set(xa.task) | set(xb.task))
    cols = rng.integers(0, len(tasks), (B, len(tasks)))
    out = {}
    for metric in metrics:
        Ma, Mb = matrix(xa, metric, tasks), matrix(xb, metric, tasks)
        with np.errstate(all="ignore"):
            out[metric] = float(np.nanmean(Ma) - np.nanmean(Mb))
        out[f"{metric}_lo"], out[f"{metric}_hi"] = ci(boot_means(Ma, rng, B, cols) - boot_means(Mb, rng, B, cols))
    return out


# ----------------------------------------------------------------------------- plain arrays


def boot_1d(flags: np.ndarray, rng: np.random.Generator, B: int) -> tuple[float, float, float]:
    """(mean, lo, hi) over questions for a 1-D array of flags."""
    if len(flags) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(flags), size=(B, len(flags)))
    m = flags[idx].mean(axis=1)
    return float(flags.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_seeds_questions(M: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    """Bootstrap means of a seeds x questions matrix: resample seeds and questions."""
    S, Q = M.shape
    si = rng.integers(0, S, size=(B, S))
    qi = rng.integers(0, Q, size=(B, Q))
    return np.stack([M[np.ix_(si[b], qi[b])].mean() for b in range(B)])


def boot_paired_seeds(Ma: np.ndarray, Mb: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    """Bootstrap of mean(Ma) - mean(Mb): questions paired (first min(Q) columns),
    seeds resampled independently per arm."""
    Q = min(Ma.shape[1], Mb.shape[1])
    Ma, Mb = Ma[:, :Q], Mb[:, :Q]
    qi = rng.integers(0, Q, size=(B, Q))
    sa = rng.integers(0, Ma.shape[0], size=(B, Ma.shape[0]))
    sb = rng.integers(0, Mb.shape[0], size=(B, Mb.shape[0]))
    return np.array([Ma[np.ix_(sa[i], qi[i])].mean() - Mb[np.ix_(sb[i], qi[i])].mean() for i in range(B)])


def boot_interleaved(MA: np.ndarray, MB: np.ndarray | None = None, rng: np.random.Generator | None = None,
                     B: int = 2000) -> np.ndarray:
    """B bootstrap means of MA (or MA - MB), resampling per draw: shared question
    columns, then MA's seeds, then MB's seeds (kaushik/orange/rl/analyze.py order)."""
    SA, Q = MA.shape
    out = np.empty(B)
    for i in range(B):
        q = rng.integers(0, Q, Q)
        a = MA[rng.integers(0, SA, SA)][:, q].mean()
        if MB is not None:
            a -= MB[rng.integers(0, MB.shape[0], MB.shape[0])][:, q].mean()
        out[i] = a
    return out


# ----------------------------------------------------------------------------- derived columns


def untrained_bars(untrained: pd.DataFrame) -> dict:
    """Median reasoning chars of the untrained rows per (model, condition, mode), with
    a (model, condition) fallback."""
    bars = untrained.groupby(["model", "condition", "mode"]).chars.median().to_dict()
    bars.update({k: v for k, v in untrained.groupby(["model", "condition"]).chars.median().to_dict().items()})
    return bars


def add_long(df: pd.DataFrame, bars: dict) -> pd.DataFrame:
    """bar / long / compl_long columns: compliance among traces at least as long as the
    untrained median of the same (model, condition, mode)."""
    bar = [bars.get((m, c, mo), bars.get((m, c), np.nan)) for m, c, mo in zip(df.model, df.condition, df["mode"])]
    df = df.assign(bar=bar)
    df["long"] = df.chars >= df.bar
    df["compl_long"] = np.where(df.long, df.compliance.astype(float), np.nan)
    return df


def with_pooled(df: pd.DataFrame, label: str = POOLED) -> pd.DataFrame:
    return pd.concat([df, df.assign(mode=label)], ignore_index=True)


def passes_hit(row, metric: str, points: float) -> bool:
    """>= `points` improvement with the 95% interval excluding 0."""
    return bool(row[metric] >= points and row[f"{metric}_lo"] > 0)


# ----------------------------------------------------------------------------- formatting


def pct(x, missing: str = "-") -> str:
    return missing if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def pct_ci(row, m: str) -> str:
    if np.isnan(row[m]):
        return "-"
    return f"{pct(row[m])} [{pct(row[f'{m}_lo'])}, {pct(row[f'{m}_hi'])}]"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    lines = [",".join(header)] + [",".join("" if v is None else str(v) for v in r) for r in rows]
    Path(path).write_text("\n".join(lines) + "\n")
