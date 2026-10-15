"""Small statistics helpers (confidence intervals and tests) used across the app."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as _st


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion, in percent."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def fisher_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]]."""
    if min(a + b, c + d, a + c, b + d) == 0:
        return float("nan")
    return float(_st.fisher_exact([[a, b], [c, d]])[1])


def chi2_independence(table) -> float:
    """p-value of a chi-square test of independence (NaN if degenerate)."""
    t = np.asarray(table, dtype=float)
    if t.ndim != 2:
        return float("nan")
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return float("nan")
    try:
        return float(_st.chi2_contingency(t)[1])
    except ValueError:
        return float("nan")


def welch(a, b) -> tuple[float, float]:
    """(difference of means b − a, two-sided Welch p-value)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    if a.std() == 0 and b.std() == 0:
        return (float(b.mean() - a.mean()), float("nan"))
    p = _st.ttest_ind(a, b, equal_var=False).pvalue
    return (float(b.mean() - a.mean()), float(p))


def levene(a, b) -> tuple[float, float]:
    """(ratio of standard deviations b / a, Brown–Forsythe p-value)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3 or a.std() == 0:
        return (float("nan"), float("nan"))
    p = _st.levene(a, b, center="median").pvalue
    return (float(b.std(ddof=1) / a.std(ddof=1)), float(p))


def normal_tail_ppm(mu: float, sd: float, lo: float, hi: float) -> float:
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(1e6 * (_st.norm.cdf((lo - mu) / sd) + _st.norm.sf((hi - mu) / sd)))


def cpk_ci(cpk: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Bissell's approximate confidence interval for Cpk."""
    if not np.isfinite(cpk) or n < 3:
        return (float("nan"), float("nan"))
    se = math.sqrt(1 / (9 * n) + cpk * cpk / (2 * (n - 1)))
    return (cpk - z * se, cpk + z * se)


def significance_label(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def is_significant(p: float, alpha: float = 0.05) -> bool:
    return bool(p is not None and np.isfinite(p) and p < alpha)
