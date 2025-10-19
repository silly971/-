"""Changepoint detection algorithms used by the GUI toolbox."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DetectionResult:
    name: str
    statistic: float
    pvalue: Optional[float]
    change_points: List[int]
    details: Dict[str, float]
    summary: str


def _format_index(idx: int, index: pd.Index) -> str:
    if 0 <= idx < len(index):
        return str(index[idx])
    return "N/A"


def mann_kendall_test(series: pd.Series, alpha: float = 0.05) -> DetectionResult:
    values = series.to_numpy()
    n = len(values)
    if n < 8:
        raise ValueError("Mann-Kendall test requires at least 8 observations.")

    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(values[k + 1 :] - values[k]))

    unique_vals, counts = np.unique(values, return_counts=True)
    var_s = (
        (n * (n - 1) * (2 * n + 5))
        - np.sum(counts * (counts - 1) * (2 * counts + 5))
    ) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    trend = "increasing" if z > 0 else "decreasing" if z < 0 else "no"
    summary = (
        f"Mann-Kendall test indicates a {trend} trend (tau={tau:.3f}, z={z:.3f})."
        f" Significance at alpha={alpha:.2f}: {'Yes' if pvalue < alpha else 'No'}."
    )

    return DetectionResult(
        name="Mann-Kendall",
        statistic=float(z),
        pvalue=float(pvalue),
        change_points=[],
        details={"tau": float(tau)},
        summary=summary,
    )


def pettitt_test(series: pd.Series, alpha: float = 0.05) -> DetectionResult:
    values = series.to_numpy()
    n = len(values)
    if n < 2:
        raise ValueError("Pettitt test requires at least 2 observations.")

    rank = stats.rankdata(values)
    K = np.zeros(n)
    for t in range(n):
        K[t] = 2 * np.sum(rank[: t + 1]) - (t + 1) * (n + 1)
    K_abs = np.abs(K)
    K_max = np.max(K_abs)
    tau = np.argmax(K_abs)
    pvalue = 2 * np.exp((-6 * (K_max ** 2)) / (n ** 3 + n ** 2))
    summary = (
        f"Pettitt test detects a change point around index {tau}"
        f" ({_format_index(tau, series.index)})."
        f" Significance at alpha={alpha:.2f}: {'Yes' if pvalue < alpha else 'No'}."
    )
    return DetectionResult(
        name="Pettitt",
        statistic=float(K_max),
        pvalue=float(pvalue),
        change_points=[int(tau)],
        details={"K": float(K_max)},
        summary=summary,
    )


def sliding_t_test(
    series: pd.Series,
    window: int = 10,
    step: int = 1,
    alpha: float = 0.05,
) -> DetectionResult:
    values = series.to_numpy()
    n = len(values)
    if window * 2 >= n:
        raise ValueError("Sliding window must leave room for both segments.")

    stats_values = []
    change_points = []
    for center in range(window, n - window, step):
        left = values[center - window : center]
        right = values[center : center + window]
        t_stat, pvalue = stats.ttest_ind(left, right, equal_var=False)
        stats_values.append((center, t_stat, pvalue))
        if pvalue < alpha:
            change_points.append(center)

    if stats_values:
        max_stat = max(stats_values, key=lambda x: abs(x[1]))
        summary = (
            f"Sliding t-test evaluated {len(stats_values)} candidate points."
            f" Maximum |t|={abs(max_stat[1]):.3f} at index {max_stat[0]}"
            f" ({_format_index(max_stat[0], series.index)})."
        )
    else:
        max_stat = (0, 0.0, 1.0)
        summary = "Sliding t-test window too large for data length."

    return DetectionResult(
        name="Sliding t-test",
        statistic=float(max_stat[1]),
        pvalue=float(max_stat[2]),
        change_points=[int(cp) for cp in change_points],
        details={"window": float(window)},
        summary=summary,
    )


def cramers_test(series: pd.Series, alpha: float = 0.05) -> DetectionResult:
    values = series.to_numpy()
    n = len(values)
    mean = np.mean(values)
    s2 = np.var(values, ddof=1)
    if s2 == 0:
        raise ValueError("Variance is zero; Cramer's test cannot be applied.")

    cumulative = np.cumsum(values - mean)
    T = np.sum((cumulative[:-1] ** 2) / (np.arange(1, n) * (n - np.arange(1, n))))
    T = T / (n * s2)
    pvalue = 1 - stats.chi2.cdf(T * (n - 1), df=n - 1)
    tau = int(np.argmax(np.abs(cumulative[:-1])))
    summary = (
        f"Cramer's test statistic={T:.3f}, indicating a potential shift"
        f" near index {tau} ({_format_index(tau, series.index)})."
        f" Significance at alpha={alpha:.2f}: {'Yes' if pvalue < alpha else 'No'}."
    )
    return DetectionResult(
        name="Cramer",
        statistic=float(T),
        pvalue=float(pvalue),
        change_points=[tau],
        details={"cumulative_max": float(np.max(np.abs(cumulative)))},
        summary=summary,
    )


def buishand_range_test(series: pd.Series, alpha: float = 0.05) -> DetectionResult:
    values = series.to_numpy()
    n = len(values)
    mean = np.mean(values)
    dev = values - mean
    Sk = np.cumsum(dev)
    R = np.max(Sk) - np.min(Sk)
    s = np.std(values, ddof=1)
    Q = R / s if s > 0 else np.inf
    # Approximate p-value using normal approximation
    pvalue = 2 * (1 - stats.norm.cdf(Q / np.sqrt(n)))
    tau = int(np.argmax(np.abs(Sk[:-1]))) if n > 1 else 0
    summary = (
        f"Buishand range test Q={Q:.3f}, with range R={R:.3f}."
        f" Potential change near index {tau} ({_format_index(tau, series.index)})."
        f" Significance at alpha={alpha:.2f}: {'Yes' if pvalue < alpha else 'No'}."
    )
    return DetectionResult(
        name="Buishand",
        statistic=float(Q),
        pvalue=float(pvalue),
        change_points=[tau],
        details={"range": float(R)},
        summary=summary,
    )


def bayesian_changepoint_detection(
    series: pd.Series,
    prior_strength: float = 0.1,
) -> DetectionResult:
    values = series.to_numpy(dtype=float)
    n = len(values)
    if n < 3:
        raise ValueError("Bayesian changepoint detection requires at least 3 points.")

    cumulative = np.cumsum(values)
    cumulative_sq = np.cumsum(values ** 2)

    log_prob = np.zeros(n - 1)
    for tau in range(1, n):
        left_n = tau
        right_n = n - tau
        left_sum = cumulative[tau - 1]
        right_sum = cumulative[-1] - left_sum
        left_sq = cumulative_sq[tau - 1]
        right_sq = cumulative_sq[-1] - left_sq

        left_mean = left_sum / left_n
        right_mean = right_sum / right_n
        left_var = left_sq - left_sum ** 2 / left_n
        right_var = right_sq - right_sum ** 2 / right_n

        left_ll = (
            -0.5 * left_n * np.log(2 * np.pi)
            - 0.5 * left_n * np.log(max(left_var / max(left_n - 1, 1), 1e-8))
        )
        right_ll = (
            -0.5 * right_n * np.log(2 * np.pi)
            - 0.5 * right_n * np.log(max(right_var / max(right_n - 1, 1), 1e-8))
        )
        penalty = prior_strength * np.log(n)
        log_prob[tau - 1] = left_ll + right_ll - penalty

    max_tau = int(np.argmax(log_prob) + 1)
    log_prob -= np.max(log_prob)
    posterior = np.exp(log_prob)
    posterior /= posterior.sum()

    summary = (
        f"Bayesian detection suggests change near index {max_tau}"
        f" ({_format_index(max_tau, series.index)})."
        f" Posterior probability at this point={posterior[max_tau - 1]:.3f}."
    )
    return DetectionResult(
        name="Bayesian",
        statistic=float(log_prob[max_tau - 1]),
        pvalue=None,
        change_points=[max_tau],
        details={"posterior": float(posterior[max_tau - 1])},
        summary=summary,
    )


def cumulative_anomaly(series: pd.Series) -> DetectionResult:
    values = series.to_numpy()
    mean = np.mean(values)
    cap = np.cumsum(values - mean)
    tau = int(np.argmax(np.abs(cap))) if len(cap) else 0
    summary = (
        f"Cumulative anomaly peaks at index {tau} ({_format_index(tau, series.index)})."
    )
    return DetectionResult(
        name="CAP",
        statistic=float(cap[tau]) if len(cap) else 0.0,
        pvalue=None,
        change_points=[tau],
        details={"max_cap": float(cap[tau]) if len(cap) else 0.0},
        summary=summary,
    )


DETECTORS = {
    "Mann-Kendall": mann_kendall_test,
    "Pettitt": pettitt_test,
    "Sliding t-test": sliding_t_test,
    "Cramer": cramers_test,
    "Buishand": buishand_range_test,
    "Bayesian": bayesian_changepoint_detection,
    "CAP": cumulative_anomaly,
}
