"""Out-of-sample model evaluation metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl


def rank_values(values: np.ndarray) -> np.ndarray:
    """Return average ranks so constant and tied inputs stay constant."""
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    ordered = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ordered[end] == ordered[start]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranked


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return None
    left_rank = rank_values(left[finite])
    right_rank = rank_values(right[finite])
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return None
    correlation = float(left_centered @ right_centered / denominator)
    return correlation if np.isfinite(correlation) else None


def evaluate_oos(frame: pl.DataFrame) -> dict[str, Any]:
    required = {"date", "symbol", "target", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"OOS 评价缺少列: {sorted(required - set(frame.columns))}")
    valid = frame.filter(
        pl.col("target").cast(pl.Float64, strict=False).is_finite()
        & pl.col("prediction").cast(pl.Float64, strict=False).is_finite()
    )
    if valid.is_empty():
        raise ValueError("没有可评价的 OOS 预测")
    target = valid["target"].to_numpy().astype(float)
    prediction = valid["prediction"].to_numpy().astype(float)
    error = prediction - target
    daily_ic: list[dict[str, Any]] = []
    daily_spread: list[float] = []
    yearly: dict[int, list[float]] = defaultdict(list)
    turnover_values: list[float] = []
    previous_top: set[str] | None = None
    for group in valid.sort(["date", "symbol"]).partition_by("date", maintain_order=True):
        if group.height < 3:
            continue
        y = group["target"].to_numpy().astype(float)
        p = group["prediction"].to_numpy().astype(float)
        ic = spearman_correlation(p, y)
        if ic is not None:
            day = group["date"][0]
            daily_ic.append({"date": str(day), "ic": ic})
            yearly[day.year].append(ic)
        count = max(1, group.height // 10)
        ranked = group.sort("prediction")
        daily_spread.append(float(ranked.tail(count)["target"].mean() - ranked.head(count)["target"].mean()))
        current_top = set(ranked.tail(count)["symbol"].to_list())
        if previous_top is not None:
            turnover_values.append(1 - len(current_top & previous_top) / max(1, len(current_top)))
        previous_top = current_top
    ic_values = np.array([item["ic"] for item in daily_ic], dtype=float)
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "rank_ic": float(np.mean(ic_values)) if len(ic_values) else None,
        "icir": float(np.mean(ic_values) / ic_std) if ic_std > 1e-12 else None,
        "ic_positive_rate": float(np.mean(ic_values > 0)) if len(ic_values) else None,
        "top_bottom_return": float(np.mean(daily_spread)) if daily_spread else None,
        "turnover": float(np.mean(turnover_values)) if turnover_values else None,
        "coverage": float(valid.height / max(1, frame.height)),
        "annual_stability": {str(year): float(np.mean(values)) for year, values in sorted(yearly.items())},
        "daily_ic": daily_ic,
    }
