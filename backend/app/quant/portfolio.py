"""Long-only portfolio optimization with explicit score-weight fallback."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class PortfolioResult:
    weights: dict[str, float]
    objective: str
    success: bool
    warnings: list[str] = field(default_factory=list)


class PortfolioOptimizer:
    def optimize(
        self, symbols: list[str], scores: np.ndarray, returns: np.ndarray,
        objective: Literal["equal", "score_weight", "min_variance", "max_sharpe", "min_tracking_error"],
        *, previous_weights: dict[str, float] | None = None, benchmark_returns: np.ndarray | None = None,
        industries: list[str | None] | None = None, max_positions: int = 10,
        max_weight: float = 0.2, industry_cap: float = 0.3, turnover_cap: float = 0.5,
        asset_type: str = "stock",
    ) -> PortfolioResult:
        if len(symbols) == 0:
            raise ValueError("候选标的为空")
        selected = np.argsort(np.nan_to_num(scores, nan=-np.inf))[-max_positions:]
        names = [symbols[i] for i in selected]
        selected_scores = scores[selected]
        if objective == "equal":
            weights = np.full(len(names), 1 / len(names))
            return PortfolioResult(dict(zip(names, weights, strict=True)), objective, True)
        fallback = self._score_weights(selected_scores, max_weight)
        if objective == "score_weight":
            return PortfolioResult(dict(zip(names, fallback, strict=True)), objective, True)
        warnings: list[str] = []
        try:
            from scipy.optimize import minimize
            selected_returns = returns[:, selected]
            covariance = np.cov(selected_returns, rowvar=False)
            diagonal = np.diag(np.diag(covariance))
            covariance = 0.8 * covariance + 0.2 * diagonal + np.eye(len(names)) * 1e-8
            expected = self._calibrate_expected(selected_scores, selected_returns)
            previous = np.array([(previous_weights or {}).get(name, 0.0) for name in names])

            def loss(weights: np.ndarray) -> float:
                if objective == "min_variance":
                    return float(weights @ covariance @ weights)
                if objective == "max_sharpe":
                    volatility = np.sqrt(max(1e-12, weights @ covariance @ weights))
                    return float(-(weights @ expected) / volatility)
                if benchmark_returns is None or len(benchmark_returns) != selected_returns.shape[0]:
                    raise ValueError("最小跟踪误差必须提供同区间基准收益")
                active = selected_returns @ weights - benchmark_returns
                return float(np.var(active))

            constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            constraints.append({"type": "ineq", "fun": lambda w: turnover_cap - 0.5 * np.abs(w - previous).sum()})
            if asset_type == "stock" and industries:
                selected_industries = [industries[i] for i in selected]
                for industry in {item for item in selected_industries if item}:
                    mask = np.array([item == industry for item in selected_industries], dtype=float)
                    constraints.append({"type": "ineq", "fun": lambda w, m=mask: industry_cap - w @ m})
            result = minimize(
                loss, fallback, method="SLSQP", bounds=[(0.0, max_weight)] * len(names),
                constraints=constraints, options={"maxiter": 500, "ftol": 1e-9},
            )
            if not result.success or not np.all(np.isfinite(result.x)):
                raise ValueError(result.message)
            return PortfolioResult(dict(zip(names, result.x, strict=True)), objective, True)
        except Exception as exc:
            warnings.append(f"{objective} 求解失败, 已回退因子分数加权: {exc}")
            return PortfolioResult(dict(zip(names, fallback, strict=True)), objective, False, warnings)

    @staticmethod
    def _score_weights(scores: np.ndarray, max_weight: float) -> np.ndarray:
        shifted = np.nan_to_num(scores - np.nanmin(scores), nan=0.0) + 1e-8
        weights = shifted / shifted.sum()
        for _ in range(20):
            over = weights > max_weight
            if not over.any():
                break
            excess = float((weights[over] - max_weight).sum())
            weights[over] = max_weight
            under = ~over
            if under.any():
                room = max_weight - weights[under]
                weights[under] += excess * room / room.sum()
        if abs(weights.sum() - 1) > 1e-6:
            raise ValueError("单标的上限与候选数量组合不可行")
        return weights

    @staticmethod
    def _calibrate_expected(scores: np.ndarray, returns: np.ndarray) -> np.ndarray:
        realized = np.nanmean(returns, axis=0)
        if np.std(scores) < 1e-12:
            return realized
        design = np.column_stack([np.ones(len(scores)), scores])
        return design @ np.linalg.lstsq(design, realized, rcond=None)[0]
