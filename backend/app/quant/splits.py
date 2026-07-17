"""Purged walk-forward split generation."""
from __future__ import annotations

from datetime import date

from app.quant.models import FoldDefinition, WalkForwardSpec


def generate_purged_folds(
    dates: list[date], spec: WalkForwardSpec, horizon: int
) -> list[FoldDefinition]:
    ordered = sorted(set(dates))
    required = spec.train_days + spec.validation_days + spec.test_days + horizon * 2
    if len(ordered) < required:
        raise ValueError(f"交易日不足: 至少需要 {required} 日, 当前只有 {len(ordered)} 日")
    folds: list[FoldDefinition] = []
    train_start = 0
    while True:
        train_end = train_start + spec.train_days
        validation_start = train_end + horizon
        validation_end = validation_start + spec.validation_days
        test_start = validation_end + horizon
        test_end = test_start + spec.test_days
        if test_end > len(ordered):
            break
        folds.append(FoldDefinition(
            index=len(folds),
            train_dates=ordered[train_start:train_end],
            validation_dates=ordered[validation_start:validation_end],
            test_dates=ordered[test_start:test_end],
        ))
        train_start += spec.step_days
    if not folds:
        raise ValueError("无法生成 Walk-forward 折, 请缩短窗口或扩大日期范围")
    return folds


def assert_no_label_overlap(fold: FoldDefinition, calendar: list[date], horizon: int) -> None:
    """Guard used by tests and training before a fold is accepted."""
    positions = {d: i for i, d in enumerate(sorted(set(calendar)))}
    if positions[fold.validation_dates[0]] - positions[fold.train_dates[-1]] <= horizon:
        raise ValueError("训练与验证标签窗口重叠")
    if positions[fold.test_dates[0]] - positions[fold.validation_dates[-1]] <= horizon:
        raise ValueError("验证与测试标签窗口重叠")
