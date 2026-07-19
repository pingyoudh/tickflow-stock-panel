"""Persistent-data catalog and single-pass storage inventory."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DataCategory = Literal["business", "research", "system"]
SyncMode = Literal["manual", "scheduled", "derived"]
ClearMode = Literal["all", "parquet_only"]

_DATE_PARTITION = re.compile(r"(?:^|/)date=(\d{4}-\d{2}-\d{2})(?:/|$)")


@dataclass(frozen=True)
class DataDimensionDef:
    id: str
    label: str
    category: DataCategory
    paths: tuple[str, ...] = ()
    children: tuple[DataDimensionDef, ...] = ()
    sensitive: bool = False
    clearable: bool = False
    clear_mode: ClearMode = "all"
    sync_mode: SyncMode | None = None
    count_rows: bool = False


def _dimension(
    id: str,
    label: str,
    category: DataCategory,
    *paths: str,
    children: tuple[DataDimensionDef, ...] = (),
    sensitive: bool = False,
    clearable: bool = False,
    clear_mode: ClearMode = "all",
    sync_mode: SyncMode | None = None,
    count_rows: bool = False,
) -> DataDimensionDef:
    return DataDimensionDef(
        id=id,
        label=label,
        category=category,
        paths=tuple(path.replace("\\", "/").strip("/") for path in paths),
        children=children,
        sensitive=sensitive,
        clearable=clearable,
        clear_mode=clear_mode,
        sync_mode=sync_mode,
        count_rows=count_rows,
    )


INDEX_CHILDREN = (
    _dimension(
        "index_instruments",
        "指数维表",
        "business",
        "instruments_index",
        clearable=True,
        sync_mode="scheduled",
        count_rows=True,
    ),
    _dimension(
        "index_daily",
        "指数日 K",
        "business",
        "kline_index_daily",
        clearable=True,
        sync_mode="scheduled",
    ),
    _dimension(
        "index_enriched",
        "指数指标",
        "business",
        "kline_index_enriched",
        clearable=True,
        sync_mode="derived",
    ),
)

ETF_CHILDREN = (
    _dimension(
        "etf_instruments",
        "ETF 维表",
        "business",
        "instruments_etf",
        clearable=True,
        sync_mode="scheduled",
        count_rows=True,
    ),
    _dimension(
        "etf_daily",
        "ETF 日 K",
        "business",
        "kline_etf_daily",
        clearable=True,
        sync_mode="scheduled",
    ),
    _dimension(
        "etf_enriched",
        "ETF 指标",
        "business",
        "kline_etf_enriched",
        clearable=True,
        sync_mode="derived",
    ),
    _dimension(
        "etf_minute",
        "ETF 分钟 K",
        "business",
        "kline_etf_minute",
        clearable=True,
        sync_mode="manual",
    ),
    _dimension(
        "etf_adj_factor",
        "ETF 复权因子",
        "business",
        "adj_factor_etf",
        clearable=True,
        sync_mode="scheduled",
        count_rows=True,
    ),
)

FINANCIAL_CHILDREN = tuple(
    _dimension(
        f"financial_{table}",
        label,
        "business",
        f"financials/{table}",
        clearable=True,
        sync_mode="scheduled",
        count_rows=True,
    )
    for table, label in (
        ("metrics", "财务指标"),
        ("income", "利润表"),
        ("balance_sheet", "资产负债表"),
        ("cash_flow", "现金流量表"),
    )
)

QUANT_CHILDREN = tuple(
    _dimension(id, label, "research", f"user_data/quant/{dirname}")
    for id, label, dirname in (
        ("quant_factor_cache", "因子缓存", "factor_cache"),
        ("quant_factors", "因子定义", "factors"),
        ("quant_runs", "实验运行", "runs"),
        ("quant_models", "模型", "models"),
        ("quant_predictions", "预测结果", "predictions"),
        ("quant_specs", "研究规格", "specs"),
        ("quant_strategies", "量化策略", "strategies"),
        ("quant_deleting", "删除暂存", ".deleting"),
    )
)

RESULT_CHILDREN = (
    _dimension("backtest_results", "回测结果", "research", "backtest_results"),
    _dimension("screener_results", "筛选结果", "research", "screener_results"),
)

AI_REPORT_CHILDREN = (
    _dimension("ai_financial_reports", "财务分析报告", "research", "user_data/ai_reports.json"),
    _dimension("ai_stock_reports", "个股分析报告", "research", "user_data/ai_stock_reports.json"),
    _dimension("ai_market_recaps", "市场复盘报告", "research", "user_data/ai_market_recaps.json"),
)

STRATEGY_CHILDREN = (
    _dimension("saved_strategies", "自定义与 AI 策略", "research", "strategies"),
    _dimension("strategy_overrides", "策略参数覆盖", "research", "user_data/strategy_overrides"),
    _dimension("custom_signals", "自定义信号", "research", "user_data/custom_signals"),
)

WORKSPACE_CHILDREN = (
    _dimension("watchlist", "自选股", "research", "user_data/watchlist.parquet"),
    _dimension("alerts", "告警历史", "research", "user_data/alerts.jsonl"),
    _dimension("monitor_rules", "监控规则", "research", "user_data/monitor_rules"),
)

CONFIG_PATHS = (
    "capabilities.json",
    "user_data/preferences.json",
    "user_data/secrets.json",
    "user_data/auth.json",
)

ALL_DIMENSIONS: tuple[DataDimensionDef, ...] = (
    _dimension(
        "instruments",
        "个股维表",
        "business",
        "instruments",
        clearable=True,
        sync_mode="scheduled",
        count_rows=True,
    ),
    _dimension(
        "daily",
        "日 K",
        "business",
        "kline_daily",
        clearable=True,
        sync_mode="scheduled",
    ),
    _dimension(
        "adj_factor",
        "除权因子",
        "business",
        "adj_factor",
        clearable=True,
        sync_mode="scheduled",
        count_rows=True,
    ),
    _dimension(
        "enriched",
        "Enriched",
        "business",
        "kline_daily_enriched",
        clearable=True,
        sync_mode="derived",
    ),
    _dimension(
        "minute",
        "分钟 K",
        "business",
        "kline_minute",
        clearable=True,
        sync_mode="manual",
    ),
    _dimension("index", "指数", "business", children=INDEX_CHILDREN),
    _dimension("etf", "ETF", "business", children=ETF_CHILDREN),
    _dimension("financials", "财务数据", "business", children=FINANCIAL_CHILDREN),
    _dimension(
        "depth5",
        "五档盘口",
        "business",
        "depth5",
        clearable=True,
        sync_mode="scheduled",
        count_rows=True,
    ),
    _dimension(
        "finance_news",
        "财联社快讯",
        "business",
        "finance_news",
        clearable=True,
        sync_mode="scheduled",
        count_rows=True,
    ),
    _dimension(
        "ext_data",
        "扩展数据",
        "business",
        "ext_data",
        "instruments_ext",
        "kline_ext",
        clearable=True,
        clear_mode="parquet_only",
        sync_mode="scheduled",
        count_rows=True,
    ),
    _dimension("quant_research", "量化研究", "research", children=QUANT_CHILDREN),
    _dimension("result_sets", "结果集", "research", children=RESULT_CHILDREN),
    _dimension("ai_reports", "AI 报告", "research", children=AI_REPORT_CHILDREN),
    _dimension("strategy_assets", "策略与信号", "research", children=STRATEGY_CHILDREN),
    _dimension("user_workspace", "自选与监控", "research", children=WORKSPACE_CHILDREN),
    _dimension("job_history", "任务历史", "system", "job_store"),
    _dimension("data_source_configs", "数据源定义", "system", "data_sources"),
    _dimension(
        "configuration",
        "配置与凭据",
        "system",
        *CONFIG_PATHS,
        sensitive=True,
    ),
    _dimension(
        "system_cache",
        "系统缓存",
        "system",
        "ai_cache",
        "user_data/strategy_cache.json",
    ),
    _dimension(
        "universe_pools",
        "标的池",
        "system",
        "pools",
        clearable=True,
        sync_mode="derived",
        count_rows=True,
    ),
)


@dataclass
class _Stats:
    files: int = 0
    parquet_files: int = 0
    size_bytes: int = 0
    records: int | None = None
    earliest_at: str | None = None
    latest_at: str | None = None
    last_modified_at: str | None = None

    def add_file(
        self,
        *,
        size: int,
        modified_at: str,
        is_parquet: bool,
        records: int | None,
        partition_date: str | None,
    ) -> None:
        self.files += 1
        self.size_bytes += size
        if is_parquet:
            self.parquet_files += 1
        if records is not None:
            self.records = (self.records or 0) + records
        if partition_date:
            if self.earliest_at is None or partition_date < self.earliest_at:
                self.earliest_at = partition_date
            if self.latest_at is None or partition_date > self.latest_at:
                self.latest_at = partition_date
        if self.last_modified_at is None or modified_at > self.last_modified_at:
            self.last_modified_at = modified_at

    def merge(self, other: _Stats) -> None:
        self.files += other.files
        self.parquet_files += other.parquet_files
        self.size_bytes += other.size_bytes
        if other.records is not None:
            self.records = (self.records or 0) + other.records
        if other.earliest_at and (
            self.earliest_at is None or other.earliest_at < self.earliest_at
        ):
            self.earliest_at = other.earliest_at
        if other.latest_at and (
            self.latest_at is None or other.latest_at > self.latest_at
        ):
            self.latest_at = other.latest_at
        if other.last_modified_at and (
            self.last_modified_at is None
            or other.last_modified_at > self.last_modified_at
        ):
            self.last_modified_at = other.last_modified_at


@dataclass
class CatalogSnapshot:
    dimensions: list[dict[str, Any]]
    category_totals: dict[str, dict[str, Any]]
    unclassified: dict[str, Any]
    total_size_mb: float
    legacy_storage: dict[str, Any] = field(default_factory=dict)


def _walk_dimensions(
    dimensions: tuple[DataDimensionDef, ...] = ALL_DIMENSIONS,
) -> list[DataDimensionDef]:
    result: list[DataDimensionDef] = []
    for definition in dimensions:
        result.append(definition)
        result.extend(_walk_dimensions(definition.children))
    return result


def dimension_by_id(dimension_id: str) -> DataDimensionDef | None:
    return next((item for item in _walk_dimensions() if item.id == dimension_id), None)


def clearable_dimensions() -> list[DataDimensionDef]:
    return [
        definition
        for definition in _walk_dimensions()
        if definition.clearable and definition.paths
    ]


def known_directory_roots() -> tuple[str, ...]:
    roots = {
        path.split("/", 1)[0]
        for definition in _walk_dimensions()
        for path in definition.paths
        if not Path(path).suffix
    }
    return tuple(sorted(roots))


def clear_registered_business_data(
    data_dir: Path,
    *,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Delete registered clearable files once while preserving catalog definitions."""
    root = Path(data_dir).resolve()
    excluded = exclude_ids or set()
    files: dict[Path, str] = {}

    for definition in clearable_dimensions():
        if definition.id in excluded:
            continue
        for relative in definition.paths:
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                logger.warning("跳过数据目录外的清理目标: %s", target)
                continue
            candidates = [target] if target.is_file() else (
                list(target.rglob("*")) if target.exists() else []
            )
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                if definition.clear_mode == "parquet_only" and (
                    candidate.suffix.lower() != ".parquet"
                ):
                    continue
                files.setdefault(candidate, definition.id)

    deleted_files = 0
    deleted_bytes = 0
    cleared_dimensions: set[str] = set()
    parents: set[Path] = set()
    for path, dimension_id in files.items():
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError as exc:
            logger.warning("清理持久化数据失败 %s: %s", path, exc)
            continue
        deleted_files += 1
        deleted_bytes += size
        cleared_dimensions.add(dimension_id)
        parents.add(path.parent)

    for parent in sorted(parents, key=lambda item: len(item.parts), reverse=True):
        current = parent
        while current != root:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    return {
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "cleared_dimension_ids": sorted(cleared_dimensions),
    }


def _parquet_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _matches(relative_path: str, registered_path: str) -> bool:
    return relative_path == registered_path or relative_path.startswith(
        registered_path + "/"
    )


def _unclassified_group(relative_path: str) -> str:
    parts = relative_path.split("/")
    if parts[0] == "user_data" and len(parts) > 1:
        return "/".join(parts[:2])
    return parts[0]


def scan_catalog(data_dir: Path) -> CatalogSnapshot:
    """Scan every persisted file once and assign it to one registered leaf."""
    root = Path(data_dir)
    definitions = _walk_dimensions()
    path_owners = sorted(
        (
            (path, definition)
            for definition in definitions
            for path in definition.paths
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    own_stats = {definition.id: _Stats() for definition in definitions}
    unclassified_stats = _Stats()
    unclassified_groups: set[str] = set()
    total_bytes = 0

    if root.exists():
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                known_directory = any(
                    _matches(relative, registered_path)
                    or _matches(registered_path, relative)
                    for registered_path, _definition in path_owners
                )
                if not known_directory:
                    unclassified_groups.add(_unclassified_group(relative))
                continue
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            total_bytes += stat.st_size
            owner = next(
                (
                    definition
                    for registered_path, definition in path_owners
                    if _matches(relative, registered_path)
                ),
                None,
            )
            modified_at = datetime.fromtimestamp(
                stat.st_mtime, tz=UTC
            ).isoformat(timespec="seconds")
            partition_match = _DATE_PARTITION.search(relative)
            partition_date = partition_match.group(1) if partition_match else None
            is_parquet = path.suffix.lower() == ".parquet"
            records = (
                _parquet_rows(path)
                if owner is not None and owner.count_rows and is_parquet
                else None
            )
            target = own_stats[owner.id] if owner is not None else unclassified_stats
            target.add_file(
                size=stat.st_size,
                modified_at=modified_at,
                is_parquet=is_parquet,
                records=records,
                partition_date=partition_date,
            )
            if owner is None:
                unclassified_groups.add(_unclassified_group(relative))

    def build(
        definition: DataDimensionDef,
    ) -> tuple[dict[str, Any], _Stats]:
        aggregate = _Stats()
        aggregate.merge(own_stats[definition.id])
        child_payloads = []
        for child in definition.children:
            child_payload, child_stats = build(child)
            child_payloads.append(child_payload)
            aggregate.merge(child_stats)
        payload = {
            "id": definition.id,
            "label": definition.label,
            "category": definition.category,
            "state": "ready" if aggregate.files else "empty",
            "records": aggregate.records,
            "files": aggregate.files,
            "parquet_files": aggregate.parquet_files,
            "size_mb": round(aggregate.size_bytes / 1048576, 2),
            "earliest_at": aggregate.earliest_at,
            "latest_at": aggregate.latest_at,
            "last_modified_at": aggregate.last_modified_at,
            "sensitive": definition.sensitive,
            "children": child_payloads,
        }
        if definition.sync_mode:
            payload["sync"] = {
                "mode": definition.sync_mode,
                "last_success_at": None,
                "next_run_at": None,
                "error": None,
            }
        return payload, aggregate

    dimensions = []
    category_totals = {
        category: {"files": 0, "parquet_files": 0, "size_mb": 0.0}
        for category in ("business", "research", "system")
    }
    top_stats: dict[str, _Stats] = {}
    for definition in ALL_DIMENSIONS:
        payload, stats = build(definition)
        dimensions.append(payload)
        top_stats[definition.id] = stats
        category = category_totals[definition.category]
        category["files"] += stats.files
        category["parquet_files"] += stats.parquet_files
        category["size_mb"] += stats.size_bytes / 1048576
    for category in category_totals.values():
        category["size_mb"] = round(category["size_mb"], 2)

    legacy_map = {
        "instruments": "instruments",
        "daily": "daily",
        "adj_factor": "adj_factor",
        "enriched": "enriched",
        "minute": "minute",
        "index_daily": "index_daily",
        "index_enriched": "index_enriched",
        "index_instruments": "index_instruments",
        "etf_daily": "etf_daily",
        "etf_enriched": "etf_enriched",
        "etf_instruments": "etf_instruments",
        "etf_adj_factor": "etf_adj_factor",
        "etf_minute": "etf_minute",
        "financials": "financials",
        "ext_data": "ext_data",
        "depth5": "depth5",
        "finance_news": "finance_news",
    }
    legacy_storage: dict[str, Any] = {}
    all_stats = {
        definition.id: own_stats[definition.id]
        for definition in definitions
    }
    all_stats.update(top_stats)
    for dimension_id, legacy_key in legacy_map.items():
        stats = all_stats.get(dimension_id, _Stats())
        legacy_storage[f"{legacy_key}_files"] = stats.files
        legacy_storage[f"{legacy_key}_size_mb"] = round(
            stats.size_bytes / 1048576, 2
        )
    legacy_storage["total_size_mb"] = round(total_bytes / 1048576, 2)
    legacy_storage["category_totals"] = category_totals

    if unclassified_groups:
        logger.warning(
            "发现未登记持久化数据分组: %s",
            ", ".join(sorted(unclassified_groups)),
        )

    return CatalogSnapshot(
        dimensions=dimensions,
        category_totals=category_totals,
        unclassified={
            "groups": len(unclassified_groups),
            "files": unclassified_stats.files,
            "size_mb": round(unclassified_stats.size_bytes / 1048576, 2),
        },
        total_size_mb=round(total_bytes / 1048576, 2),
        legacy_storage=legacy_storage,
    )


def apply_runtime_status(
    dimensions: list[dict[str, Any]],
    runtime: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay live service state without re-scanning the filesystem."""
    def apply(item: dict[str, Any]) -> dict[str, Any]:
        result = {**item, "children": [apply(child) for child in item["children"]]}
        live = runtime.get(result["id"])
        if not live:
            return result
        sync = {**(result.get("sync") or {})}
        sync.update(live.get("sync") or {})
        if sync:
            result["sync"] = sync
        for key in (
            "records",
            "earliest_at",
            "latest_at",
            "last_modified_at",
        ):
            if live.get(key) is not None:
                result[key] = live[key]
        if live.get("state"):
            result["state"] = live["state"]
        elif sync.get("error"):
            result["state"] = "error"
        elif live.get("syncing"):
            result["state"] = "syncing"
        return result

    return [apply(item) for item in dimensions]
