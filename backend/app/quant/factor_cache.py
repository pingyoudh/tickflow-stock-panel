"""Persistent point-in-time raw factor values for repeated AutoML research."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from app.quant.factors import FactorRegistry
from app.quant.models import FactorDefinition, ModelSpec

_CACHE_SEMANTIC_VERSION = 1


class FactorValueCache:
    def __init__(
        self,
        data_dir: Path,
        registry: FactorRegistry,
        *,
        max_bytes: int = 10 * 1024**3,
    ) -> None:
        self.data_dir = data_dir
        self.registry = registry
        self.root = data_dir / "user_data" / "quant" / "factor_cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._active: set[Path] = set()

    def get_or_compute(
        self,
        spec: ModelSpec,
        definition: FactorDefinition,
        panel: pl.DataFrame,
        *,
        load_values: bool = True,
        source_state: dict[str, list[int]] | None = None,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        directory = self._factor_dir(spec, definition)
        with self._active_path(directory):
            return self._get_or_compute(
                spec,
                definition,
                panel,
                directory,
                load_values=load_values,
                source_state=source_state,
            )

    def _get_or_compute(
        self,
        spec: ModelSpec,
        definition: FactorDefinition,
        panel: pl.DataFrame,
        directory: Path,
        *,
        load_values: bool,
        source_state: dict[str, list[int]] | None,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        values_path = directory / "values.parquet"
        manifest_path = directory / "manifest.json"
        requested_min = panel["date"].min()
        requested_max = panel["date"].max()
        source_state = source_state or self.source_state(
            spec.asset_type, requested_min, requested_max
        )
        manifest = self._read_manifest(manifest_path)
        previous_state = self._comparable_source_state(
            manifest.get("source_partitions", {}) if manifest else {},
            source_state,
        )
        changed_dates = self._changed_dates(
            previous_state,
            source_state,
        )
        metadata_covers = (
            manifest is not None
            and self._manifest_covers(manifest, requested_min, requested_max)
        )
        metadata_hit = (
            manifest is not None
            and manifest.get("semantic_version") == _CACHE_SEMANTIC_VERSION
            and manifest.get("factor_version") == definition.version
            and not changed_dates
            and metadata_covers
            and values_path.exists()
            and values_path.stat().st_size > 0
        )
        if metadata_hit and not load_values:
            self._touch_manifest(manifest_path, manifest)
            return pl.DataFrame(), {
                "factor_id": definition.id,
                "hit": True,
                "rows": int(manifest.get("rows", 0)),
                "bytes_written": 0,
                "metadata_only": True,
            }

        cached = self._read_values(values_path)
        cache_covers = (
            not cached.is_empty()
            and cached["date"].min() <= requested_min
            and cached["date"].max() >= requested_max
        )
        exact_hit = metadata_hit and cache_covers
        if exact_hit:
            self._touch_manifest(manifest_path, manifest)
            selected = cached.filter(
                (pl.col("date") >= requested_min)
                & (pl.col("date") <= requested_max)
            )
            return selected, {
                "factor_id": definition.id,
                "hit": True,
                "rows": selected.height,
                "bytes_written": 0,
            }

        recompute_start = requested_min
        if not cached.is_empty():
            if requested_min < cached["date"].min():
                recompute_start = requested_min
            elif changed_dates:
                recompute_start = min(changed_dates)
            elif requested_max > cached["date"].max():
                recompute_start = cached["date"].max() + timedelta(days=1)
        warmup_start = recompute_start - timedelta(
            days=max(20, definition.warmup * 2)
        )
        evaluation_panel = panel.filter(pl.col("date") >= warmup_start)
        evaluated = self.registry.evaluate(evaluation_panel, definition)
        if definition.id not in evaluated.columns:
            raise ValueError(f"因子 {definition.id} 没有生成输出列")
        fresh = (
            evaluated.select([
                "symbol",
                "date",
                pl.col(definition.id)
                .cast(pl.Float64, strict=False)
                .alias("value"),
            ])
            .filter(pl.col("date") >= recompute_start)
            .sort(["date", "symbol"])
        )
        if not cached.is_empty():
            cached = cached.filter(pl.col("date") < recompute_start)
            values = pl.concat([cached, fresh], how="diagonal_relaxed")
        else:
            values = fresh
        values = (
            values.unique(subset=["symbol", "date"], keep="last")
            .sort(["date", "symbol"])
        )
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_write_parquet(values, values_path)
        payload = {
            "semantic_version": _CACHE_SEMANTIC_VERSION,
            "factor_id": definition.id,
            "factor_version": definition.version,
            "asset_type": spec.asset_type,
            "universe_fingerprint": self.universe_fingerprint(spec),
            "source_partitions": source_state,
            "min_date": str(values["date"].min()) if not values.is_empty() else None,
            "max_date": str(values["date"].max()) if not values.is_empty() else None,
            "rows": values.height,
            "updated_at": datetime.now().isoformat(),
            "last_accessed_at": datetime.now().isoformat(),
        }
        self._atomic_write_json(payload, manifest_path)
        written = values_path.stat().st_size if values_path.exists() else 0
        selected_values = values.filter(
            (pl.col("date") >= requested_min)
            & (pl.col("date") <= requested_max)
        )
        selected = selected_values if load_values else pl.DataFrame()
        return selected, {
            "factor_id": definition.id,
            "hit": False,
            "rows": selected_values.height,
            "bytes_written": written,
            "recompute_start": str(recompute_start),
        }

    def load(
        self,
        spec: ModelSpec,
        definition: FactorDefinition,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        directory = self._factor_dir(spec, definition)
        values = self._read_values(directory / "values.parquet")
        if values.is_empty():
            raise ValueError(f"因子缓存不存在: {definition.id}")
        if start is not None:
            values = values.filter(pl.col("date") >= start)
        if end is not None:
            values = values.filter(pl.col("date") <= end)
        manifest_path = directory / "manifest.json"
        manifest = self._read_manifest(manifest_path)
        if manifest:
            self._touch_manifest(manifest_path, manifest)
        return values

    def status(self) -> dict[str, Any]:
        entries = self._entries()
        total = sum(item["bytes"] for item in entries)
        return {
            "max_bytes": self.max_bytes,
            "used_bytes": total,
            "used_ratio": total / self.max_bytes if self.max_bytes else 0.0,
            "entries": len(entries),
            "active_entries": len(self._active),
            "oldest_accessed_at": min(
                (item["last_accessed_at"] for item in entries), default=None
            ),
        }

    def clear(self) -> dict[str, Any]:
        removed_entries = 0
        removed_bytes = 0
        with self._lock:
            for item in self._entries():
                path = item["path"]
                if path in self._active:
                    continue
                removed_entries += 1
                removed_bytes += item["bytes"]
                shutil.rmtree(path, ignore_errors=False)
        return {
            "cleared": True,
            "entries_removed": removed_entries,
            "bytes_removed": removed_bytes,
        }

    def evict(self) -> dict[str, int]:
        with self._lock:
            entries = self._entries()
            total = sum(item["bytes"] for item in entries)
            removed = 0
            for item in sorted(
                entries, key=lambda value: value["last_accessed_at"] or ""
            ):
                if total <= self.max_bytes:
                    break
                path = item["path"]
                if path in self._active:
                    continue
                shutil.rmtree(path, ignore_errors=True)
                total -= item["bytes"]
                removed += 1
            return {"entries_removed": removed, "used_bytes": max(0, total)}

    def universe_fingerprint(self, spec: ModelSpec) -> str:
        payload = {
            "asset_type": spec.asset_type,
            "symbols": sorted(set(spec.symbols or [])) if spec.symbols else "all",
            "universe_filters": spec.universe_filters,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]

    def inspect(
        self,
        spec: ModelSpec,
        definitions: list[FactorDefinition],
    ) -> dict[str, Any]:
        hits = 0
        misses = 0
        bytes_present = 0
        min_history_days = int(
            spec.universe_filters.get("min_history_days", 0) or 0
        )
        min_median_amount = float(
            spec.universe_filters.get("min_median_amount_20d", 0.0) or 0.0
        )
        warmup = max([
            120,
            min_history_days,
            20 if min_median_amount > 0 else 0,
            *(item.warmup for item in definitions),
        ])
        inspect_start = spec.start - timedelta(days=warmup * 2)
        source_state = self.source_state(
            spec.asset_type, inspect_start, spec.end
        )
        available_dates = [
            parsed
            for name in source_state
            if (parsed := self._path_date(Path(name))) is not None
        ]
        available_end = max(available_dates, default=spec.end)
        for definition in definitions:
            directory = self._factor_dir(spec, definition)
            values = directory / "values.parquet"
            manifest = self._read_manifest(directory / "manifest.json")
            previous_state = self._comparable_source_state(
                manifest.get("source_partitions", {}) if manifest else {},
                source_state,
            )
            if values.exists() and manifest and (
                manifest.get("factor_version") == definition.version
                and manifest.get("semantic_version") == _CACHE_SEMANTIC_VERSION
                and previous_state == source_state
                and self._manifest_covers(manifest, spec.start, available_end)
            ):
                hits += 1
                bytes_present += values.stat().st_size
            else:
                misses += 1
        return {
            "factor_hits": hits,
            "factor_misses": misses,
            "hit_ratio": hits / max(1, len(definitions)),
            "bytes_present": bytes_present,
            **self.status(),
        }

    @staticmethod
    def _manifest_covers(
        manifest: dict[str, Any], start: date, end: date
    ) -> bool:
        try:
            return (
                date.fromisoformat(manifest["min_date"]) <= start
                and date.fromisoformat(manifest["max_date"]) >= end
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _factor_dir(
        self, spec: ModelSpec, definition: FactorDefinition
    ) -> Path:
        return (
            self.root
            / spec.asset_type
            / self.universe_fingerprint(spec)
            / definition.id
            / definition.version
        )

    def source_state(
        self, asset_type: str, start: date, end: date
    ) -> dict[str, list[int]]:
        dirname = (
            "kline_etf_enriched"
            if asset_type == "etf"
            else "kline_daily_enriched"
        )
        base = self.data_dir / dirname
        result: dict[str, list[int]] = {}
        for path in base.rglob("*.parquet"):
            partition_date = self._path_date(path)
            if partition_date is not None and not (start <= partition_date <= end):
                continue
            stat = path.stat()
            result[str(path.relative_to(base))] = [stat.st_size, stat.st_mtime_ns]
        return result

    @classmethod
    def _comparable_source_state(
        cls,
        previous: dict[str, list[int]],
        current: dict[str, list[int]],
    ) -> dict[str, list[int]]:
        current_dates = [
            parsed
            for key in current
            if (parsed := cls._path_date(Path(key))) is not None
        ]
        if not current_dates:
            return previous
        start = min(current_dates)
        end = max(current_dates)
        return {
            key: value
            for key, value in previous.items()
            if (
                (parsed := cls._path_date(Path(key))) is None
                or start <= parsed <= end
            )
        }

    @classmethod
    def _changed_dates(
        cls,
        previous: dict[str, list[int]],
        current: dict[str, list[int]],
    ) -> list[date]:
        changed = {
            key for key in set(previous) | set(current)
            if previous.get(key) != current.get(key)
        }
        return sorted({
            parsed for key in changed
            if (parsed := cls._path_date(Path(key))) is not None
        })

    @staticmethod
    def _path_date(path: Path) -> date | None:
        for part in path.parts:
            if not part.startswith("date="):
                continue
            try:
                return date.fromisoformat(part.split("=", 1)[1])
            except ValueError:
                return None
        return None

    def _entries(self) -> list[dict[str, Any]]:
        result = []
        for manifest_path in self.root.glob("*/*/*/*/manifest.json"):
            directory = manifest_path.parent
            manifest = self._read_manifest(manifest_path) or {}
            result.append({
                "path": directory.resolve(strict=False),
                "bytes": sum(
                    item.stat().st_size
                    for item in directory.rglob("*")
                    if item.is_file()
                ),
                "last_accessed_at": manifest.get("last_accessed_at", ""),
            })
        return result

    @contextmanager
    def _active_path(self, path: Path) -> Iterator[None]:
        resolved = path.resolve(strict=False)
        with self._lock:
            self._active.add(resolved)
        try:
            yield
        finally:
            with self._lock:
                self._active.discard(resolved)

    @staticmethod
    def _read_values(path: Path) -> pl.DataFrame:
        if not path.exists():
            return pl.DataFrame()
        try:
            return pl.read_parquet(path)
        except Exception:
            return pl.DataFrame()

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _touch_manifest(path: Path, payload: dict[str, Any]) -> None:
        FactorValueCache._atomic_write_json(
            {**payload, "last_accessed_at": datetime.now().isoformat()},
            path,
        )

    @staticmethod
    def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
        temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            frame.write_parquet(temp)
            FactorValueCache._replace_with_retry(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
        temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            FactorValueCache._replace_with_retry(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _replace_with_retry(source: Path, target: Path) -> None:
        for attempt in range(20):
            try:
                source.replace(target)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.025)
