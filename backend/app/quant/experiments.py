"""Persistent local experiments and cancellable background execution."""
from __future__ import annotations

import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from app.quant.models import ExperimentManifest, MLBacktestSpec, MLSearchSpec, ModelSpec
from app.quant.trainer import MLTrainer


class ExperimentStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "user_data" / "quant" / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create(self, kind: str, spec: dict[str, Any]) -> ExperimentManifest:
        manifest = ExperimentManifest(run_id=uuid.uuid4().hex, kind=kind, spec=spec)
        (self.root / manifest.run_id).mkdir(parents=True, exist_ok=False)
        self.save(manifest)
        return manifest

    def get(self, run_id: str) -> ExperimentManifest:
        path = self.root / run_id / "manifest.json"
        if not path.exists():
            raise ValueError(f"实验不存在: {run_id}")
        return ExperimentManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, manifest: ExperimentManifest) -> None:
        manifest.updated_at = datetime.now()
        path = self.root / manifest.run_id / "manifest.json"
        with self._lock:
            temp = path.with_suffix(".tmp")
            temp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            temp.replace(path)

    def list(self) -> list[ExperimentManifest]:
        result = []
        for path in self.root.glob("*/manifest.json"):
            try:
                result.append(ExperimentManifest.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(result, key=lambda item: item.created_at, reverse=True)

    def delete(self, run_id: str) -> None:
        manifest = self.get(run_id)
        if manifest.status in {"queued", "running", "cancelling"}:
            raise ValueError("运行中的实验不能删除, 请先取消")
        shutil.rmtree(self.root / run_id)


class ExperimentManager:
    def __init__(
        self,
        store: ExperimentStore,
        trainer: MLTrainer,
        backtester: Any = None,
        searcher: Any = None,
    ) -> None:
        self.store = store
        self.trainer = trainer
        self.backtester = backtester
        self.searcher = searcher
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quant-ml")
        self._cancel: dict[str, threading.Event] = {}
        self._futures: dict[str, Any] = {}
        self._recover_interrupted()

    def submit_ml(self, spec: ModelSpec) -> ExperimentManifest:
        manifest = self.store.create("ml_training", spec.model_dump(mode="json"))
        event = threading.Event()
        self._cancel[manifest.run_id] = event
        self._futures[manifest.run_id] = self.executor.submit(
            self._execute_ml, manifest.run_id, spec, event
        )
        return manifest

    def submit_backtest(self, spec: MLBacktestSpec) -> ExperimentManifest:
        if self.backtester is None:
            raise ValueError("OOS 回测服务未初始化")
        manifest = self.store.create("ml_backtest", spec.model_dump(mode="json"))
        event = threading.Event()
        self._cancel[manifest.run_id] = event
        self._futures[manifest.run_id] = self.executor.submit(
            self._execute_backtest, manifest.run_id, spec, event
        )
        return manifest

    def submit_search(self, spec: MLSearchSpec) -> ExperimentManifest:
        if self.searcher is None:
            raise ValueError("智能训练服务未初始化")
        manifest = self.store.create("ml_search", spec.model_dump(mode="json"))
        event = threading.Event()
        self._cancel[manifest.run_id] = event
        self._futures[manifest.run_id] = self.executor.submit(
            self._execute_search, manifest.run_id, spec, event
        )
        return manifest

    def cancel(self, run_id: str) -> ExperimentManifest:
        manifest = self.store.get(run_id)
        if manifest.status not in {"queued", "running", "cancelling"}:
            return manifest
        event = self._cancel.setdefault(run_id, threading.Event())
        future = self._futures.get(run_id)
        if manifest.status == "queued" and future is not None and future.cancel():
            manifest.status = "cancelled"
            manifest.message = "训练已取消"
            self._cancel.pop(run_id, None)
            self._futures.pop(run_id, None)
        else:
            manifest.status = "cancelling"
            manifest.message = "正在停止当前计算"
        self.store.save(manifest)
        event.set()
        return manifest

    def rerun(self, run_id: str) -> ExperimentManifest:
        source = self.store.get(run_id)
        if source.kind == "ml_training":
            return self.submit_ml(ModelSpec.model_validate(source.spec))
        if source.kind == "ml_backtest":
            return self.submit_backtest(MLBacktestSpec.model_validate(source.spec))
        if source.kind == "ml_search":
            return self.submit_search(MLSearchSpec.model_validate(source.spec))
        raise ValueError("当前实验类型不支持重跑")

    def _execute_ml(self, run_id: str, spec: ModelSpec, cancelled: threading.Event) -> None:
        manifest = self.store.get(run_id)
        if cancelled.is_set():
            manifest.status = "cancelled"
            manifest.message = "训练已取消"
            self.store.save(manifest)
            return
        manifest.status = "running"
        self.store.save(manifest)

        def progress(value: float, message: str) -> None:
            if cancelled.is_set():
                raise InterruptedError("训练已取消")
            current = self.store.get(run_id)
            current.progress = value
            current.message = message
            self.store.save(current)

        try:
            result = self.trainer.run(spec, self.store.root / run_id, progress, cancelled)
            if cancelled.is_set():
                raise InterruptedError("训练已取消")
            manifest = self.store.get(run_id)
            manifest.status = "completed"
            manifest.progress = 1.0
            manifest.message = "训练完成"
            manifest.result = result
            manifest.warnings = result.get("warnings", [])
        except InterruptedError as exc:
            manifest = self.store.get(run_id)
            manifest.status = "cancelled"
            manifest.message = str(exc)
        except Exception as exc:
            manifest = self.store.get(run_id)
            manifest.status = "failed"
            manifest.error = str(exc)
            manifest.message = "训练失败"
        finally:
            if cancelled.is_set():
                manifest.status = "cancelled"
                manifest.message = "训练已取消"
            self.store.save(manifest)
            self._cancel.pop(run_id, None)
            self._futures.pop(run_id, None)

    def _execute_backtest(
        self, run_id: str, spec: MLBacktestSpec, cancelled: threading.Event
    ) -> None:
        manifest = self.store.get(run_id)
        if cancelled.is_set():
            manifest.status = "cancelled"
            manifest.message = "回测已取消"
            self.store.save(manifest)
            return
        manifest.status = "running"
        manifest.message = "正在准备 OOS 组合回测"
        self.store.save(manifest)

        def progress(value: float, message: str) -> None:
            if cancelled.is_set():
                raise InterruptedError("回测已取消")
            current = self.store.get(run_id)
            current.progress = value
            current.message = message
            self.store.save(current)

        try:
            result = self.backtester.run(
                spec, self.store.root / run_id, progress, cancelled
            )
            if cancelled.is_set():
                raise InterruptedError("回测已取消")
            manifest = self.store.get(run_id)
            manifest.status = "completed"
            manifest.progress = 1.0
            manifest.message = "OOS 组合回测完成"
            manifest.result = result
            manifest.warnings = result.get("warnings", [])
        except InterruptedError as exc:
            manifest = self.store.get(run_id)
            manifest.status = "cancelled"
            manifest.message = str(exc)
        except Exception as exc:
            manifest = self.store.get(run_id)
            manifest.status = "failed"
            manifest.error = str(exc)
            manifest.message = "OOS 组合回测失败"
        finally:
            if cancelled.is_set():
                manifest.status = "cancelled"
                manifest.message = "回测已取消"
            self.store.save(manifest)
            self._cancel.pop(run_id, None)
            self._futures.pop(run_id, None)

    def _execute_search(
        self, run_id: str, spec: MLSearchSpec, cancelled: threading.Event
    ) -> None:
        manifest = self.store.get(run_id)
        if cancelled.is_set():
            manifest.status = "cancelled"
            manifest.message = "智能训练已取消"
            self.store.save(manifest)
            return
        manifest.status = "running"
        manifest.message = "正在准备多因子智能训练"
        self.store.save(manifest)

        def progress(value: float, message: str) -> None:
            if cancelled.is_set():
                raise InterruptedError("智能训练已取消")
            current = self.store.get(run_id)
            current.progress = value
            current.message = message
            self.store.save(current)

        try:
            result = self.searcher.run(
                spec, self.store.root / run_id, progress, cancelled
            )
            if cancelled.is_set():
                raise InterruptedError("智能训练已取消")
            manifest = self.store.get(run_id)
            manifest.status = "completed"
            manifest.progress = 1.0
            manifest.message = "智能训练完成"
            manifest.result = result
            manifest.warnings = result.get("warnings", [])
        except InterruptedError as exc:
            manifest = self.store.get(run_id)
            manifest.status = "cancelled"
            manifest.message = str(exc)
        except Exception as exc:
            manifest = self.store.get(run_id)
            manifest.status = "failed"
            manifest.error = str(exc)
            manifest.message = "智能训练失败"
        finally:
            if cancelled.is_set():
                manifest.status = "cancelled"
                manifest.message = "智能训练已取消"
            self.store.save(manifest)
            self._cancel.pop(run_id, None)
            self._futures.pop(run_id, None)

    def _recover_interrupted(self) -> None:
        for manifest in self.store.list():
            if manifest.status in {"queued", "running", "cancelling"}:
                manifest.status = "cancelled"
                manifest.message = "后端重启, 实验已停止"
                self.store.save(manifest)
