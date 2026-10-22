from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass
from typing import Callable

from ..durability import is_durable_persistence_failure
from .types import SubagentResult


@dataclass(frozen=True)
class SubagentExecutor:
    max_parallel_workers: int = 4
    worker_timeout_seconds: float = 30.0

    def execute_batch(
        self,
        *,
        items: list[dict],
        run_item: Callable[[int, dict], SubagentResult],
        failure_event: threading.Event | None = None,
    ) -> list[SubagentResult]:
        if not items:
            return []
        batch_failure = failure_event or threading.Event()
        failure_lock = threading.Lock()
        first_durable_failure: list[BaseException] = []
        first_ordinary_failure: list[BaseException] = []

        def _run_guarded(index: int, item: dict) -> SubagentResult:
            if batch_failure.is_set():
                raise concurrent.futures.CancelledError()
            try:
                return run_item(index, item)
            except BaseException as error:  # noqa: BLE001 - Future boundary
                if is_durable_persistence_failure(error):
                    with failure_lock:
                        if not first_durable_failure:
                            first_durable_failure.append(error)
                            batch_failure.set()
                raise

        max_workers = max(1, min(int(self.max_parallel_workers), len(items)))
        results: list[SubagentResult | None] = [None] * len(items)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_guarded, index, dict(item)): index
                for index, item in enumerate(items)
            }
            try:
                for future in concurrent.futures.as_completed(
                    futures,
                    timeout=float(self.worker_timeout_seconds),
                ):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except concurrent.futures.CancelledError:
                        if first_durable_failure:
                            continue
                        raise
                    except BaseException as error:  # noqa: BLE001 - Future boundary
                        # Join running workers before propagating the original
                        # failure; a durable failure takes precedence below.
                        if is_durable_persistence_failure(error):
                            batch_failure.set()
                            for pending in futures:
                                if pending is not future:
                                    pending.cancel()
                            continue
                        if first_durable_failure:
                            continue
                        if not first_ordinary_failure:
                            first_ordinary_failure.append(error)
            except concurrent.futures.TimeoutError:
                for future, index in futures.items():
                    if results[index] is None:
                        future.cancel()
                        results[index] = SubagentResult(
                            mode="worker",
                            agent_name="",
                            template_name=None,
                            status="timeout",
                            error="worker batch timed out",
                        )
        if first_durable_failure:
            raise first_durable_failure[0]
        if first_ordinary_failure:
            raise first_ordinary_failure[0]
        return [result if isinstance(result, SubagentResult) else SubagentResult(mode="worker", agent_name="", template_name=None, status="failed", error="worker execution failed") for result in results]
