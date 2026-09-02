"""Job queue abstraction (proposal §2, D3).

The API contract is async from the first commit, but the queue behind it starts
as a thread pool. Switching to Redis/RQ is a config change (SECURESCAN_REDIS_URL)
with no change to the API, the schema, or the frontend -- which is the entire
reason for building the job schema up front.

The in-process backend is for local development only: jobs die with the process
and there is no retry. Production sets a Redis URL.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.config import Settings

log = logging.getLogger(__name__)


class JobQueue(ABC):
    @abstractmethod
    def enqueue(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class InProcessQueue(JobQueue):
    def __init__(self, max_workers: int = 2):
        # Single-digit workers on purpose: each holds a model in memory, and
        # CPU inference does not get faster by oversubscribing cores.
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scan")
        self._counter = 0

    def enqueue(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        self._counter += 1
        job_id = f"inproc-{self._counter}"

        def run() -> None:
            try:
                func(*args, **kwargs)
            except Exception:  # noqa: BLE001 - a dead thread must not be silent
                log.exception("job %s failed", job_id)

        self._pool.submit(run)
        return job_id

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


class RedisQueue(JobQueue):  # pragma: no cover - requires Redis
    def __init__(self, redis_url: str, queue_name: str = "scans"):
        from redis import Redis
        from rq import Queue

        self._queue = Queue(queue_name, connection=Redis.from_url(redis_url))

    def enqueue(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job = self._queue.enqueue(func, *args, job_timeout="30m", **kwargs)
        return job.id

    def shutdown(self) -> None:
        return None


_queue: JobQueue | None = None


def get_queue(settings: Settings) -> JobQueue:
    global _queue
    if _queue is None:
        if settings.use_redis:
            _queue = RedisQueue(settings.redis_url)
            log.info("job queue: redis at %s", settings.redis_url)
        else:
            _queue = InProcessQueue()
            log.warning("job queue: in-process (development only; jobs die with the process)")
    return _queue


def shutdown_queue() -> None:
    global _queue
    if _queue is not None:
        _queue.shutdown()
        _queue = None
