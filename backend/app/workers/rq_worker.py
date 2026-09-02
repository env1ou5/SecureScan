"""RQ worker entry point for production (proposal §2, D3).

    python -m app.workers.rq_worker

Preloads the model before consuming the queue so the first job is not the slow
one. Runs as its own container alongside the API.
"""

from __future__ import annotations

import logging

from app.config import get_settings

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    if not settings.use_redis:
        raise SystemExit("SECURESCAN_REDIS_URL is required to run the RQ worker")

    from redis import Redis
    from rq import Queue, Worker

    from app.workers.scan_worker import get_predictor

    get_predictor(settings)  # pay the load cost before taking jobs

    connection = Redis.from_url(settings.redis_url)
    Worker([Queue("scans", connection=connection)], connection=connection).work()


if __name__ == "__main__":
    main()
