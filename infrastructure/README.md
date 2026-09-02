# Infrastructure

## Current

- `docker/worker.Dockerfile` — scan worker image. Same contents as the API
  image, different entry point, so both always run identical model and parsing
  code.
- `../backend/Dockerfile` — API image. CPU-only: no CUDA base, no GPU at serve
  time (proposal D4).
- `../docker-compose.yml` — local stack. Default profile runs API + Postgres
  with an in-process worker; `--profile queue` adds Redis and a worker
  container, matching production topology.

## Not yet written

Terraform for the AWS target, deferred to roadmap Phase 6:

- ECS Fargate services for `api` and `worker` (CPU tasks; no GPU)
- RDS PostgreSQL
- ElastiCache Redis
- S3 buckets for checkpoints and datasets
- ALB + ACM certificate
- Secrets Manager for `SECURESCAN_JWT_SECRET` and database credentials

Training infrastructure is deliberately **not** included. Training runs on a
rented GPU by the hour; nothing in the serving path needs one.
