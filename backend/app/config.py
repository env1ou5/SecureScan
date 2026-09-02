"""Application settings, loaded from the environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SECURESCAN_", extra="ignore")

    environment: str = "development"
    debug: bool = False

    database_url: str = "postgresql+psycopg://securescan:securescan@localhost:5432/securescan"

    # Empty means the in-process worker (local dev). Setting it switches to RQ
    # with no other code change -- that is the point of the D3 job schema.
    redis_url: str = ""

    # MUST be overridden in every non-development environment. The app refuses
    # to start with the default outside development; a shared dev secret that
    # leaks into production is the classic way this goes wrong.
    jwt_secret: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 60 * 12

    model_dir: str = "artifacts/unixcoder-v1"
    model_version: str = "unixcoder-v1"
    max_length: int = 512
    inference_batch_size: int = 16
    # Below this calibrated confidence a prediction is dropped rather than
    # shown. False positives are what get a scanner uninstalled (§10).
    min_confidence: float = 0.5

    # Archive limits. Untrusted input -- see services/ingest.py.
    max_upload_bytes: int = 50 * 1024 * 1024
    max_uncompressed_bytes: int = 500 * 1024 * 1024
    max_archive_members: int = 20_000
    max_compression_ratio: float = 100.0
    max_file_bytes: int = 2 * 1024 * 1024

    # Comma-separated list of allowed browser origins. Must be tightened from
    # the localhost default before any deployment.
    cors_origins: str = "http://localhost:3000"

    storage_backend: str = "local"  # local | s3
    local_storage_dir: str = "./.storage"
    s3_bucket: str = ""
    aws_region: str = "us-east-1"

    @property
    def use_redis(self) -> bool:
        return bool(self.redis_url)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def validate_for_environment(self) -> None:
        if self.environment != "development" and "dev-only" in self.jwt_secret:
            raise RuntimeError("SECURESCAN_JWT_SECRET must be set outside development")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_for_environment()
    return settings
