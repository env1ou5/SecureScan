"""End-to-end API test: register -> login -> upload -> scan -> findings.

Runs against SQLite with a stubbed predictor, so it needs neither a database
server nor a checkpoint and is fast enough for CI. The scan itself runs
synchronously here (the in-process queue is replaced with a direct call) --
the async contract is exercised by asserting the 202 and the poll, not by
actually racing a thread in a test.
"""

from __future__ import annotations

import io
import zipfile

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

VULNERABLE_SOURCE = """import subprocess


def safe_helper(a, b):
    return a + b


def run_ping(host):
    return subprocess.run("ping " + host, shell=True)
"""


@pytest.fixture
def client(monkeypatch):
    """API client backed by a fresh schema, a stub model, and an inline queue."""
    from app.db import Base, engine, init_db

    Base.metadata.drop_all(bind=engine)
    init_db()

    # Stub the model: this test covers the HTTP, persistence, and ingestion
    # path, not inference quality.
    from securescan_ml.inference.predictor import Prediction
    from securescan_ml.labels import Label, severity_for

    class StubPredictor:
        def predict_chunks(self, chunks, **kwargs):
            out = []
            for chunk in chunks:
                vulnerable = "shell=True" in chunk.code
                label = Label.COMMAND_INJECTION if vulnerable else Label.SAFE
                out.append(
                    Prediction(
                        label=label,
                        severity=severity_for(label),
                        confidence=0.93 if vulnerable else 0.99,
                        calibrated=True,
                        file_path=chunk.file_path,
                        function_name=chunk.name,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        contributing_lines=[],
                        probabilities={label.value: 0.93},
                    )
                )
            return out

    import app.workers.scan_worker as worker

    monkeypatch.setattr(worker, "get_predictor", lambda *a, **k: StubPredictor())

    # Run the job inline so the test does not depend on thread timing. The 202
    # contract is still asserted; only the scheduling is made deterministic.
    import app.api.scans as scans_api

    class InlineQueue:
        def enqueue(self, func, *args, **kwargs):
            func(*args, **kwargs)
            return "inline"

    monkeypatch.setattr(scans_api, "get_queue", lambda settings: InlineQueue())

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as test_client:
        yield test_client

    Base.metadata.drop_all(bind=engine)


def make_archive() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project/handlers.py", VULNERABLE_SOURCE)
        zf.writestr("project/README.md", "docs")
    return buf.getvalue()


def auth_headers(client: TestClient, email: str = "dev@example.com") -> dict:
    client.post("/api/auth/register", json={"email": email, "password": "correct-horse-battery"})
    resp = client.post(
        "/api/auth/login", data={"username": email, "password": "correct-horse-battery"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_taxonomy_served_from_ml_package(client):
    labels = client.get("/api/taxonomy").json()["labels"]
    names = {row["name"] for row in labels}
    assert "SQL_INJECTION" in names
    assert len(labels) == 7
    sqli = next(row for row in labels if row["name"] == "SQL_INJECTION")
    assert sqli["severity"] == "CRITICAL"
    assert sqli["has_remediation"] is True


def test_scans_require_authentication(client):
    assert client.get("/api/scans").status_code == 401


def test_registration_rejects_duplicate_email(client):
    body = {"email": "dupe@example.com", "password": "correct-horse-battery"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json=body).status_code == 409


def test_login_failure_does_not_leak_account_existence(client):
    auth_headers(client, "real@example.com")
    unknown = client.post(
        "/api/auth/login", data={"username": "ghost@example.com", "password": "whatever12345"}
    )
    wrong = client.post(
        "/api/auth/login", data={"username": "real@example.com", "password": "wrongwrong123"}
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["detail"] == wrong.json()["detail"]


def test_full_scan_flow(client):
    headers = auth_headers(client)

    resp = client.post(
        "/api/scans",
        files={"file": ("project.zip", make_archive(), "application/zip")},
        headers=headers,
    )
    assert resp.status_code == 202, resp.text
    scan_id = resp.json()["scan_id"]
    assert resp.json()["status_url"] == f"/api/scans/{scan_id}"

    scan = client.get(f"/api/scans/{scan_id}", headers=headers).json()
    assert scan["status"] == "COMPLETED", scan.get("error")
    assert scan["files_scanned"] == 1
    assert scan["functions_scanned"] == 2
    assert scan["findings_count"] == 1

    findings = client.get(f"/api/scans/{scan_id}/findings", headers=headers).json()
    assert len(findings) == 1
    finding = findings[0]
    assert finding["vulnerability_type"] == "COMMAND_INJECTION"
    assert finding["severity"] == "CRITICAL"
    assert finding["function_name"] == "run_ping"
    # Remediation is attached from the ML package's template library.
    assert finding["remediation"]["title"].startswith("Avoid shell=True")

    summary = client.get(f"/api/scans/{scan_id}/summary", headers=headers).json()
    assert summary["total_findings"] == 1
    assert summary["by_severity"][0]["severity"] == "CRITICAL"
    assert summary["by_file"][0]["file_path"] == "project/handlers.py"


def test_tenant_isolation(client):
    """A scan must be invisible to another account, and 404 rather than 403."""
    alice = auth_headers(client, "alice@example.com")
    resp = client.post(
        "/api/scans",
        files={"file": ("p.zip", make_archive(), "application/zip")},
        headers=alice,
    )
    scan_id = resp.json()["scan_id"]

    bob = auth_headers(client, "bob@example.com")
    assert client.get(f"/api/scans/{scan_id}", headers=bob).status_code == 404
    assert client.get(f"/api/scans/{scan_id}/findings", headers=bob).json() == []
    assert client.get("/api/scans", headers=bob).json() == []


def test_dismiss_finding_removes_it_from_summary(client):
    headers = auth_headers(client)
    scan_id = client.post(
        "/api/scans",
        files={"file": ("p.zip", make_archive(), "application/zip")},
        headers=headers,
    ).json()["scan_id"]

    finding_id = client.get(f"/api/scans/{scan_id}/findings", headers=headers).json()[0]["id"]
    resp = client.post(
        f"/api/findings/{finding_id}/dismiss",
        json={"reason": "guarded by an allowlist upstream"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["dismissed"] is True

    assert client.get(f"/api/scans/{scan_id}/findings", headers=headers).json() == []
    assert (
        client.get(f"/api/scans/{scan_id}/summary", headers=headers).json()["total_findings"] == 0
    )


def test_malicious_archive_fails_the_scan_not_the_server(client):
    """A zip bomb must be rejected by the worker, leaving a FAILED scan."""
    headers = auth_headers(client)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.py", "A" * 20_000_000)

    resp = client.post(
        "/api/scans",
        files={"file": ("bomb.zip", buf.getvalue(), "application/zip")},
        headers=headers,
    )
    assert resp.status_code == 202
    scan = client.get(f"/api/scans/{resp.json()['scan_id']}", headers=headers).json()
    assert scan["status"] == "FAILED"
    assert "ratio" in scan["error"] or "uncompressed" in scan["error"]
