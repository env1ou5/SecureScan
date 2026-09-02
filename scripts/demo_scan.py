"""End-to-end smoke test against the real model and the real API.

    python scripts/demo_scan.py

Unlike tests/backend/test_api_e2e.py, which stubs the predictor, this loads the
actual fine-tuned checkpoint and pushes a zip through the running application:
register -> login -> upload -> poll -> findings.

It is the check that the whole system works together, not just each half.
"""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

os.environ.setdefault("SECURESCAN_ENVIRONMENT", "development")
os.environ.setdefault("SECURESCAN_DATABASE_URL", f"sqlite+pysqlite:///{ROOT}/.storage/demo.db")
os.environ.setdefault("SECURESCAN_LOCAL_STORAGE_DIR", str(ROOT / ".storage" / "demo"))
os.environ.setdefault("SECURESCAN_JWT_SECRET", "demo-secret-not-for-production-000000")
os.environ.setdefault("SECURESCAN_MODEL_DIR", str(ROOT / "artifacts" / "unixcoder-v1"))

# A small application with a mix of real vulnerabilities and safe code.
DEMO_FILES = {
    "shop/db.py": """import sqlite3


def get_user(cursor, user_id):
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    return cursor.execute(query)


def get_user_safe(cursor, user_id):
    return cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
""",
    "shop/tools.py": """import subprocess


def convert_image(path):
    return subprocess.run(f"convert {path} out.png", shell=True)


def list_dir(path):
    return subprocess.run(["ls", "-la", path], shell=False)
""",
    "shop/store.py": """import os
import pickle


def load_session(blob):
    return pickle.loads(blob)


def read_upload(filename):
    with open(os.path.join("/srv/uploads", filename), "rb") as fh:
        return fh.read()
""",
    "shop/config.py": """import os

STRIPE_KEY = "sk_live_4eC39HqLyjWDarjtT1zdp7dc"
DATABASE_URL = os.environ["DATABASE_URL"]


def get_stripe_key():
    return STRIPE_KEY
""",
    "shop/utils.py": """def normalize(text):
    return text.strip().lower()


def chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]
""",
}


def build_archive() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in DEMO_FILES.items():
            zf.writestr(name, body)
    return buf.getvalue()


def main() -> int:
    model_dir = Path(os.environ["SECURESCAN_MODEL_DIR"])
    if not model_dir.exists():
        print(f"No checkpoint at {model_dir}. Run `make train` first.")
        return 1

    (ROOT / ".storage").mkdir(exist_ok=True)
    db = Path(os.environ["SECURESCAN_DATABASE_URL"].split("///")[-1])
    db.unlink(missing_ok=True)

    from app.db import init_db
    from fastapi.testclient import TestClient

    init_db()

    # Run the scan inline so the script is deterministic and exits when done.
    import app.api.scans as scans_api

    class InlineQueue:
        def enqueue(self, func, *args, **kwargs):
            func(*args, **kwargs)
            return "inline"

    scans_api.get_queue = lambda settings: InlineQueue()

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        creds = {"email": "demo@example.com", "password": "demo-password-1234"}
        client.post("/api/auth/register", json=creds)
        token = client.post(
            "/api/auth/login",
            data={"username": creds["email"], "password": creds["password"]},
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        started = time.perf_counter()
        resp = client.post(
            "/api/scans",
            files={"file": ("shop.zip", build_archive(), "application/zip")},
            headers=headers,
        )
        if resp.status_code != 202:
            print("upload failed:", resp.status_code, resp.text)
            return 1
        scan_id = resp.json()["scan_id"]

        scan = client.get(f"/api/scans/{scan_id}", headers=headers).json()
        elapsed = time.perf_counter() - started

        print(f"\nscan {scan_id[:8]}  status={scan['status']}  {elapsed:.1f}s")
        if scan["status"] == "FAILED":
            print("error:", scan["error"])
            return 1
        print(
            f"files={scan['files_scanned']} functions={scan['functions_scanned']} "
            f"findings={scan['findings_count']} model={scan['model_version']}"
        )

        findings = client.get(f"/api/scans/{scan_id}/findings", headers=headers).json()
        print("\n" + "=" * 74)
        for f in findings:
            cal = "" if f["confidence_calibrated"] else "  (UNCALIBRATED)"
            print(
                f"\n{f['severity']:8} {f['vulnerability_type']:24} "
                f"{f['confidence'] * 100:5.1f}%{cal}"
            )
            print(f"  {f['file_path']}:{f['anchor_line']}  in {f['function_name']}()")
            for line in f["contributing_lines"][:3]:
                bar = "#" * int(line["score"] * 12)
                print(f"    L{line['line']:<4} {bar:<12} {line['text'][:52]}")
            if f["remediation"]:
                print(f"  fix: {f['remediation']['title']}")
        print("\n" + "=" * 74)

        summary = client.get(f"/api/scans/{scan_id}/summary", headers=headers).json()
        print("by severity:", {s["severity"]: s["count"] for s in summary["by_severity"]})
        mean = summary["mean_confidence"]
        print(f"mean confidence: {mean:.3f}" if mean else "mean confidence: n/a")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
