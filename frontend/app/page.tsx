"use client";

import { useEffect, useState } from "react";
import { listScans } from "@/lib/api";
import { UploadCard } from "@/components/UploadCard";
import type { Scan } from "@/lib/types";

export default function HomePage() {
  const [scans, setScans] = useState<Scan[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listScans().then(setScans).catch((e) => setError(e.message));
  }, []);

  return (
    <div className="container">
      <h1 style={{ fontSize: "1.4rem" }}>Scans</h1>

      <UploadCard />

      {error && (
        <div className="card" style={{ borderColor: "var(--critical)" }}>
          <strong>Could not load scans</strong>
          <p className="muted" style={{ margin: 0 }}>{error}</p>
          <p className="muted" style={{ fontSize: "0.85rem" }}>
            Sign in first, and confirm the API is running at{" "}
            <code>{process.env.NEXT_PUBLIC_API_URL}</code>.
          </p>
        </div>
      )}

      {!error && scans.length === 0 && (
        <p className="muted">No scans yet. Upload a repository archive to begin.</p>
      )}

      <div style={{ display: "grid", gap: "0.75rem" }}>
        {scans.map((scan) => (
          <a
            key={scan.id}
            href={`/scans/${scan.id}`}
            className="card"
            style={{ color: "inherit", textDecoration: "none" }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <strong>{scan.repository_name}</strong>
              <span className="muted">{scan.status}</span>
            </div>
            <div className="muted" style={{ fontSize: "0.85rem" }}>
              {scan.findings_count} findings · {scan.files_scanned} files ·{" "}
              {scan.functions_scanned} functions · model {scan.model_version}
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}
