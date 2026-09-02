"use client";

import { useEffect, useState } from "react";
import { FindingCard } from "@/components/FindingCard";
import { SeverityBadge } from "@/components/SeverityBadge";
import { getSummary, listFindings, pollScan } from "@/lib/api";
import type { Finding, ScanSummary } from "@/lib/types";

export default function ScanPage({ params }: { params: { id: string } }) {
  const [summary, setSummary] = useState<ScanSummary | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [status, setStatus] = useState<string>("QUEUED");
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    // Poll until the job finishes, then load results (proposal D3).
    pollScan(params.id, (scan) => {
      if (!cancelled) setStatus(scan.status);
    })
      .then(async () => {
        if (cancelled) return;
        setSummary(await getSummary(params.id));
        setFindings(await listFindings(params.id));
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setStatus("FAILED");
        setLoadError(err instanceof Error ? err.message : "Could not load this scan");
      });

    return () => {
      cancelled = true;
    };
  }, [params.id]);

  if (!summary) {
    return (
      <div className="container">
        {loadError ? (
          <div className="card" style={{ borderColor: "var(--critical)" }}>
            <strong>Could not load scan</strong>
            <p className="muted" style={{ margin: 0 }}>{loadError}</p>
          </div>
        ) : (
          <p className="muted">Scan {status.toLowerCase()}…</p>
        )}
      </div>
    );
  }

  const { scan } = summary;

  return (
    <div className="container">
      <h1 style={{ fontSize: "1.4rem" }}>{scan.repository_name}</h1>

      {/* A failed scan has a stored reason. Show it instead of an empty
          dashboard that looks like "nothing found". */}
      {scan.status === "FAILED" && (
        <div className="card" style={{ borderColor: "var(--critical)", marginBottom: "1.5rem" }}>
          <strong>Scan failed</strong>
          <p style={{ margin: "0.25rem 0 0" }}>{scan.error ?? "No reason recorded."}</p>
          {scan.error?.includes("no Python files") && (
            <p className="muted" style={{ fontSize: "0.85rem", marginBottom: 0 }}>
              SecureScan analyzes Python only. Upload an archive containing{" "}
              <code>.py</code> files.
            </p>
          )}
        </div>
      )}

      {scan.status === "COMPLETED" && (
        <p className="muted" style={{ marginTop: 0 }}>
          {summary.total_findings} findings across {scan.files_scanned} files ·{" "}
          {scan.functions_scanned} functions analyzed · model {scan.model_version}
          {scan.duration_seconds !== null && ` · ${scan.duration_seconds.toFixed(1)}s`}
        </p>
      )}

      {scan.status === "COMPLETED" && summary.total_findings === 0 && (
        <div className="card" style={{ borderColor: "var(--safe)", marginBottom: "1.5rem" }}>
          <strong>No findings</strong>
          <p className="muted" style={{ margin: "0.25rem 0 0", fontSize: "0.85rem" }}>
            Nothing above the confidence threshold in {scan.functions_scanned} functions. This
            is not proof of safety — the model sees one function at a time and cannot follow
            data across function boundaries.
          </p>
        </div>
      )}

      {summary.by_severity.length > 0 && (
        <div
          style={{
            display: "grid",
            gap: "0.75rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
            marginBottom: "1.5rem",
          }}
        >
          {summary.by_severity.map(({ severity, count }) => (
            <div key={severity} className="card">
              <SeverityBadge severity={severity} />
              <div style={{ fontSize: "1.8rem", fontWeight: 700 }}>{count}</div>
            </div>
          ))}
        </div>
      )}

      {findings.length > 0 && <h2 style={{ fontSize: "1.1rem" }}>Findings</h2>}
      {findings.map((finding) => (
        <FindingCard key={finding.id} finding={finding} />
      ))}
    </div>
  );
}
