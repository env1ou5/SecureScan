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
      .catch(() => setStatus("FAILED"));

    return () => {
      cancelled = true;
    };
  }, [params.id]);

  if (!summary) {
    return (
      <div className="container">
        <p className="muted">Scan {status.toLowerCase()}…</p>
      </div>
    );
  }

  return (
    <div className="container">
      <h1 style={{ fontSize: "1.4rem" }}>{summary.scan.repository_name}</h1>
      <p className="muted" style={{ marginTop: 0 }}>
        {summary.total_findings} findings across {summary.scan.files_scanned} files ·{" "}
        {summary.scan.functions_scanned} functions analyzed · model{" "}
        {summary.scan.model_version}
        {summary.scan.duration_seconds !== null &&
          ` · ${summary.scan.duration_seconds.toFixed(1)}s`}
      </p>

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

      <h2 style={{ fontSize: "1.1rem" }}>Findings</h2>
      {findings.map((finding) => (
        <FindingCard key={finding.id} finding={finding} />
      ))}
    </div>
  );
}
