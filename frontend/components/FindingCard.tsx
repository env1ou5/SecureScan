"use client";

import { useState } from "react";
import type { Finding } from "@/lib/types";
import { ConfidenceBar } from "./ConfidenceBar";
import { SeverityBadge } from "./SeverityBadge";

export function FindingCard({ finding }: { finding: Finding }) {
  const [showFix, setShowFix] = useState(false);

  // Attribution scores are keyed by absolute file line.
  const scoreByLine = new Map(finding.contributing_lines.map((l) => [l.line, l.score]));
  const lines = (finding.code_snippet ?? "").split("\n");

  return (
    <article className="card" style={{ marginBottom: "1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
        <div>
          <SeverityBadge severity={finding.severity} />{" "}
          <strong>{finding.vulnerability_type.replaceAll("_", " ")}</strong>
          <div className="muted" style={{ fontSize: "0.85rem" }}>
            {finding.file_path}:{finding.anchor_line} · {finding.function_name}()
          </div>
        </div>
        <ConfidenceBar
          confidence={finding.confidence}
          calibrated={finding.confidence_calibrated}
        />
      </div>

      {lines.length > 0 && (
        <pre
          style={{
            marginTop: "1rem",
            padding: "0.75rem",
            background: "var(--bg)",
            borderRadius: 6,
            overflowX: "auto",
          }}
        >
          {lines.map((text, i) => {
            const lineNo = finding.start_line + i;
            const score = scoreByLine.get(lineNo);
            return (
              <div
                key={lineNo}
                className={`code-line${score !== undefined ? " attributed" : ""}`}
                style={score !== undefined ? ({ "--score": score } as React.CSSProperties) : undefined}
              >
                <span className="lineno">{lineNo}</span>
                <span>{text}</span>
              </div>
            );
          })}
        </pre>
      )}

      {finding.contributing_lines.length > 0 && (
        <p className="muted" style={{ fontSize: "0.78rem" }}>
          Highlighted lines are what most influenced the prediction. Attribution is
          correlational, not proof of exploitability.
        </p>
      )}

      {finding.remediation && (
        <div style={{ marginTop: "0.75rem" }}>
          <button
            onClick={() => setShowFix((v) => !v)}
            style={{
              background: "none",
              border: "1px solid var(--border)",
              color: "var(--text)",
              borderRadius: 6,
              padding: "0.35rem 0.7rem",
              cursor: "pointer",
              fontSize: "0.8rem",
            }}
          >
            {showFix ? "Hide" : "Show"} suggested fix
          </button>
          {showFix && (
            <div style={{ marginTop: "0.75rem" }}>
              <strong style={{ fontSize: "0.9rem" }}>{finding.remediation.title}</strong>
              <p className="muted" style={{ fontSize: "0.85rem" }}>
                {finding.remediation.explanation}
              </p>
              <div style={{ display: "grid", gap: "0.75rem", gridTemplateColumns: "1fr 1fr" }}>
                <div>
                  <div className="muted" style={{ fontSize: "0.75rem" }}>Unsafe</div>
                  <pre style={{ background: "var(--bg)", padding: "0.6rem", borderRadius: 6, overflowX: "auto" }}>
                    {finding.remediation.unsafe_example}
                  </pre>
                </div>
                <div>
                  <div className="muted" style={{ fontSize: "0.75rem" }}>Safer</div>
                  <pre style={{ background: "var(--bg)", padding: "0.6rem", borderRadius: 6, overflowX: "auto" }}>
                    {finding.remediation.safe_example}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </article>
  );
}
