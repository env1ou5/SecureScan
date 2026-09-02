"use client";

/**
 * Confidence readout.
 *
 * When `calibrated` is false the number is a raw softmax score, which is
 * systematically overconfident. Saying so is the difference between a number
 * and a misleading number (proposal §2).
 */
export function ConfidenceBar({
  confidence,
  calibrated,
}: {
  confidence: number;
  calibrated: boolean;
}) {
  const percent = Math.round(confidence * 100);
  return (
    <div style={{ minWidth: 140 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem" }}>
        <span className="muted">{calibrated ? "Confidence" : "Confidence (uncalibrated)"}</span>
        <span>{percent}%</span>
      </div>
      <div
        style={{
          height: 4,
          background: "var(--border)",
          borderRadius: 2,
          overflow: "hidden",
          marginTop: 4,
        }}
      >
        <div
          style={{
            width: `${percent}%`,
            height: "100%",
            background: calibrated ? "var(--low)" : "var(--muted)",
          }}
        />
      </div>
    </div>
  );
}
