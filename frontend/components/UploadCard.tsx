"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { uploadScan } from "@/lib/api";

/**
 * Repository upload.
 *
 * POST returns 202 with a scan id (proposal D3), so this navigates straight to
 * the scan page and lets that page poll. Nothing here waits on the scan.
 */
export function UploadCard() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  async function send(file: File) {
    setError(null);
    setBusy(true);
    try {
      const { scan_id } = await uploadScan(file);
      router.push(`/scans/${scan_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="card"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) void send(file);
      }}
      style={{
        borderStyle: "dashed",
        borderColor: dragging ? "var(--low)" : "var(--border)",
        textAlign: "center",
        marginBottom: "1.5rem",
      }}
    >
      <p style={{ margin: "0 0 0.5rem" }}>
        {busy ? "Uploading…" : "Drop a repository archive here"}
      </p>
      <p className="muted" style={{ fontSize: "0.8rem", margin: "0 0 0.75rem" }}>
        .zip or .tar.gz — Python files are extracted and scanned, then discarded.
      </p>

      <label style={{ cursor: "pointer", color: "var(--low)", fontSize: "0.85rem" }}>
        or choose a file
        <input
          type="file"
          accept=".zip,.tar,.tar.gz,.tgz"
          hidden
          disabled={busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void send(file);
          }}
        />
      </label>

      {error && (
        <p style={{ color: "var(--critical)", fontSize: "0.85rem" }}>{error}</p>
      )}
    </div>
  );
}
