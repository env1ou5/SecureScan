// Mirrors backend/app/schemas.py. The label set is served by GET /api/taxonomy
// so the frontend never hardcodes a taxonomy that could drift from the model.

export type ScanStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";

export type Severity = "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface ContributingLine {
  line: number;
  score: number;
  text: string;
}

export interface Remediation {
  title: string;
  explanation: string;
  unsafe_example: string;
  safe_example: string;
}

export interface Finding {
  id: string;
  file_path: string;
  function_name: string;
  vulnerability_type: string;
  severity: Severity;
  confidence: number;
  /** False when the checkpoint had no fitted temperature. Surface this to the
   *  user rather than presenting a raw softmax as a probability. */
  confidence_calibrated: boolean;
  start_line: number;
  end_line: number;
  anchor_line: number;
  contributing_lines: ContributingLine[];
  code_snippet: string | null;
  dismissed: boolean;
  remediation: Remediation | null;
}

export interface Scan {
  id: string;
  repository_name: string;
  status: ScanStatus;
  model_version: string;
  files_scanned: number;
  functions_scanned: number;
  findings_count: number;
  duration_seconds: number | null;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface ScanSummary {
  scan: Scan;
  by_severity: { severity: Severity; count: number }[];
  by_type: { vulnerability_type: string; count: number }[];
  by_file: { file_path: string; count: number; highest_severity: Severity }[];
  total_findings: number;
  mean_confidence: number | null;
}

export interface ScanAccepted {
  scan_id: string;
  status: ScanStatus;
  status_url: string;
}
