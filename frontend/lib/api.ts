import type { Finding, Scan, ScanAccepted, ScanSummary } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "securescan_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  window.localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, detail.detail ?? "Request failed");
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export async function login(email: string, password: string): Promise<void> {
  // OAuth2PasswordRequestForm expects form encoding with a `username` field.
  const body = new URLSearchParams({ username: email, password });
  const res = await fetch(`${BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) throw new ApiError(res.status, "Incorrect email or password");
  const { access_token } = await res.json();
  setToken(access_token);
}

export function register(email: string, password: string) {
  return request("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function listScans(): Promise<Scan[]> {
  return request("/api/scans");
}

export function getScan(scanId: string): Promise<Scan> {
  return request(`/api/scans/${scanId}`);
}

export function getSummary(scanId: string): Promise<ScanSummary> {
  return request(`/api/scans/${scanId}/summary`);
}

export function listFindings(scanId: string): Promise<Finding[]> {
  return request(`/api/scans/${scanId}/findings`);
}

export function uploadScan(file: File): Promise<ScanAccepted> {
  const form = new FormData();
  form.append("file", file);
  return request("/api/scans", { method: "POST", body: form });
}

export function dismissFinding(findingId: string, reason: string): Promise<Finding> {
  return request(`/api/findings/${findingId}/dismiss`, {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
}

/**
 * Poll a scan until it leaves QUEUED/RUNNING.
 *
 * The API is async by design (proposal D3), so the client polls rather than
 * holding a request open.
 */
export async function pollScan(
  scanId: string,
  onUpdate: (scan: Scan) => void,
  intervalMs = 2000,
  timeoutMs = 15 * 60 * 1000,
): Promise<Scan> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const scan = await getScan(scanId);
    onUpdate(scan);
    if (scan.status === "COMPLETED" || scan.status === "FAILED") return scan;
    if (Date.now() > deadline) throw new ApiError(408, "Scan timed out");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
