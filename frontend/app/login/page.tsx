"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { login, register } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "register") {
        await register(email, password);
      }
      await login(email, password);
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="container" style={{ maxWidth: 380 }}>
      <h1 style={{ fontSize: "1.3rem" }}>
        {mode === "login" ? "Sign in" : "Create an account"}
      </h1>

      <form onSubmit={submit} className="card" style={{ display: "grid", gap: "0.75rem" }}>
        <label style={{ display: "grid", gap: "0.25rem", fontSize: "0.85rem" }}>
          Email
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={inputStyle}
          />
        </label>

        <label style={{ display: "grid", gap: "0.25rem", fontSize: "0.85rem" }}>
          Password
          <input
            type="password"
            required
            minLength={12}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={inputStyle}
          />
          {mode === "register" && (
            <span className="muted" style={{ fontSize: "0.75rem" }}>
              At least 12 characters.
            </span>
          )}
        </label>

        {error && (
          <p style={{ color: "var(--critical)", fontSize: "0.85rem", margin: 0 }}>{error}</p>
        )}

        <button type="submit" disabled={busy} style={buttonStyle}>
          {busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}
        </button>
      </form>

      <button
        onClick={() => {
          setMode(mode === "login" ? "register" : "login");
          setError(null);
        }}
        style={{
          background: "none",
          border: "none",
          color: "var(--low)",
          cursor: "pointer",
          marginTop: "0.75rem",
          fontSize: "0.85rem",
          padding: 0,
        }}
      >
        {mode === "login" ? "Need an account? Register" : "Have an account? Sign in"}
      </button>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  background: "var(--bg)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  color: "var(--text)",
  padding: "0.5rem",
  fontSize: "0.9rem",
};

const buttonStyle: React.CSSProperties = {
  background: "var(--low)",
  border: "none",
  borderRadius: 6,
  color: "#08121c",
  padding: "0.6rem",
  fontWeight: 700,
  cursor: "pointer",
};
