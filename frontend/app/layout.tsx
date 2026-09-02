import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SecureScan",
  description: "AI-powered Python code vulnerability detection",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header
          style={{
            borderBottom: "1px solid var(--border)",
            padding: "1rem 1.5rem",
            display: "flex",
            alignItems: "baseline",
            gap: "0.75rem",
          }}
        >
          <strong style={{ fontSize: "1.05rem" }}>SecureScan</strong>
          <span className="muted" style={{ fontSize: "0.8rem" }}>
            Python vulnerability detection
          </span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
