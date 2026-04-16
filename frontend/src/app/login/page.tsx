"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { login, setToken, getToken } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (getToken()) router.replace("/");
  }, [router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const auth = await login(username, password);
      setToken(auth.access_token);
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex" style={{ backgroundColor: "#ffffff" }}>
      {/* Left panel — dark brand */}
      <div
        className="hidden lg:flex flex-col justify-between w-1/2 p-16"
        style={{ backgroundColor: "#161616" }}
      >
        {/* Logo mark */}
        <div className="flex flex-col gap-[2px]">
          {[...Array(8)].map((_, i) => (
            <div
              key={i}
              style={{
                height: "3px",
                width: i % 2 === 0 ? "28px" : "22px",
                backgroundColor: "#ffffff",
              }}
            />
          ))}
        </div>

        {/* Brand headline */}
        <div>
          <h1
            style={{
              fontSize: "2.625rem",
              fontWeight: 300,
              color: "#ffffff",
              lineHeight: 1.19,
              marginBottom: "16px",
            }}
          >
            Churn Intelligence
            <br />
            Platform
          </h1>
          <p
            style={{
              fontSize: "1rem",
              color: "#c6c6c6",
              lineHeight: 1.5,
              maxWidth: "480px",
            }}
          >
            Agentic AI customer churn prediction and retention. Six-agent LangGraph
            pipeline — data intelligence, prediction, SHAP explanation, counterfactual
            analysis, retention strategy, and human-in-the-loop review.
          </p>
        </div>

        {/* Footer note */}
        <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}>
          Powered by LangGraph · IBM Carbon Design System
        </p>
      </div>

      {/* Right panel — sign in form */}
      <div className="flex flex-col justify-center flex-1 px-8 lg:px-20">
        <div style={{ maxWidth: "400px", width: "100%" }}>
          {/* Mobile logo */}
          <div className="flex flex-col gap-[2px] mb-8 lg:hidden">
            {[...Array(8)].map((_, i) => (
              <div
                key={i}
                style={{
                  height: "2px",
                  width: i % 2 === 0 ? "20px" : "16px",
                  backgroundColor: "#161616",
                }}
              />
            ))}
          </div>

          <p
            style={{
              fontSize: "0.75rem",
              color: "#525252",
              letterSpacing: "0.32px",
              marginBottom: "8px",
              textTransform: "uppercase",
            }}
          >
            IBM Carbon Design
          </p>
          <h2
            style={{
              fontSize: "1.75rem",
              fontWeight: 400,
              color: "#161616",
              lineHeight: 1.29,
              marginBottom: "32px",
            }}
          >
            Sign in
          </h2>

          <form onSubmit={handleSubmit}>
            {/* Username field */}
            <div className="mb-6">
              <label
                htmlFor="username"
                className="cds-label"
              >
                Username
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="cds-input"
                placeholder="admin"
                autoComplete="username"
                required
                style={{ borderBottomColor: "#161616" }}
              />
            </div>

            {/* Password field */}
            <div className="mb-6">
              <label
                htmlFor="password"
                className="cds-label"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="cds-input"
                placeholder="••••••••"
                autoComplete="current-password"
                required
                style={{ borderBottomColor: "#161616" }}
              />
            </div>

            {error && (
              <div
                className="mb-6"
                style={{
                  backgroundColor: "#fff1f1",
                  borderLeft: "4px solid #da1e28",
                  padding: "12px 16px",
                  fontSize: "0.875rem",
                  color: "#da1e28",
                  letterSpacing: "0.16px",
                }}
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="cds-btn cds-btn--primary w-full"
              style={{ justifyContent: "center", padding: "0 16px" }}
            >
              {loading ? "Signing in…" : "Sign in"}
            </button>
          </form>

          {/* Demo credentials */}
          <div
            className="mt-8 p-4"
            style={{ backgroundColor: "#f4f4f4" }}
          >
            <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }} className="mb-1">
              Demo credentials
            </p>
            <p style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}>
              <span className="font-mono">admin / admin123</span>
              <span style={{ color: "#8d8d8d" }}> · role: admin</span>
            </p>
            <p style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }} className="mt-1">
              <span className="font-mono">analyst / analyst123</span>
              <span style={{ color: "#8d8d8d" }}> · role: analyst</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
