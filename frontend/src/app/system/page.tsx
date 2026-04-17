"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getToken } from "@/lib/api";

interface HealthData {
  status: string;
  version: string;
  uptime_seconds: number;
  redis: string;
  env: string;
}

interface ServiceStatus {
  name: string;
  url: string;
  status: "healthy" | "degraded" | "unknown";
  latency?: number;
  detail?: string;
}

export default function SystemPage() {
  const router = useRouter();
  const [health, setHealth] = useState<HealthData | null>(null);
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  useEffect(() => {
    if (!getToken()) { router.replace("/login"); return; }
    loadHealth();
    const interval = setInterval(loadHealth, 30_000);
    return () => clearInterval(interval);
  }, [router]);

  async function loadHealth() {
    const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

    // FastAPI health
    let apiHealth: HealthData | null = null;
    const t0 = performance.now();
    try {
      const token = getToken();
      const res = await fetch(`${API}/health`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (res.ok) {
        apiHealth = await res.json();
      }
    } catch { /* network error */ }
    const apiLatency = Math.round(performance.now() - t0);

    setHealth(apiHealth);
    setLastRefresh(new Date());

    // Build service status list
    const svcList: ServiceStatus[] = [
      {
        name: "FastAPI Backend",
        url: `${API}/health`,
        status: apiHealth ? "healthy" : "degraded",
        latency: apiLatency,
        detail: apiHealth ? `v${apiHealth.version} · uptime ${formatUptime(apiHealth.uptime_seconds)}` : "unreachable",
      },
      {
        name: "Redis",
        url: "",
        status: apiHealth?.redis === "connected" ? "healthy" : "degraded",
        detail: apiHealth?.redis ?? "unknown",
      },
      {
        name: "Next.js Frontend",
        url: "/",
        status: "healthy",
        latency: 0,
        detail: "This page loaded",
      },
      {
        name: "Prometheus",
        url: "http://localhost:9090",
        status: "unknown",
        detail: "External service (check :9090)",
      },
      {
        name: "Grafana",
        url: "http://localhost:3001",
        status: "unknown",
        detail: "External service (check :3001)",
      },
      {
        name: "MLflow",
        url: "http://localhost:5001",
        status: "unknown",
        detail: "External service (check :5001)",
      },
    ];

    setServices(svcList);
    setLoading(false);
  }

  function formatUptime(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function statusColor(s: "healthy" | "degraded" | "unknown"): string {
    return s === "healthy" ? "#24a148" : s === "degraded" ? "#da1e28" : "#8d8d8d";
  }
  function statusBg(s: "healthy" | "degraded" | "unknown"): string {
    return s === "healthy" ? "#defbe6" : s === "degraded" ? "#fff1f1" : "#f4f4f4";
  }

  const healthyCount = services.filter((s) => s.status === "healthy").length;
  const totalCount = services.length;

  return (
    <div className="space-y-0">
      {/* ── Header ────────────────────────────────────────── */}
      <div className="mb-8" style={{ paddingBottom: "16px", borderBottom: "1px solid #e0e0e0" }}>
        <h1 style={{ fontSize: "1.75rem", fontWeight: 400, color: "#161616", lineHeight: 1.29 }}>
          System Health
        </h1>
        <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px", marginTop: "4px" }}>
          Live service status · refreshes every 30 seconds
          {lastRefresh && (
            <span style={{ color: "#8d8d8d", marginLeft: "12px" }}>
              Last updated: {lastRefresh.toLocaleTimeString()}
            </span>
          )}
        </p>
      </div>

      {loading ? (
        <div className="flex items-center justify-center" style={{ height: "200px" }}>
          <p style={{ fontSize: "0.875rem", color: "#525252" }}>Checking services…</p>
        </div>
      ) : (
        <>
          {/* ── Summary tiles ──────────────────────────────── */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px mb-px" style={{ backgroundColor: "#c6c6c6" }}>
            {[
              {
                label: "OVERALL STATUS",
                value: healthyCount === totalCount ? "HEALTHY" : "DEGRADED",
                color: healthyCount === totalCount ? "#24a148" : "#da1e28",
              },
              { label: "SERVICES UP",    value: `${healthyCount} / ${totalCount}`, color: "#161616" },
              { label: "API VERSION",    value: health?.version ?? "—",            color: "#161616" },
              { label: "ENVIRONMENT",   value: health?.env ?? "—",                 color: "#0f62fe" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ backgroundColor: "#f4f4f4", padding: "24px" }}>
                <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", textTransform: "uppercase", marginBottom: "8px" }}>
                  {label}
                </p>
                <p style={{ fontSize: "1.75rem", fontWeight: 300, color, lineHeight: 1.25 }}>{value}</p>
              </div>
            ))}
          </div>

          {/* ── Service status table ────────────────────────── */}
          <div style={{ backgroundColor: "#ffffff", marginTop: "8px" }}>
            <div className="px-6 py-4" style={{ borderBottom: "1px solid #e0e0e0" }}>
              <p style={{ fontSize: "1rem", fontWeight: 600, color: "#161616" }}>Service Status</p>
            </div>
            {services.map((svc, i) => (
              <div
                key={svc.name}
                className="flex items-center justify-between px-6 py-4"
                style={{
                  backgroundColor: i % 2 === 0 ? "#ffffff" : "#f4f4f4",
                  borderBottom: "1px solid #e0e0e0",
                }}
              >
                <div className="flex items-center gap-4">
                  {/* Status dot */}
                  <div
                    style={{
                      width: "10px",
                      height: "10px",
                      borderRadius: "50%",
                      backgroundColor: statusColor(svc.status),
                      flexShrink: 0,
                    }}
                  />
                  <div>
                    <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616" }}>{svc.name}</p>
                    {svc.detail && (
                      <p style={{ fontSize: "0.75rem", color: "#525252", marginTop: "2px" }}>{svc.detail}</p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  {svc.latency != null && svc.latency > 0 && (
                    <span style={{ fontSize: "0.75rem", color: "#8d8d8d" }}>{svc.latency}ms</span>
                  )}
                  <span
                    style={{
                      padding: "2px 10px",
                      borderRadius: "24px",
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      letterSpacing: "0.32px",
                      backgroundColor: statusBg(svc.status),
                      color: statusColor(svc.status),
                    }}
                  >
                    {svc.status.toUpperCase()}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* ── Quick links ─────────────────────────────────── */}
          <div
            className="grid grid-cols-1 md:grid-cols-3 gap-px mt-px"
            style={{ backgroundColor: "#c6c6c6" }}
          >
            {[
              { label: "API Documentation", href: "http://localhost:8000/docs", desc: "Swagger UI — all endpoints" },
              { label: "Grafana Dashboards", href: "http://localhost:3001", desc: "Metrics · admin / admin123" },
              { label: "MLflow Experiments", href: "http://localhost:5001", desc: "Model registry + runs" },
              { label: "Prometheus Metrics", href: "http://localhost:9090", desc: "Raw metrics scrape endpoint" },
              { label: "Fairness Report",    href: "http://localhost:8000/api/v1/analytics/fairness/report", desc: "HTML bias report" },
              { label: "Health Check",       href: "http://localhost:8000/health", desc: "JSON health status" },
            ].map(({ label, href, desc }) => (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: "block",
                  backgroundColor: "#ffffff",
                  padding: "20px 24px",
                  textDecoration: "none",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.backgroundColor = "#edf5ff"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.backgroundColor = "#ffffff"; }}
              >
                <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#0f62fe" }}>{label}</p>
                <p style={{ fontSize: "0.75rem", color: "#525252", marginTop: "4px" }}>{desc}</p>
              </a>
            ))}
          </div>

          {/* ── Deployment info ─────────────────────────────── */}
          <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginTop: "1px" }}>
            <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", marginBottom: "12px" }}>
              Deployment Commands
            </p>
            <div className="space-y-2">
              {[
                { label: "Start all services",  cmd: "docker compose -f docker/docker-compose.yml up -d" },
                { label: "View logs (API)",      cmd: "docker compose -f docker/docker-compose.yml logs -f api" },
                { label: "Apply K8s manifests",  cmd: "kubectl apply -f k8s/ -n churn-app" },
                { label: "Run CI pipeline",      cmd: "gh workflow run ci.yml" },
                { label: "Trigger retraining",   cmd: "gh workflow run retrain.yml --field force_retrain=true" },
              ].map(({ label, cmd }) => (
                <div key={label} className="flex items-center gap-4">
                  <span style={{ fontSize: "0.75rem", color: "#525252", minWidth: "180px" }}>{label}</span>
                  <code
                    className="font-mono"
                    style={{
                      fontSize: "0.75rem",
                      color: "#161616",
                      backgroundColor: "#ffffff",
                      padding: "4px 12px",
                      border: "1px solid #e0e0e0",
                    }}
                  >
                    {cmd}
                  </code>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
