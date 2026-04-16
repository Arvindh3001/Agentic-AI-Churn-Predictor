"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clearToken } from "@/lib/api";
import { useRouter } from "next/navigation";

const NAV_ITEMS = [
  { href: "/",             label: "Dashboard" },
  { href: "/hitl",         label: "HITL Queue" },
  { href: "/optimization", label: "Optimizer" },
  { href: "/models",       label: "Model Intel" },
];

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();

  function handleLogout() {
    clearToken();
    router.push("/login");
  }

  return (
    <header
      className="fixed top-0 left-0 right-0 z-20 flex items-center"
      style={{ height: "48px", backgroundColor: "#161616" }}
    >
      {/* Brand */}
      <div
        className="flex items-center h-full px-4 shrink-0"
        style={{ borderRight: "1px solid #393939" }}
      >
        {/* IBM-style 8-bar logo mark */}
        <div className="flex flex-col gap-[2px] mr-3 shrink-0">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-[2px] bg-white" style={{ width: i % 2 === 0 ? "20px" : "16px" }} />
          ))}
        </div>
        <div>
          <p className="text-white text-sm font-semibold leading-none tracking-wide">
            Churn Intelligence
          </p>
          <p className="text-[#c6c6c6] text-[11px] leading-none mt-0.5 tracking-[0.16px]">
            Platform v1.0
          </p>
        </div>
      </div>

      {/* Navigation links */}
      <nav className="flex items-center h-full">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className="relative h-full flex items-center px-4 text-sm transition-colors"
              style={{
                color: active ? "#ffffff" : "#c6c6c6",
                backgroundColor: active ? "#262626" : "transparent",
                letterSpacing: "0.16px",
              }}
              onMouseEnter={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.color = "#ffffff";
                  (e.currentTarget as HTMLElement).style.backgroundColor = "#2e2e2e";
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.color = "#c6c6c6";
                  (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
                }
              }}
            >
              {item.label}
              {/* Active indicator — 2px white bottom border */}
              {active && (
                <span
                  className="absolute bottom-0 left-0 right-0"
                  style={{ height: "2px", backgroundColor: "#ffffff" }}
                />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Sign out */}
      <button
        onClick={handleLogout}
        className="h-full px-4 text-sm transition-colors"
        style={{ color: "#c6c6c6", letterSpacing: "0.16px" }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.color = "#ffffff";
          (e.currentTarget as HTMLElement).style.backgroundColor = "#2e2e2e";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.color = "#c6c6c6";
          (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
        }}
      >
        Sign out
      </button>
    </header>
  );
}
