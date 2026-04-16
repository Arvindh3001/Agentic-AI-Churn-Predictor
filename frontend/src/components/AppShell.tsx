"use client";

import { usePathname } from "next/navigation";
import Navbar from "./Navbar";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isAuthPage = pathname === "/login";

  if (isAuthPage) {
    return <>{children}</>;
  }

  return (
    <>
      <Navbar />
      {/* pt-12 = 48px to clear the fixed masthead */}
      <main className="pt-12 min-h-screen" style={{ backgroundColor: "#ffffff" }}>
        <div className="max-w-[1584px] mx-auto px-8 py-8">
          {children}
        </div>
      </main>
    </>
  );
}
