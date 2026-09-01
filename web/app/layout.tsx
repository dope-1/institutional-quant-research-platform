import type { Metadata } from "next";
import { DashboardShell } from "@/components/dashboard-shell";
import { SidebarProvider } from "@/components/ui/sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Institutional Quant Research",
    template: "%s | Institutional Quant Research",
  },
  description: "Frozen research evidence, model diagnostics, and prospective holdout monitoring.",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <SidebarProvider>
          <DashboardShell>{children}</DashboardShell>
        </SidebarProvider>
      </body>
    </html>
  );
}
