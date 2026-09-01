"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  DatabaseZap,
  FileLock2,
  Gauge,
  LayoutDashboard,
  Microscope,
  Orbit,
  ShieldAlert,
} from "lucide-react";
import { overview } from "@/lib/data";
import { Badge } from "@/components/ui/badge";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";

const nav = [
  ["/", "Overview", LayoutDashboard],
  ["/models", "Models", BrainCircuit],
  ["/performance", "Performance", BarChart3],
  ["/risk", "Risk", ShieldAlert],
  ["/explainability", "Explainability", Microscope],
  ["/regimes", "Regimes", Orbit],
  ["/optimization", "Optimization", Gauge],
  ["/holdout", "2026 Holdout", Activity],
  ["/data-quality", "Data Quality", DatabaseZap],
  ["/methodology", "Methodology", FileLock2],
] as const;

export function DashboardShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { setOpenMobile } = useSidebar();
  return (
    <>
      <Sidebar collapsible="icon">
        <SidebarHeader className="border-b border-sidebar-border p-4">
          <Link href="/" className="flex items-center gap-3" onClick={() => setOpenMobile(false)}>
            <div className="grid size-9 shrink-0 place-items-center rounded-lg border border-cyan-400/30 bg-cyan-400/10 font-mono text-sm font-semibold text-cyan-300">IQ</div>
            <div className="min-w-0 group-data-[collapsible=icon]:hidden">
              <p className="truncate text-sm font-semibold tracking-tight text-white">Quant Research</p>
              <p className="text-[11px] uppercase tracking-[0.16em] text-sidebar-foreground/55">Evidence console</p>
            </div>
          </Link>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Research record</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {nav.map(([href, title, Icon]) => (
                  <SidebarMenuItem key={href}>
                    <SidebarMenuButton asChild isActive={pathname === href} tooltip={title}>
                      <Link href={href} onClick={() => setOpenMobile(false)}>
                        <Icon />
                        <span>{title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter className="border-t border-sidebar-border p-4">
          <div className="space-y-1.5 group-data-[collapsible=icon]:hidden">
            <div className="flex items-center justify-between text-xs"><span className="text-sidebar-foreground/60">Methodology</span><Badge className="border-emerald-400/25 bg-emerald-400/10 text-emerald-300">FROZEN</Badge></div>
            <p className="font-mono text-[11px] text-sidebar-foreground/50">As of {overview.project.latest_market_date}</p>
          </div>
        </SidebarFooter>
        <SidebarRail />
      </Sidebar>
      <SidebarInset className="min-w-0 bg-transparent">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border/80 bg-background/90 px-4 backdrop-blur-xl sm:px-6">
          <SidebarTrigger />
          <div className="h-5 w-px bg-border" />
          <p className="text-xs text-muted-foreground">Static artifact snapshot</p>
          <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground"><span className="hidden sm:inline">Latest market date</span><span className="numeric font-mono text-foreground">{overview.project.latest_market_date}</span></div>
        </header>
        <main className="w-full px-4 py-6 sm:px-6 lg:px-8">{children}</main>
      </SidebarInset>
    </>
  );
}
