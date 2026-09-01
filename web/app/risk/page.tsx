import type { Metadata } from "next";
import { performance } from "@/lib/data";
import { pct } from "@/lib/format";
import { PageHeader } from "@/components/page-header";
import { RiskScatterChart } from "@/components/research-charts";
import { PerformanceTable, type PerformanceMetric } from "@/components/performance-table";
import { MetricCard } from "@/components/metric-card";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, Gauge, ShieldAlert, Waves } from "lucide-react";

export const metadata: Metadata = { title: "Risk" };

export default function RiskPage() {
  const period = performance.periods.find((item) => item.id === "holdout_2026")!;
  const rows = period.metrics as PerformanceMetric[];
  const lowestVol = [...rows].sort((a, b) => a.annualized_volatility - b.annualized_volatility)[0];
  const shallowestDrawdown = [...rows].sort((a, b) => b.max_drawdown - a.max_drawdown)[0];
  const lowestCvar = [...rows].sort((a, b) => a.cvar_95_daily - b.cvar_95_daily)[0];
  return <div className="mx-auto max-w-[1500px]">
    <PageHeader eyebrow="Risk diagnostics" title="2026 risk surface" description="Realized volatility, drawdown, tail loss, turnover, and risk-adjusted return for the current partial-year holdout." status="PARTIAL YEAR" />
    <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><MetricCard label="Lowest volatility" value={pct(lowestVol.annualized_volatility)} note={lowestVol.strategy ?? lowestVol.model ?? ""} icon={Waves} /><MetricCard label="Shallowest drawdown" value={pct(shallowestDrawdown.max_drawdown)} note={shallowestDrawdown.strategy ?? shallowestDrawdown.model ?? ""} icon={ShieldAlert} tone="green" /><MetricCard label="Lowest daily CVaR" value={pct(lowestCvar.cvar_95_daily)} note={lowestCvar.strategy ?? lowestCvar.model ?? ""} icon={Activity} /><MetricCard label="Risk frame" value="Realized" note="No forward-looking risk forecast in this panel" icon={Gauge} tone="amber" /></div>
    <Card className="mb-6 border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Risk / return map</CardTitle><CardDescription>Bubble size represents Sharpe ratio. Positions reflect annualized partial-year metrics.</CardDescription></CardHeader><CardContent><RiskScatterChart rows={rows} /></CardContent></Card>
    <PerformanceTable rows={rows} />
  </div>;
}
