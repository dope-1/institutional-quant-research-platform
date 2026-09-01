import { Activity, CheckCircle2, Database, LockKeyhole, Users } from "lucide-react";
import { overview, performance, rankIc } from "@/lib/data";
import { integer, label, num } from "@/lib/format";
import { MetricCard } from "@/components/metric-card";
import { PageHeader } from "@/components/page-header";
import { RankIcComparisonChart } from "@/components/research-charts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Home() {
  const holdoutPeriod = performance.periods.find((period) => period.id === "holdout_2026")!;
  const holdoutMetrics = holdoutPeriod.metrics as { strategy: string; sharpe: number; annualized_return: number }[];
  return (
    <div className="mx-auto max-w-[1500px]">
      <PageHeader eyebrow="Milestone 4B · Research evidence" title="Institutional research overview" description="Decision-ready summaries of the frozen methodology, historical evidence, and genuine 2026 prospective holdout. Static artifact JSON is the primary data source." status="FROZEN · 2026-09-01" />

      <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Universe" value={`${overview.project.universe_size} equities`} note="Static prototype universe" icon={Users} />
        <MetricCard label="Research rows" value={integer(overview.data_quality.rows)} note={`${overview.data_quality.feature_count} features · 2012–2025 audit`} icon={Database} />
        <MetricCard label="Data audit" value="PASS" note="No duplicates, label mismatches, or nonfinite features" icon={CheckCircle2} tone="green" />
        <MetricCard label="Holdout evidence" value="4 / 4 negative" note="Mean Rank IC over seven matured 2026 months" icon={Activity} tone="red" />
      </div>

      <Card className="mb-6 overflow-hidden border-rose-400/35 bg-rose-950/20 shadow-none">
        <div className="h-1 bg-gradient-to-r from-rose-500 via-amber-400 to-transparent" />
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center gap-3"><Badge className="border-rose-300/25 bg-rose-400/15 text-rose-200">PROSPECTIVE EVIDENCE</Badge><span className="font-mono text-xs text-rose-200/70">7 matured months · partial year</span></div>
          <CardTitle className="mt-3 text-xl text-white">Every learned model has negative mean Rank IC in 2026</CardTitle>
          <CardDescription className="max-w-4xl text-rose-100/65">This is negative early evidence, not a final statistical verdict. The record is shown without model suppression or retrospective tuning.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {rankIc.prospective_holdout.summary.map((row) => (
            <div key={row.model} className="rounded-lg border border-rose-300/15 bg-black/15 p-4"><p className="text-xs text-rose-100/60">{label(row.model)}</p><p className="numeric mt-1 font-mono text-xl font-semibold text-rose-300">{num(row.mean_rank_ic, 4)}</p><p className="mt-1 text-[11px] text-rose-100/45">Positive months {Math.round(row.positive_rank_ic_fraction * 7)} / 7</p></div>
          ))}
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[1.45fr_0.75fr]">
        <Card className="border-border/80 bg-card/70 shadow-none">
          <CardHeader><CardTitle>Historical vs prospective Rank IC</CardTitle><CardDescription>Mean monthly cross-sectional Rank IC. Historical confidence intervals for every model include zero.</CardDescription></CardHeader>
          <CardContent><RankIcComparisonChart historical={rankIc.historical_evaluation.summary} holdout={rankIc.prospective_holdout.summary} /></CardContent>
        </Card>
        <Card className="border-border/80 bg-card/70 shadow-none">
          <CardHeader><CardTitle>Research record</CardTitle><CardDescription>Periods are labeled by evidentiary status.</CardDescription></CardHeader>
          <CardContent className="space-y-3">
            {overview.evaluation_periods.map((period) => <div key={period.id} className="rounded-lg border border-border/70 bg-background/25 p-4"><div className="flex items-start justify-between gap-4"><p className="text-sm font-medium text-foreground">{period.label}</p><Badge variant="outline" className="shrink-0 font-mono text-[10px]">{period.status}</Badge></div><p className="mt-2 font-mono text-xs text-muted-foreground">{period.start} → {period.end}</p></div>)}
          </CardContent>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Key findings</CardTitle></CardHeader><CardContent><ol className="space-y-3">{overview.key_findings.map((finding, index) => <li key={finding} className="flex gap-3 text-sm leading-6 text-muted-foreground"><span className="mt-0.5 font-mono text-xs text-cyan-300">0{index + 1}</span><span>{finding}</span></li>)}</ol></CardContent></Card>
        <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>2026 portfolio context</CardTitle><CardDescription>Positive realized portfolio returns do not erase negative cross-sectional ranking evidence.</CardDescription></CardHeader><CardContent className="space-y-3">{holdoutMetrics.slice(0, 5).map((row) => <div key={row.strategy} className="flex items-center justify-between gap-4 border-b border-border/60 pb-3 last:border-0"><span className="text-sm text-muted-foreground">{label(row.strategy)}</span><div className="text-right"><p className="numeric font-mono text-sm text-foreground">Sharpe {num(row.sharpe)}</p><p className="numeric text-xs text-muted-foreground">Ann. return {(row.annualized_return * 100).toFixed(1)}%</p></div></div>)}</CardContent></Card>
      </div>

      <div className="mt-6 flex items-start gap-3 rounded-lg border border-amber-400/25 bg-amber-400/5 p-4 text-sm leading-6 text-amber-100/75"><LockKeyhole className="mt-0.5 size-4 shrink-0 text-amber-300" /><p>After viewing 2026 results, changing features, hyperparameters, portfolio rules, or optimizer settings would contaminate the holdout. Future methodology changes require a new prospective period.</p></div>
    </div>
  );
}
