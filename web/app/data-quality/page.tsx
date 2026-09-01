import type { Metadata } from "next";
import { overview } from "@/lib/data";
import { integer, pct } from "@/lib/format";
import { PageHeader } from "@/components/page-header";
import { MetricCard } from "@/components/metric-card";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, CheckCircle2, Database, Rows3 } from "lucide-react";

export const metadata: Metadata = { title: "Data Quality" };

export default function DataQualityPage() {
  const quality = overview.data_quality;
  const checks = [
    ["Duplicate date/ticker rows", quality.duplicate_date_ticker_rows],
    ["Label alignment mismatches", quality.label_alignment_mismatches],
    ["Nonfinite feature values", quality.nonfinite_feature_values],
    ["Dates with all macro missing", quality.dates_with_all_missing_macro_observations],
    ["Extreme daily-return flags", quality.extreme_abs_daily_return_rows],
    ["Zero / negative volume flags", quality.zero_or_negative_volume_rows],
  ] as const;
  return <div className="mx-auto max-w-[1500px]">
    <PageHeader eyebrow="Derived audit summary" title="Data quality control" description="Research-panel coverage, chronology checks, and review flags. Raw vendor OHLCV is intentionally not exposed in the web layer." status={quality.audit_pass ? "AUDIT PASS" : "REVIEW"} />
    <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><MetricCard label="Rows" value={integer(quality.rows)} note={`${quality.start_date} → ${quality.end_date}`} icon={Rows3} /><MetricCard label="Tickers" value={String(quality.tickers)} note={`Minimum ${quality.minimum_assets_per_date} assets per date`} icon={Database} /><MetricCard label="Complete features" value={pct(quality.median_complete_feature_fraction, 0)} note="Median complete-feature fraction" icon={CheckCircle2} tone="green" /><MetricCard label="Review flags" value={String(quality.extreme_abs_daily_return_rows + quality.zero_or_negative_volume_rows)} note="Flags are not automatic data errors" icon={AlertTriangle} tone="amber" /></div>
    <div className="grid gap-6 lg:grid-cols-[1fr_0.85fr]">
      <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Audit checks</CardTitle><CardDescription>Zero-count structural failures and separately labeled review flags.</CardDescription></CardHeader><CardContent className="space-y-3">{checks.map(([name, value], index) => { const reviewFlag = index >= 4; return <div key={name} className="flex items-center justify-between rounded-lg border border-border/70 bg-background/25 p-4"><div className="flex items-center gap-3">{reviewFlag ? <AlertTriangle className="size-4 text-amber-300" /> : <CheckCircle2 className="size-4 text-emerald-300" />}<span className="text-sm">{name}</span></div><Badge variant="outline" className={reviewFlag && value > 0 ? "border-amber-400/25 text-amber-300" : "border-emerald-400/25 text-emerald-300"}>{value}</Badge></div>; })}</CardContent></Card>
      <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Audit notes</CardTitle></CardHeader><CardContent><ol className="space-y-4">{quality.notes.map((note, index) => <li key={note} className="flex gap-3 text-sm leading-6 text-muted-foreground"><span className="font-mono text-xs text-cyan-300">0{index + 1}</span><span>{note}</span></li>)}</ol><div className="mt-6 border-t border-border/60 pt-5"><p className="text-xs text-muted-foreground">Extreme-return threshold</p><p className="numeric mt-1 font-mono text-lg">{pct(quality.extreme_abs_daily_return_threshold, 0)}</p></div></CardContent></Card>
    </div>
    <div className="mt-6 rounded-lg border border-cyan-400/20 bg-cyan-400/5 p-4 text-sm leading-6 text-cyan-100/65">Publication boundary: derived summaries only. Raw Yahoo market data remain excluded from the API and web export.</div>
  </div>;
}
