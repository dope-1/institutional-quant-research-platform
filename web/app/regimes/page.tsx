import type { Metadata } from "next";
import { regimes } from "@/lib/data";
import { label, num } from "@/lib/format";
import { PageHeader } from "@/components/page-header";
import { RegimeDispersionChart } from "@/components/research-charts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export const metadata: Metadata = { title: "Regimes" };

export default function RegimesPage() {
  return <div className="mx-auto max-w-[1500px]">
    <PageHeader eyebrow="Conditional diagnostics" title="Market regime analysis" description="Trailing-history-only regime labels show how historical ranking evidence varied across market, volatility, rates, and curve conditions." status="NO EX-POST THRESHOLDS" />
    <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
      <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Random Forest Rank IC dispersion</CardTitle><CardDescription>Difference between the highest and lowest conditional mean Rank IC by regime dimension.</CardDescription></CardHeader><CardContent><RegimeDispersionChart rows={regimes.dispersion} /></CardContent></Card>
      <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Regime coverage</CardTitle><CardDescription>Signal months in each historical condition.</CardDescription></CardHeader><CardContent className="grid grid-cols-2 gap-3">{regimes.counts.map((row) => <div key={`${row.regime_dimension}-${row.regime}`} className="rounded-lg border border-border/70 bg-background/25 p-3"><p className="text-[10px] uppercase tracking-wide text-muted-foreground">{label(row.regime_dimension.replace("_regime", ""))}</p><div className="mt-1 flex items-end justify-between gap-2"><Badge variant="secondary">{label(row.regime)}</Badge><span className="numeric font-mono text-sm">{row.signal_months}m</span></div></div>)}</CardContent></Card>
    </div>
    <Card className="mt-6 border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Conditional dispersion ledger</CardTitle><CardDescription>All models and dimensions, including small differences and negative conditional means.</CardDescription></CardHeader><CardContent className="p-0"><Table><TableHeader><TableRow><TableHead>Model</TableHead><TableHead>Dimension</TableHead><TableHead>Highest regime</TableHead><TableHead className="text-right">High mean IC</TableHead><TableHead>Lowest regime</TableHead><TableHead className="text-right">Low mean IC</TableHead><TableHead className="text-right">Dispersion</TableHead></TableRow></TableHeader><TableBody>{regimes.dispersion.map((row) => <TableRow key={`${row.model}-${row.regime_dimension}`}><TableCell className="font-medium">{label(row.model)}</TableCell><TableCell>{label(row.regime_dimension.replace("_regime", ""))}</TableCell><TableCell>{label(row.highest_regime)}</TableCell><TableCell className="numeric text-right font-mono">{num(row.highest_mean_rank_ic, 4)}</TableCell><TableCell>{label(row.lowest_regime)}</TableCell><TableCell className="numeric text-right font-mono">{num(row.lowest_mean_rank_ic, 4)}</TableCell><TableCell className="numeric text-right font-mono text-amber-300">{num(row.rank_ic_dispersion, 4)}</TableCell></TableRow>)}</TableBody></Table></CardContent></Card>
    <div className="mt-6 rounded-lg border border-border/70 bg-card/50 p-4 text-sm leading-6 text-muted-foreground">{regimes.disclosures.join(" ")}</div>
  </div>;
}
