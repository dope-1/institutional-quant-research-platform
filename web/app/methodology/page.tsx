import type { Metadata } from "next";
import { methodology } from "@/lib/data";
import { integer, label } from "@/lib/format";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Check, LockKeyhole } from "lucide-react";

export const metadata: Metadata = { title: "Methodology" };

export default function MethodologyPage() {
  return <div className="mx-auto max-w-[1500px]">
    <PageHeader eyebrow="Immutable research design" title="Frozen methodology" description="The research question, target, feature family, chronological safeguards, split boundaries, and publication rules carried forward unchanged." status="DO NOT RETUNE" />
    <Card className="mb-6 border-cyan-400/20 bg-cyan-950/15 shadow-none"><CardHeader><CardTitle>Research question</CardTitle></CardHeader><CardContent><p className="max-w-5xl text-lg leading-8 text-cyan-50">{methodology.research_question}</p></CardContent></Card>
    <div className="mb-6 grid gap-6 lg:grid-cols-2">
      <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Target definition</CardTitle></CardHeader><CardContent className="space-y-3">{Object.entries(methodology.target).map(([key, value]) => <div key={key} className="flex items-start justify-between gap-6 border-b border-border/60 pb-3 last:border-0"><span className="text-sm text-muted-foreground">{label(key)}</span><span className="max-w-[65%] text-right font-mono text-sm text-foreground">{value}</span></div>)}</CardContent></Card>
      <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Universe</CardTitle></CardHeader><CardContent className="grid grid-cols-2 gap-3">{Object.entries(methodology.universe).map(([key, value]) => <div key={key} className="rounded-lg border border-border/70 bg-background/25 p-3"><p className="text-[10px] uppercase tracking-wide text-muted-foreground">{label(key)}</p><p className="mt-1 text-sm font-medium">{typeof value === "boolean" ? (value ? "Yes" : "No") : value}</p></div>)}</CardContent></Card>
    </div>
    <Card className="mb-6 border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Feature family</CardTitle><CardDescription>{methodology.features.length} stock-specific features; model inputs use their cross-sectional ranks.</CardDescription></CardHeader><CardContent className="flex flex-wrap gap-2">{methodology.features.map((feature) => <Badge key={feature} variant="secondary" className="px-3 py-1.5">{label(feature)}</Badge>)}</CardContent></Card>
    <Card className="mb-6 border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Purged chronological splits</CardTitle></CardHeader><CardContent className="p-0"><Table><TableHeader><TableRow><TableHead>Partition</TableHead><TableHead>Start</TableHead><TableHead>End</TableHead><TableHead className="text-right">Rows</TableHead><TableHead className="text-right">Purged rows</TableHead></TableRow></TableHeader><TableBody>{methodology.purged_splits.map((split) => <TableRow key={split.name}><TableCell className="font-medium">{label(split.name)}</TableCell><TableCell className="font-mono">{split.start}</TableCell><TableCell className="font-mono">{split.end}</TableCell><TableCell className="numeric text-right">{integer(split.rows)}</TableCell><TableCell className="numeric text-right">{integer(split.purged_rows_removed)}</TableCell></TableRow>)}</TableBody></Table></CardContent></Card>
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Chronology safeguards</CardTitle></CardHeader><CardContent className="space-y-3">{methodology.chronology_safeguards.map((item) => <div key={item} className="flex items-start gap-3 text-sm leading-6 text-muted-foreground"><span className="mt-1 rounded-full bg-emerald-400/10 p-1 text-emerald-300"><Check className="size-3" /></span><span>{item}</span></div>)}</CardContent></Card>
      <Card className="border-amber-400/25 bg-amber-950/10 shadow-none"><CardHeader><CardTitle className="flex items-center gap-2 text-amber-100"><LockKeyhole className="size-4 text-amber-300" />Frozen research rules</CardTitle></CardHeader><CardContent className="space-y-4">{methodology.frozen_research_rules.map((rule, index) => <div key={rule} className="flex gap-3 text-sm leading-6 text-amber-100/65"><span className="font-mono text-xs text-amber-300">0{index + 1}</span><span>{rule}</span></div>)}</CardContent></Card>
    </div>
  </div>;
}
