import type { Metadata } from "next";
import { performance } from "@/lib/data";
import { PageHeader } from "@/components/page-header";
import { PerformanceBarChart } from "@/components/research-charts";
import { PerformanceTable, type PerformanceMetric } from "@/components/performance-table";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const metadata: Metadata = { title: "Performance" };

export default function PerformancePage() {
  return <div className="mx-auto max-w-[1500px]">
    <PageHeader eyebrow="Portfolio evidence" title="Performance ledger" description="Historical baselines, inspected model evaluation, turnover-aware optimization, and the partial 2026 holdout are kept in separate evidence buckets." status="NET OF COSTS" />
    <Tabs defaultValue="historical_2023_2025">
      <TabsList className="mb-5 h-auto flex-wrap justify-start">{performance.periods.map((period) => <TabsTrigger key={period.id} value={period.id}>{period.label.replace(/, .*/, "")}</TabsTrigger>)}</TabsList>
      {performance.periods.map((period) => <TabsContent key={period.id} value={period.id} className="space-y-6"><Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>{period.label}</CardTitle><CardDescription>Status: {period.status.replace(/_/g, " ")}</CardDescription></CardHeader><CardContent><PerformanceBarChart rows={period.metrics as PerformanceMetric[]} /></CardContent></Card><PerformanceTable rows={period.metrics as PerformanceMetric[]} /></TabsContent>)}
    </Tabs>
    <p className="mt-5 text-xs leading-5 text-muted-foreground">Annualized figures and partial-year results should be interpreted in the context of each labeled evaluation period. The 2026 panel is a genuine but incomplete prospective record.</p>
  </div>;
}
