import type { Metadata } from "next";
import { shap } from "@/lib/data";
import { label, num, pct } from "@/lib/format";
import { PageHeader } from "@/components/page-header";
import { FeatureImportanceChart } from "@/components/research-charts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const metadata: Metadata = { title: "Explainability" };

export default function ExplainabilityPage() {
  return <div className="mx-auto max-w-[1500px]">
    <PageHeader eyebrow="Post-hoc diagnostics" title="SHAP explainability" description="Global feature attribution and interaction summaries for frozen nonlinear models. Explanations are descriptive and did not drive model selection." status="POST-HOC ONLY" />
    <div className="mb-6 rounded-lg border border-amber-400/25 bg-amber-400/5 p-4 text-sm leading-6 text-amber-100/70">{shap.policy}</div>
    <Tabs defaultValue="random_forest">
      <TabsList className="mb-5">{Object.keys(shap.model_summaries).map((model) => <TabsTrigger key={model} value={model}>{label(model)}</TabsTrigger>)}</TabsList>
      {Object.entries(shap.model_summaries).map(([model, summary]) => <TabsContent key={model} value={model} className="space-y-6">
        <div className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
          <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>{label(model)} feature attribution</CardTitle><CardDescription>Normalized mean absolute SHAP across the inspected historical sample.</CardDescription></CardHeader><CardContent><FeatureImportanceChart rows={summary.top_shap_features} /></CardContent></Card>
          <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Explanation sample</CardTitle></CardHeader><CardContent><p className="numeric font-mono text-3xl font-semibold text-white">{summary.pretest_training_rows.toLocaleString()}</p><p className="mt-2 text-sm text-muted-foreground">Pre-test training rows</p><div className="mt-6 space-y-3">{summary.top_permutation_features.map((row, index) => <div key={row.feature} className="flex items-center justify-between border-b border-border/60 pb-3"><div className="flex items-center gap-2"><Badge variant="outline" className="font-mono text-[10px]">0{index + 1}</Badge><span className="text-sm">{label(row.feature)}</span></div><span className="numeric font-mono text-xs text-cyan-300">{num(row.importance_mean, 4)}</span></div>)}</div></CardContent></Card>
        </div>
        <Card className="border-border/80 bg-card/70 shadow-none"><CardHeader><CardTitle>Leading SHAP interactions</CardTitle><CardDescription>Pairwise interaction magnitude; not a causal interpretation.</CardDescription></CardHeader><CardContent className="p-0"><Table><TableHeader><TableRow><TableHead>Feature A</TableHead><TableHead>Feature B</TableHead><TableHead className="text-right">Mean |interaction|</TableHead></TableRow></TableHeader><TableBody>{summary.top_shap_interactions.map((row) => <TableRow key={`${row.feature_a}-${row.feature_b}`}><TableCell>{label(row.feature_a)}</TableCell><TableCell>{label(row.feature_b)}</TableCell><TableCell className="numeric text-right font-mono">{num(row.mean_abs_interaction, 6)}</TableCell></TableRow>)}</TableBody></Table></CardContent></Card>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">{summary.top_shap_features.map((row) => <div key={row.feature} className="rounded-lg border border-border/70 bg-card/55 p-3"><p className="text-xs text-muted-foreground">{label(row.feature)}</p><p className="numeric mt-1 font-mono text-sm text-violet-300">{pct(row.normalized_mean_abs_shap)}</p></div>)}</div>
      </TabsContent>)}
    </Tabs>
  </div>;
}
