import type { Metadata } from "next";
import { performance } from "@/lib/data";
import { PageHeader } from "@/components/page-header";
import { PerformanceBarChart } from "@/components/research-charts";
import {
  PerformanceTable,
  type PerformanceMetric,
} from "@/components/performance-table";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";

export const metadata: Metadata = { title: "Performance" };

export default function PerformancePage() {
  return (
    <div className="mx-auto max-w-[1500px]">
      <PageHeader
        eyebrow="Portfolio evidence"
        title="Performance ledger"
        description="Historical baselines, inspected model evaluation, turnover-aware optimization, and the partial 2026 holdout are kept in separate evidence buckets."
        status="NET OF COSTS"
      />

      <Tabs defaultValue="historical_2023_2025">
        <TabsList className="mb-5 grid h-auto! w-full grid-cols-1 gap-1 p-1 sm:grid-cols-2 xl:inline-flex xl:w-fit xl:flex-nowrap">
          {performance.periods.map((period) => (
            <TabsTrigger
              key={period.id}
              value={period.id}
              className="h-auto min-h-10 w-full whitespace-normal px-3 py-2 text-center leading-tight xl:whitespace-nowrap"
            >
              {period.label.replace(/, .*/, "")}
            </TabsTrigger>
          ))}
        </TabsList>

        {performance.periods.map((period) => (
          <TabsContent
            key={period.id}
            value={period.id}
            className="space-y-6"
          >
            <Card className="border-border/80 bg-card/70 shadow-none">
              <CardHeader className="px-4 sm:px-6">
                <CardTitle className="text-lg leading-tight">
                  {period.label}
                </CardTitle>
                <CardDescription>
                  Status: {period.status.replace(/_/g, " ")}
                </CardDescription>
              </CardHeader>

              <CardContent className="px-2 sm:px-6">
                <PerformanceBarChart
                  rows={period.metrics as PerformanceMetric[]}
                />
              </CardContent>
            </Card>

            <PerformanceTable
              rows={period.metrics as PerformanceMetric[]}
            />
          </TabsContent>
        ))}
      </Tabs>

      <p className="mt-5 text-xs leading-5 text-muted-foreground">
        Annualized figures and partial-year results should be interpreted in
        the context of each labeled evaluation period. The 2026 panel is a
        genuine but incomplete prospective record.
      </p>
    </div>
  );
}