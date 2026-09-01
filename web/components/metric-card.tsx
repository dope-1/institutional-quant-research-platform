import type { LucideIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export function MetricCard({ label, value, note, icon: Icon, tone = "blue" }: { label: string; value: string; note: string; icon?: LucideIcon; tone?: "blue" | "green" | "red" | "amber" }) {
  const colors = { blue: "text-blue-300 bg-blue-400/10", green: "text-emerald-300 bg-emerald-400/10", red: "text-rose-300 bg-rose-400/10", amber: "text-amber-300 bg-amber-400/10" };
  return (
    <Card className="border-border/80 bg-card/75 shadow-none">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div><p className="text-xs uppercase tracking-[0.12em] text-muted-foreground">{label}</p><p className="numeric mt-2 text-2xl font-semibold tracking-tight text-white">{value}</p></div>
          {Icon && <span className={`rounded-lg p-2 ${colors[tone]}`}><Icon className="size-4" /></span>}
        </div>
        <p className="mt-3 text-xs leading-5 text-muted-foreground">{note}</p>
      </CardContent>
    </Card>
  );
}
