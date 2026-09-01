import { pct, num, label } from "@/lib/format";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export type PerformanceMetric = {
  strategy?: string;
  model?: string;
  annualized_return: number;
  annualized_volatility: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  annualized_turnover: number;
  cvar_95_daily: number;
  information_ratio?: number | null;
};

export function PerformanceTable({ rows }: { rows: PerformanceMetric[] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-border/80 bg-card/60">
      <Table>
        <TableHeader><TableRow className="hover:bg-transparent"><TableHead>Strategy</TableHead><TableHead className="text-right">Return</TableHead><TableHead className="text-right">Volatility</TableHead><TableHead className="text-right">Sharpe</TableHead><TableHead className="text-right">Sortino</TableHead><TableHead className="text-right">Max DD</TableHead><TableHead className="text-right">Turnover</TableHead></TableRow></TableHeader>
        <TableBody>
          {rows.map((row) => <TableRow key={row.strategy ?? row.model}><TableCell className="font-medium text-foreground">{label(row.strategy ?? row.model ?? "unknown")}</TableCell><TableCell className="numeric text-right">{pct(row.annualized_return)}</TableCell><TableCell className="numeric text-right">{pct(row.annualized_volatility)}</TableCell><TableCell className="numeric text-right">{num(row.sharpe)}</TableCell><TableCell className="numeric text-right">{num(row.sortino)}</TableCell><TableCell className="numeric text-right text-rose-300">{pct(row.max_drawdown)}</TableCell><TableCell className="numeric text-right">{num(row.annualized_turnover)}×</TableCell></TableRow>)}
        </TableBody>
      </Table>
    </div>
  );
}
