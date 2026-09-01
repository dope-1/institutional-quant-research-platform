"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { label, num, pct } from "@/lib/format";

const baseConfig = {
  historical: { label: "2023–2025", color: "#60a5fa" },
  holdout: { label: "2026 holdout", color: "#fb7185" },
} satisfies ChartConfig;

export function RankIcComparisonChart({ historical, holdout }: { historical: { model: string; mean_rank_ic: number }[]; holdout: { model: string; mean_rank_ic: number }[] }) {
  const data = historical.map((row) => ({ model: label(row.model), historical: row.mean_rank_ic, holdout: holdout.find((item) => item.model === row.model)?.mean_rank_ic ?? 0 }));
  return (
    <ChartContainer config={baseConfig} className="h-[310px] w-full aspect-auto">
      <BarChart data={data} margin={{ left: 4, right: 8, top: 8 }}>
        <CartesianGrid vertical={false} strokeDasharray="3 3" />
        <XAxis dataKey="model" tickLine={false} axisLine={false} fontSize={11} />
        <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => num(value, 2)} width={42} />
        <ReferenceLine y={0} stroke="#64748b" />
        <ChartTooltip content={<ChartTooltipContent formatter={(value) => num(Number(value), 4)} />} />
        <Legend />
        <Bar dataKey="historical" fill="var(--color-historical)" radius={[3, 3, 0, 0]} />
        <Bar dataKey="holdout" fill="var(--color-holdout)" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ChartContainer>
  );
}

const modelColors: Record<string, string> = { lightgbm: "#60a5fa", random_forest: "#22d3ee", ridge: "#f59e0b", xgboost: "#a78bfa" };

export function HoldoutMonthlyChart({ rows }: { rows: { date: string; model: string; rank_ic: number }[] }) {
  const dates = [...new Set(rows.map((row) => row.date))];
  const models = [...new Set(rows.map((row) => row.model))];
  const data = dates.map((date) => Object.fromEntries([["date", date.slice(5, 7)], ...models.map((model) => [model, rows.find((row) => row.date === date && row.model === model)?.rank_ic])]));
  const config = Object.fromEntries(models.map((model) => [model, { label: label(model), color: modelColors[model] }])) as ChartConfig;
  return (
    <ChartContainer config={config} className="h-[330px] w-full aspect-auto">
      <LineChart data={data} margin={{ left: 0, right: 12, top: 8 }}>
        <CartesianGrid vertical={false} strokeDasharray="3 3" />
        <XAxis dataKey="date" tickLine={false} axisLine={false} tickFormatter={(value) => `2026-${value}`} />
        <YAxis domain={[-0.5, 0.5]} tickLine={false} axisLine={false} tickFormatter={(value) => num(value, 1)} width={36} />
        <ReferenceLine y={0} stroke="#94a3b8" />
        <ChartTooltip content={<ChartTooltipContent formatter={(value) => num(Number(value), 4)} />} />
        <Legend formatter={(value) => label(value)} />
        {models.map((model) => <Line key={model} dataKey={model} type="monotone" stroke={`var(--color-${model})`} strokeWidth={2} dot={{ r: 3 }} connectNulls />)}
      </LineChart>
    </ChartContainer>
  );
}

export function PerformanceBarChart({ rows }: { rows: { strategy?: string; model?: string; annualized_return: number; sharpe: number }[] }) {
  const data = rows.map((row) => ({ name: label(row.strategy ?? row.model ?? "unknown"), annualized_return: row.annualized_return, sharpe: row.sharpe }));
  const config = { annualized_return: { label: "Annualized return", color: "#22d3ee" } } satisfies ChartConfig;
  return (
    <ChartContainer config={config} className="h-[340px] w-full aspect-auto">
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 12 }}>
        <CartesianGrid horizontal={false} strokeDasharray="3 3" />
        <XAxis type="number" tickFormatter={(value) => pct(value, 0)} tickLine={false} axisLine={false} />
        <YAxis dataKey="name" type="category" tickLine={false} axisLine={false} width={145} fontSize={10} />
        <ChartTooltip content={<ChartTooltipContent formatter={(value) => pct(Number(value))} />} />
        <Bar dataKey="annualized_return" fill="var(--color-annualized_return)" radius={[0, 3, 3, 0]} />
      </BarChart>
    </ChartContainer>
  );
}

export function RiskScatterChart({ rows }: { rows: { strategy?: string; model?: string; annualized_return: number; annualized_volatility: number; sharpe: number }[] }) {
  const data = rows.map((row) => ({ name: label(row.strategy ?? row.model ?? "unknown"), x: row.annualized_volatility, y: row.annualized_return, z: Math.max(row.sharpe, 0.1) * 60 }));
  const config = { strategy: { label: "Strategy", color: "#60a5fa" } } satisfies ChartConfig;
  return (
    <ChartContainer config={config} className="h-[360px] w-full aspect-auto">
      <ScatterChart margin={{ left: 8, right: 20, top: 14, bottom: 12 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" type="number" name="Volatility" tickFormatter={(value) => pct(value, 0)} tickLine={false} axisLine={false} />
        <YAxis dataKey="y" type="number" name="Return" tickFormatter={(value) => pct(value, 0)} tickLine={false} axisLine={false} width={42} />
        <ZAxis dataKey="z" range={[70, 280]} />
        <ChartTooltip cursor={{ strokeDasharray: "3 3" }} content={<ChartTooltipContent formatter={(value, name) => name === "x" || name === "y" ? pct(Number(value)) : String(value)} />} />
        <Scatter data={data} fill="var(--color-strategy)">{data.map((entry, index) => <Cell key={entry.name} fill={index === 0 ? "#22d3ee" : "#60a5fa"} />)}</Scatter>
      </ScatterChart>
    </ChartContainer>
  );
}

export function FeatureImportanceChart({ rows }: { rows: { feature: string; normalized_mean_abs_shap: number }[] }) {
  const data = [...rows].sort((a, b) => b.normalized_mean_abs_shap - a.normalized_mean_abs_shap).slice(0, 8).reverse().map((row) => ({ feature: label(row.feature), importance: row.normalized_mean_abs_shap }));
  const config = { importance: { label: "Normalized |SHAP|", color: "#a78bfa" } } satisfies ChartConfig;
  return (
    <ChartContainer config={config} className="h-[340px] w-full aspect-auto">
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 12 }}>
        <CartesianGrid horizontal={false} strokeDasharray="3 3" />
        <XAxis type="number" tickFormatter={(value) => pct(value, 0)} tickLine={false} axisLine={false} />
        <YAxis dataKey="feature" type="category" width={170} tickLine={false} axisLine={false} fontSize={10} />
        <ChartTooltip content={<ChartTooltipContent formatter={(value) => pct(Number(value))} />} />
        <Bar dataKey="importance" fill="var(--color-importance)" radius={[0, 3, 3, 0]} />
      </BarChart>
    </ChartContainer>
  );
}

export function ConcentrationChart({ rows }: { rows: { model: string; mean_effective_n: number; mean_positions: number }[] }) {
  const data = rows.map((row) => ({ model: label(row.model), effective: row.mean_effective_n, positions: row.mean_positions }));
  const config = { effective: { label: "Effective N", color: "#22d3ee" }, positions: { label: "Mean positions", color: "#60a5fa" } } satisfies ChartConfig;
  return (
    <ChartContainer config={config} className="h-[310px] w-full aspect-auto">
      <BarChart data={data} margin={{ left: 4, right: 8, top: 8 }}>
        <CartesianGrid vertical={false} strokeDasharray="3 3" />
        <XAxis dataKey="model" tickLine={false} axisLine={false} fontSize={11} />
        <YAxis tickLine={false} axisLine={false} width={30} />
        <ChartTooltip content={<ChartTooltipContent formatter={(value) => num(Number(value), 1)} />} />
        <Legend />
        <Bar dataKey="effective" fill="var(--color-effective)" radius={[3, 3, 0, 0]} />
        <Bar dataKey="positions" fill="var(--color-positions)" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ChartContainer>
  );
}

export function RegimeDispersionChart({ rows }: { rows: { model: string; regime_dimension: string; rank_ic_dispersion: number }[] }) {
  const selected = rows.filter((row) => row.model === "random_forest").map((row) => ({ dimension: label(row.regime_dimension.replace("_regime", "")), dispersion: row.rank_ic_dispersion }));
  const config = { dispersion: { label: "Rank IC dispersion", color: "#f59e0b" } } satisfies ChartConfig;
  return (
    <ChartContainer config={config} className="h-[320px] w-full aspect-auto">
      <BarChart data={selected} layout="vertical" margin={{ left: 8, right: 12 }}>
        <CartesianGrid horizontal={false} strokeDasharray="3 3" />
        <XAxis type="number" tickLine={false} axisLine={false} />
        <YAxis dataKey="dimension" type="category" width={110} tickLine={false} axisLine={false} fontSize={10} />
        <ChartTooltip content={<ChartTooltipContent formatter={(value) => num(Number(value), 4)} />} />
        <Bar dataKey="dispersion" fill="var(--color-dispersion)" radius={[0, 3, 3, 0]} />
      </BarChart>
    </ChartContainer>
  );
}
