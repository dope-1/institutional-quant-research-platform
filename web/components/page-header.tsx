import { Badge } from "@/components/ui/badge";

export function PageHeader({ eyebrow, title, description, status }: { eyebrow: string; title: string; description: string; status?: string }) {
  return (
    <div className="mb-6 flex flex-col gap-4 border-b border-border/70 pb-6 lg:flex-row lg:items-end lg:justify-between">
      <div className="max-w-3xl">
        <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.2em] text-cyan-300">{eyebrow}</p>
        <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">{title}</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">{description}</p>
      </div>
      {status && <Badge variant="outline" className="w-fit border-cyan-400/30 bg-cyan-400/5 px-3 py-1 font-mono text-cyan-300">{status}</Badge>}
    </div>
  );
}
