"use client";

import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function ErrorPage({ reset }: { reset: () => void }) {
  return <div className="mx-auto grid min-h-[65vh] max-w-xl place-items-center"><div className="w-full rounded-xl border border-rose-400/30 bg-rose-950/15 p-8 text-center"><AlertTriangle className="mx-auto size-7 text-rose-300" /><h2 className="mt-4 text-xl font-semibold">Research view could not be rendered</h2><p className="mt-2 text-sm leading-6 text-muted-foreground">Confirm that the Milestone 4A JSON export exists in public/data, then retry.</p><Button onClick={reset} className="mt-6">Retry</Button></div></div>;
}
