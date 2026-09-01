import { Skeleton } from "@/components/ui/skeleton";

export default function Loading() {
  return <div className="mx-auto max-w-[1500px] space-y-6"><div className="space-y-3"><Skeleton className="h-3 w-40" /><Skeleton className="h-9 w-96 max-w-full" /><Skeleton className="h-4 w-[36rem] max-w-full" /></div><div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{Array.from({ length: 4 }, (_, index) => <Skeleton key={index} className="h-36 rounded-xl" />)}</div><Skeleton className="h-[380px] rounded-xl" /></div>;
}
