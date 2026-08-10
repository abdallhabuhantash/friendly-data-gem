import { cn } from "@/lib/utils";

type Tone = "online" | "offline" | "degraded" | "idle";

const toneClass: Record<Tone, string> = {
  online: "bg-success shadow-[0_0_8px_currentColor] text-success",
  offline: "bg-destructive text-destructive",
  degraded: "bg-warning text-warning",
  idle: "bg-muted-foreground text-muted-foreground",
};

export function StatusDot({ tone, pulse = false }: { tone: Tone; pulse?: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-block size-2 rounded-full",
        toneClass[tone],
        pulse && "animate-pulse-dot",
      )}
    />
  );
}