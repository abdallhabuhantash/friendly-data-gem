import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type Tone = "default" | "primary" | "critical" | "warning" | "success";

const toneValue: Record<Tone, string> = {
  default: "text-foreground",
  primary: "text-primary",
  critical: "text-destructive",
  warning: "text-warning",
  success: "text-success",
};

interface StatTileProps {
  label: string;
  value: string | number;
  hint?: string;
  icon?: LucideIcon;
  tone?: Tone;
  loading?: boolean;
}

export function StatTile({
  label,
  value,
  hint,
  icon: Icon,
  tone = "default",
  loading = false,
}: StatTileProps) {
  return (
    <div className="panel hud-grid relative overflow-hidden px-3 py-2.5">
      <span className="pointer-events-none absolute inset-y-0 left-0 w-px bg-primary/50" />
      <div className="flex items-start justify-between gap-2">
        <p className="label-tech">{label}</p>
        {Icon && <Icon className="size-4 text-muted-foreground" />}
      </div>
      <p
        className={cn(
          "mt-1 font-mono text-2xl leading-none tabular-nums",
          toneValue[tone],
          loading && "opacity-40",
        )}
      >
        {loading ? "--" : value}
      </p>
      {hint && <p className="mt-1 text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}