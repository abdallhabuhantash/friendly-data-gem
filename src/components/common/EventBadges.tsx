import { cn } from "@/lib/utils";
import type { AssociationStatus, EventSeverity, EventStatus } from "@/types";

export { eventTypeLabel } from "@/lib/event-presentation";

export const eventStatusLabel: Record<EventStatus, string> = {
  new: "New",
  under_review: "Under Review",
  confirmed: "Confirmed",
  rejected: "Rejected",
};

const severityClass: Record<EventSeverity, string> = {
  critical: "border-destructive/60 text-destructive bg-destructive/10",
  warning: "border-warning/60 text-warning bg-warning/10",
  info: "border-info/60 text-info bg-info/10",
};

const statusClass: Record<EventStatus, string> = {
  new: "border-primary/60 text-primary bg-primary/10",
  under_review: "border-warning/50 text-warning bg-warning/10",
  confirmed: "border-success/50 text-success bg-success/10",
  rejected: "border-border text-muted-foreground bg-muted/40",
};

const base =
  "inline-flex items-center gap-1 rounded-[3px] border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em]";

export function SeverityBadge({ severity }: { severity: EventSeverity }) {
  return <span className={cn(base, severityClass[severity])}>{severity}</span>;
}

export function StatusBadge({ status }: { status: EventStatus }) {
  return <span className={cn(base, statusClass[status])}>{eventStatusLabel[status]}</span>;
}

export function ConfidenceMeter({ value }: { value: number }) {
  const percent = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="h-1 w-16 overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-primary" style={{ width: `${percent}%` }} />
      </div>
      <span className="font-mono text-xs tabular-nums text-muted-foreground">{percent}%</span>
    </div>
  );
}

const associationClass: Record<AssociationStatus, string> = {
  associated: "border-primary/60 text-primary bg-primary/10",
  uncertain: "border-warning/60 text-warning bg-warning/10",
  unassociated: "border-border text-muted-foreground bg-muted/40",
  not_applicable: "border-border text-muted-foreground bg-muted/20",
};

const associationText: Record<AssociationStatus, string> = {
  associated: "associated",
  uncertain: "uncertain",
  unassociated: "no person",
  not_applicable: "n/a",
};

export function AssociationBadge({ status }: { status: AssociationStatus }) {
  return <span className={cn(base, associationClass[status])}>{associationText[status]}</span>;
}
