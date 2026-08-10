import { Link } from "@tanstack/react-router";
import { AlertTriangle, Filter, Image as ImageIcon } from "lucide-react";
import { AssociationBadge, StatusBadge } from "@/components/common/EventBadges";
import { Button } from "@/components/ui/button";
import { useEventSnapshot } from "@/hooks/use-monitoring";
import {
  displayPersonId,
  displaySeverity,
  eventSubtitle,
  eventTitle,
  formatSeconds,
} from "@/lib/event-presentation";
import { formatRelative } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { DetectionEvent } from "@/types";

/**
 * Small evidence thumbnail from a short-lived signed URL. Nothing is persisted:
 * the private snapshots bucket stays private and the URL is never stored.
 */
function EvidenceThumbnail({ event, enabled }: { event: DetectionEvent; enabled: boolean }) {
  const snapshot = useEventSnapshot(event.snapshotPath, enabled);
  const showImage = Boolean(event.snapshotPath) && !snapshot.isError && Boolean(snapshot.data);
  return (
    <div className="relative grid aspect-[4/3] place-items-center overflow-hidden border border-border bg-background hud-grid">
      {showImage ? (
        <img
          src={snapshot.data ?? ""}
          alt={`Snapshot evidence for ${eventTitle(event)} on ${event.cameraName}`}
          className="size-full object-cover"
          loading="lazy"
        />
      ) : (
        <ImageIcon className="size-4 text-muted-foreground" />
      )}
      {event.sourceMode === "demo" && (
        <span className="absolute left-1 top-1 border border-warning/40 bg-background/80 px-0.5 font-mono text-[7px] text-warning">
          DEMO
        </span>
      )}
    </div>
  );
}

function LiveEventCard({
  event,
  enableSnapshot,
}: {
  event: DetectionEvent;
  enableSnapshot: boolean;
}) {
  // Structured fields only — never parsed out of the reviewer note.
  const person = displayPersonId(event);
  const subtitle = eventSubtitle(event);
  const severity = displaySeverity(event);
  return (
    <article
      className={cn(
        "grid grid-cols-[64px_minmax(0,1fr)] gap-2 border-b border-border/70 p-3 transition-colors hover:bg-accent/30",
        severity === "critical" && "border-l-2 border-l-destructive bg-destructive/5",
        severity === "warning" && "border-l-2 border-l-warning",
      )}
    >
      <EvidenceThumbnail event={event} enabled={enableSnapshot} />
      <div className="min-w-0">
        <div className="flex items-start justify-between gap-2">
          <span
            className={cn(
              "font-mono text-[8px] font-bold uppercase",
              severity === "critical"
                ? "text-destructive"
                : severity === "warning"
                  ? "text-warning"
                  : "text-info",
            )}
          >
            {severity}
          </span>
          <span className="font-mono text-[8px] text-muted-foreground">
            {formatRelative(event.detectedAt)}
          </span>
        </div>
        <h3 className="mt-0.5 truncate text-[11px] font-semibold text-foreground">
          {eventTitle(event)}
        </h3>
        <p className="truncate text-[9px] text-muted-foreground">
          {subtitle ?? "AI detection · Review required"}
        </p>
        <p className="mt-1 truncate font-mono text-[8px] text-muted-foreground">
          {event.cameraName}
          {person ? ` · TRACK ${person}` : ""} · {formatSeconds(event.detectionDurationSeconds)}
        </p>
        <div className="mt-1.5 flex items-center justify-between gap-2">
          <span className="font-mono text-[9px] text-primary">
            {Math.round((event.triggerConfidence ?? event.confidence) * 100)}% CONF.
          </span>
          <div className="flex items-center gap-1">
            <AssociationBadge status={event.associationStatus} />
            <StatusBadge status={event.status} />
          </div>
        </div>
      </div>
    </article>
  );
}

export function LiveEventPanel({ events }: { events: DetectionEvent[] }) {
  const pending = events.filter((event) => event.status === "new").length;
  return (
    <aside className="flex min-h-0 w-full flex-col border-l border-border bg-surface lg:w-[350px] lg:shrink-0">
      <header className="grid h-12 grid-cols-[minmax(0,1fr)_auto] items-center border-b border-border px-3">
        <div className="flex min-w-0 items-center gap-2">
          <AlertTriangle className="size-3.5 text-warning" />
          <h2 className="label-tech text-foreground">Live events</h2>
          <span className="font-mono text-[9px] text-primary">{pending} NEW</span>
        </div>
        <div className="flex">
          <Button variant="ghost" size="icon" className="size-7" aria-label="Filter events">
            <Filter className="size-3.5" />
          </Button>
          <Button asChild variant="ghost" size="sm" className="h-7 px-2 font-mono text-[9px]">
            <Link to="/events">VIEW ALL</Link>
          </Button>
        </div>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {events.map((event, index) => (
          // Signed URLs are only requested for the first few visible events.
          <LiveEventCard key={event.id} event={event} enableSnapshot={index < 8} />
        ))}
      </div>
      <div className="border-t border-border p-2 text-center font-mono text-[8px] uppercase text-muted-foreground">
        AI detections are advisory · Human review required
      </div>
    </aside>
  );
}
