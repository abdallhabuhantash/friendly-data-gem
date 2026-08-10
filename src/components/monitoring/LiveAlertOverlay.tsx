import { AlertTriangle } from "lucide-react";
import {
  displayPersonId,
  eventSubtitle,
  eventTitle,
  formatPercent,
  formatSeconds,
} from "@/lib/event-presentation";
import type { Camera, DetectionEvent } from "@/types";

export function LiveAlertOverlay({ event, camera }: { event?: DetectionEvent; camera: Camera }) {
  // Frontend safety guard: an uncertain association is never presented as a
  // confirmed critical accusation, even if an upstream payload claims
  // severity = critical. It degrades to a warning-level review prompt.
  const uncertain = event?.associationStatus === "uncertain";
  if (!event || (event.severity !== "critical" && !uncertain)) return null;
  // Person IDs are only definitive for reliably associated detections.
  const personId = displayPersonId(event);
  const toneLabel = uncertain
    ? "Warning · human review required"
    : "Critical alert · human review required";
  return (
    <div
      className={`pointer-events-none absolute left-1/2 top-20 z-30 w-[min(92%,470px)] -translate-x-1/2 animate-alert-in border border-l-4 bg-background/92 backdrop-blur-md ${uncertain ? "border-warning/70 shadow-[0_0_24px_color-mix(in_oklab,var(--warning)_32%,transparent)]" : "border-destructive/70 shadow-[0_0_24px_color-mix(in_oklab,var(--destructive)_32%,transparent)]"}`}
    >
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] gap-3 p-3">
        <div
          className={`grid size-9 place-items-center border ${uncertain ? "border-warning/50 bg-warning/12 text-warning" : "border-destructive/50 bg-destructive/12 text-destructive"}`}
        >
          <AlertTriangle className="size-5" />
        </div>
        <div className="min-w-0">
          <p
            className={`font-mono text-[9px] font-bold uppercase ${uncertain ? "text-warning" : "text-destructive"}`}
          >
            {toneLabel}
          </p>
          <h3 className="mt-0.5 text-sm font-bold uppercase text-foreground">
            {eventTitle(event)}
          </h3>
          <p className="text-xs text-muted-foreground">{eventSubtitle(event) ?? "AI detection"}</p>
        </div>
        {event.sourceMode === "demo" && (
          <span className="border border-warning/40 px-1.5 py-0.5 font-mono text-[9px] text-warning">
            DEMO
          </span>
        )}
      </div>
      <dl
        className={`grid grid-cols-3 border-t font-mono text-[9px] ${uncertain ? "border-warning/30 bg-warning/5" : "border-destructive/30 bg-destructive/5"}`}
      >
        <div
          className={`border-r p-2 ${uncertain ? "border-warning/20" : "border-destructive/20"}`}
        >
          <dt className="text-muted-foreground">TRACK / TRIGGER</dt>
          <dd className="mt-0.5 text-foreground">
            {personId ? `ID ${personId}` : "—"} ·{" "}
            {formatPercent(event.triggerConfidence ?? event.confidence)}
          </dd>
        </div>
        <div
          className={`border-r p-2 ${uncertain ? "border-warning/20" : "border-destructive/20"}`}
        >
          <dt className="text-muted-foreground">DURATION / ASSOC.</dt>
          <dd className="mt-0.5 text-foreground">
            {formatSeconds(event.detectionDurationSeconds)} ·{" "}
            {formatPercent(event.associationConfidence)}
          </dd>
        </div>
        <div className="p-2">
          <dt className="text-muted-foreground">CAMERA</dt>
          <dd className="mt-0.5 truncate text-foreground">{camera.name}</dd>
        </div>
      </dl>
    </div>
  );
}
