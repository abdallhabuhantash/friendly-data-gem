import { useMemo, useState } from "react";
import {
  AssociationBadge,
  ConfidenceMeter,
  SeverityBadge,
  StatusBadge,
} from "@/components/common/EventBadges";
import { DetectionOverlayLayer } from "@/components/monitoring/DetectionOverlayLayer";
import { ReviewConfirmDialog, type ReviewDecision } from "@/components/events/ReviewConfirmDialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useEventSnapshot } from "@/hooks/use-monitoring";
import {
  associationLabel,
  displayPersonId,
  displaySeverity,
  eventSubtitle,
  eventTitle,
  formatPercent,
  formatSeconds,
} from "@/lib/event-presentation";
import { formatTimestamp } from "@/lib/format";
import { overlaysFromEvidence } from "@/services/detection-overlays";
import type { DetectionEvent, EventStatus } from "@/types";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border/50 py-1.5 last:border-0">
      <span className="label-tech text-muted-foreground">{label}</span>
      <span className="text-right font-mono text-[11px] text-foreground">{children}</span>
    </div>
  );
}

/** Private snapshot rendered from a short-lived signed URL, never a stored URL. */
function SnapshotEvidence({ event, showOverlay }: { event: DetectionEvent; showOverlay: boolean }) {
  const snapshot = useEventSnapshot(event.snapshotPath);
  const overlays = useMemo(
    () => overlaysFromEvidence(event.evidence, displaySeverity(event) === "critical"),
    [event],
  );

  if (!event.snapshotPath) {
    return (
      <div className="flex aspect-video items-center justify-center rounded-[4px] border border-dashed border-border bg-background/60">
        <p className="text-[11px] text-muted-foreground">
          No snapshot was captured for this event.
        </p>
      </div>
    );
  }
  if (snapshot.isPending) {
    return (
      <div className="flex aspect-video items-center justify-center rounded-[4px] border border-border bg-background/60">
        <p className="label-tech text-muted-foreground">Loading evidence…</p>
      </div>
    );
  }
  if (snapshot.isError || !snapshot.data) {
    return (
      <div className="flex aspect-video items-center justify-center rounded-[4px] border border-destructive/50 bg-destructive/5">
        <p className="text-[11px] text-destructive">Snapshot unavailable.</p>
      </div>
    );
  }
  return (
    <div className="relative aspect-video overflow-hidden rounded-[4px] border border-border bg-black">
      <img
        src={snapshot.data}
        alt={`Snapshot evidence for ${eventTitle(event)} on ${event.cameraName}`}
        className="size-full object-cover"
        loading="lazy"
      />
      <DetectionOverlayLayer detections={overlays} visible={showOverlay && overlays.length > 0} />
    </div>
  );
}

export function EventDetailsDialog({
  event,
  pending,
  onOpenChange,
  onReview,
}: {
  event: DetectionEvent | null;
  pending: boolean;
  onOpenChange: (open: boolean) => void;
  onReview: (input: { id: string; status: EventStatus; note?: string | undefined }) => void;
}) {
  const [note, setNote] = useState("");
  const [showOverlay, setShowOverlay] = useState(true);
  const [pendingDecision, setPendingDecision] = useState<ReviewDecision | null>(null);
  if (!event) return null;
  const subtitle = eventSubtitle(event);
  const decide = (status: EventStatus) => {
    onReview({ id: event.id, status, note: note.trim() === "" ? undefined : note.trim() });
    setNote("");
  };
  return (
    <Dialog open={Boolean(event)} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl border-border bg-surface">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 font-mono text-sm uppercase tracking-[0.1em]">
            {eventTitle(event)}
            <SeverityBadge severity={displaySeverity(event)} />
            <StatusBadge status={event.status} />
          </DialogTitle>
          <DialogDescription className="text-[11px]">
            {subtitle ? `${subtitle} · ` : ""}Advisory detection. Requires human review before any
            conclusion.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="label-tech text-muted-foreground">Snapshot evidence</span>
              <label className="flex items-center gap-2">
                <span className="label-tech text-muted-foreground">Detection overlay</span>
                <Switch checked={showOverlay} onCheckedChange={setShowOverlay} />
              </label>
            </div>
            <SnapshotEvidence event={event} showOverlay={showOverlay} />
            <p className="text-[10px] text-muted-foreground">
              Snapshots are stored privately and served through short-lived signed links only.
            </p>
          </div>
          <div className="space-y-3">
            <div className="rounded-[4px] border border-border/70 bg-background/50 px-3 py-1.5">
              <Row label="Event ID">{event.id}</Row>
              <Row label="Detected">{formatTimestamp(event.detectedAt)}</Row>
              <Row label="Camera">{event.cameraName}</Row>
              <Row label="Source">{event.sourceMode}</Row>
              <Row label="Trigger object">{event.triggerObjectClass ?? "—"}</Row>
              <Row label="Trigger confidence">
                <ConfidenceMeter value={event.triggerConfidence ?? event.confidence} />
              </Row>
              <Row label="Association">
                <span className="flex items-center justify-end gap-2">
                  <AssociationBadge status={event.associationStatus} />
                  {associationLabel[event.associationStatus]}
                </span>
              </Row>
              <Row label="Association conf.">{formatPercent(event.associationConfidence)}</Row>
              <Row label="Person track">{displayPersonId(event) ?? "—"}</Row>
              <Row label="Duration">{formatSeconds(event.detectionDurationSeconds)}</Row>
              <Row label="Frames">{event.detectionFrameCount ?? "—"}</Row>
              <Row label="Reviewed by">{event.reviewedBy ?? "—"}</Row>
              <Row label="Reviewed at">
                {event.reviewedAt ? formatTimestamp(event.reviewedAt) : "—"}
              </Row>
            </div>
            {event.note && (
              <p className="rounded-[4px] border border-border/70 bg-background/50 p-2 text-[11px] text-muted-foreground">
                {event.note}
              </p>
            )}
            <div className="space-y-1.5">
              <span className="label-tech text-muted-foreground">Review note (optional)</span>
              <Textarea
                value={note}
                maxLength={1000}
                onChange={(input) => setNote(input.target.value)}
                placeholder="Context for this decision…"
                className="min-h-20 text-xs"
              />
              <div className="flex justify-end gap-1.5">
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-[11px] text-warning"
                  disabled={pending || event.status === "under_review"}
                  onClick={() => decide("under_review")}
                >
                  Mark under review
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-[11px] text-success"
                  disabled={pending || event.status === "confirmed"}
                  onClick={() => setPendingDecision("confirmed")}
                >
                  Confirm
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-[11px] text-destructive"
                  disabled={pending || event.status === "rejected"}
                  onClick={() => setPendingDecision("rejected")}
                >
                  Reject
                </Button>
              </div>
            </div>
          </div>
        </div>
        <ReviewConfirmDialog
          decision={pendingDecision}
          onOpenChange={(open) => !open && setPendingDecision(null)}
          onConfirm={(decision) => {
            decide(decision);
            setPendingDecision(null);
          }}
        />
      </DialogContent>
    </Dialog>
  );
}
