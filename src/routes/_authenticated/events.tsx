import { createFileRoute } from "@tanstack/react-router";
import { Check, X } from "lucide-react";
import { useMemo, useState } from "react";
import {
  ConfidenceMeter,
  SeverityBadge,
  StatusBadge,
  AssociationBadge,
  eventTypeLabel,
} from "@/components/common/EventBadges";
import { Panel } from "@/components/common/Panel";
import { EventDetailsDialog } from "@/components/events/EventDetailsDialog";
import { ReviewConfirmDialog, type ReviewDecision } from "@/components/events/ReviewConfirmDialog";
import { StatTile } from "@/components/common/StatTile";
import { PageContainer } from "@/components/layout/PageContainer";
import { TopBar } from "@/components/layout/TopBar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useEvents, useEventsSummary, useReviewEvent } from "@/hooks/use-monitoring";
import { useRealtimeEvents } from "@/hooks/use-realtime-events";
import { displayPersonId, displaySeverity, formatSeconds } from "@/lib/event-presentation";
import { formatTimestamp } from "@/lib/format";
import type { DetectionEvent, EventSeverity, EventStatus } from "@/types";

export const Route = createFileRoute("/_authenticated/events")({
  head: () => ({
    meta: [
      { title: "Events — Vigilant Eye AI Smart Surveillance" },
      {
        name: "description",
        content:
          "Review suspicious cheating activity events raised by the AI service and confirm or reject them.",
      },
    ],
  }),
  component: EventsPage,
});

function EventsPage() {
  const events = useEvents();
  const summary = useEventsSummary();
  const review = useReviewEvent();
  useRealtimeEvents();
  const [query, setQuery] = useState("");
  const [severity, setSeverity] = useState<EventSeverity | "all">("all");
  const [status, setStatus] = useState<EventStatus | "all">("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Confirm / Reject always pass through an explicit confirmation step.
  const [pendingDecision, setPendingDecision] = useState<{
    id: string;
    decision: ReviewDecision;
  } | null>(null);
  const filtered = useMemo(
    () =>
      (events.data ?? []).filter((event) => {
        const matchesQuery =
          query.trim() === "" ||
          `${event.cameraName} ${eventTypeLabel(event.type)} ${event.id}`
            .toLowerCase()
            .includes(query.toLowerCase());
        const matchesSeverity = severity === "all" || displaySeverity(event) === severity;
        const matchesStatus = status === "all" || event.status === status;
        return matchesQuery && matchesSeverity && matchesStatus;
      }),
    [events.data, query, severity, status],
  );
  const decide = (id: string, next: EventStatus) => {
    // The reviewer identity is recorded server-side from the signed-in session.
    review.mutate({ id, status: next });
  };
  const selected: DetectionEvent | null =
    (events.data ?? []).find((event) => event.id === selectedId) ?? null;
  return (
    <>
      <TopBar title="Event Review" subtitle="AI detections require operator confirmation" />
      <PageContainer>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          <StatTile label="Events Today" value={summary.data?.today ?? 0} />
          <StatTile label="Critical" value={summary.data?.critical ?? 0} tone="critical" />
          <StatTile
            label="Pending review"
            value={summary.data?.pendingReview ?? 0}
            tone="warning"
          />
          <StatTile label="Confirmed" value={summary.data?.confirmed ?? 0} tone="success" />
          <StatTile label="Rejected" value={summary.data?.rejected ?? 0} />
        </div>
        <Panel
          title="Event log"
          subtitle="Terminology is advisory: suspicious / possible activity, never confirmed cheating."
          bodyClassName="p-0"
          actions={
            <div className="flex items-center gap-2">
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search camera or event…"
                className="h-8 w-52 text-xs"
              />
              <Select
                value={severity}
                onValueChange={(value) => setSeverity(value as EventSeverity | "all")}
              >
                <SelectTrigger className="h-8 w-32 text-xs">
                  <SelectValue placeholder="Severity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All severities</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                  <SelectItem value="warning">Warning</SelectItem>
                  <SelectItem value="info">Info</SelectItem>
                </SelectContent>
              </Select>
              <Select
                value={status}
                onValueChange={(value) => setStatus(value as EventStatus | "all")}
              >
                <SelectTrigger className="h-8 w-36 text-xs">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All statuses</SelectItem>
                  <SelectItem value="new">New</SelectItem>
                  <SelectItem value="under_review">Under review</SelectItem>
                  <SelectItem value="confirmed">Confirmed</SelectItem>
                  <SelectItem value="rejected">Rejected</SelectItem>
                </SelectContent>
              </Select>
            </div>
          }
        >
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="border-b border-border/70">
                <th className="label-tech px-3 py-2">ID</th>
                <th className="label-tech px-3 py-2">Detected</th>
                <th className="label-tech px-3 py-2">Event</th>
                <th className="label-tech px-3 py-2">Camera</th>
                <th className="label-tech px-3 py-2">Track</th>
                <th className="label-tech px-3 py-2">Trigger conf.</th>
                <th className="label-tech px-3 py-2">Association</th>
                <th className="label-tech px-3 py-2">Duration</th>
                <th className="label-tech px-3 py-2">Severity</th>
                <th className="label-tech px-3 py-2">Status</th>
                <th className="label-tech px-3 py-2 text-right">Review</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {filtered.map((event) => (
                <tr
                  key={event.id}
                  className="cursor-pointer hover:bg-surface-2/60"
                  onClick={() => setSelectedId(event.id)}
                >
                  <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                    {event.id}
                  </td>
                  <td className="px-3 py-2 font-mono text-[11px] tabular-nums text-muted-foreground">
                    {formatTimestamp(event.detectedAt)}
                  </td>
                  <td className="px-3 py-2 text-foreground">{eventTypeLabel(event.type)}</td>
                  <td className="px-3 py-2 text-muted-foreground">{event.cameraName}</td>
                  <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                    {displayPersonId(event) ?? "—"}
                  </td>
                  <td className="px-3 py-2">
                    <ConfidenceMeter value={event.triggerConfidence ?? event.confidence} />
                  </td>
                  <td className="px-3 py-2">
                    <AssociationBadge status={event.associationStatus} />
                  </td>
                  <td className="px-3 py-2 font-mono text-[11px] tabular-nums">
                    {formatSeconds(event.detectionDurationSeconds)}
                  </td>
                  <td className="px-3 py-2">
                    <SeverityBadge severity={displaySeverity(event)} />
                  </td>
                  <td className="px-3 py-2">
                    <StatusBadge status={event.status} />
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex justify-end gap-1.5" onClick={(e) => e.stopPropagation()}>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 gap-1 px-2 text-[11px] text-success"
                        disabled={event.status === "confirmed" || review.isPending}
                        onClick={() => setPendingDecision({ id: event.id, decision: "confirmed" })}
                      >
                        <Check className="size-3" /> Confirm
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 gap-1 px-2 text-[11px] text-destructive"
                        disabled={event.status === "rejected" || review.isPending}
                        onClick={() => setPendingDecision({ id: event.id, decision: "rejected" })}
                      >
                        <X className="size-3" /> Reject
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-[11px]"
                        onClick={() => setSelectedId(event.id)}
                      >
                        Details
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <p className="py-10 text-center text-xs text-muted-foreground">No events match.</p>
          )}
        </Panel>
      </PageContainer>
      <EventDetailsDialog
        event={selected}
        pending={review.isPending}
        onOpenChange={(open) => !open && setSelectedId(null)}
        onReview={(input) => review.mutate(input)}
      />
      <ReviewConfirmDialog
        decision={pendingDecision?.decision ?? null}
        onOpenChange={(open) => !open && setPendingDecision(null)}
        onConfirm={(decision) => {
          if (pendingDecision) decide(pendingDecision.id, decision);
          setPendingDecision(null);
        }}
      />
    </>
  );
}
