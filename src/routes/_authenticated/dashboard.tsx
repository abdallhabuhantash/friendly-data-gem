import { createFileRoute, Link } from "@tanstack/react-router";
import { Activity, AlertTriangle, Camera, Cpu, HardDrive, ShieldAlert, Video } from "lucide-react";
import { CameraTile } from "@/components/common/CameraTile";
import {
  ConfidenceMeter,
  SeverityBadge,
  StatusBadge,
  eventTypeLabel,
} from "@/components/common/EventBadges";
import { Panel } from "@/components/common/Panel";
import { StatTile } from "@/components/common/StatTile";
import { StatusDot } from "@/components/common/StatusDot";
import { PageContainer } from "@/components/layout/PageContainer";
import { TopBar } from "@/components/layout/TopBar";
import {
  useAiServiceStatus,
  useCameraSummary,
  useCameras,
  useEventsSummary,
  useNvrStatus,
  useRecentEvents,
} from "@/hooks/use-monitoring";
import { displaySeverity } from "@/lib/event-presentation";
import { formatRelative } from "@/lib/format";

export const Route = createFileRoute("/_authenticated/dashboard")({
  head: () => ({
    meta: [
      { title: "Dashboard — Vigilant Eye AI Smart Surveillance" },
      {
        name: "description",
        content: "Live camera fleet, AI service health and recent suspicious activity events.",
      },
    ],
  }),
  component: DashboardPage,
});

function DashboardPage() {
  const cameras = useCameras();
  const fleet = useCameraSummary();
  const events = useEventsSummary();
  const recent = useRecentEvents(5);
  const ai = useAiServiceStatus();
  const nvr = useNvrStatus();
  const monitored = (cameras.data ?? [])
    .filter((camera) => camera.status !== "offline")
    .slice(0, 3);
  return (
    <>
      <TopBar title="Operations Dashboard" subtitle="System overview and live detection summary" />
      <PageContainer>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
          <StatTile
            label="Cameras"
            value={fleet.data?.total ?? 0}
            hint="Configured in system"
            icon={Camera}
            loading={fleet.isLoading}
          />
          <StatTile
            label="Online"
            value={fleet.data?.online ?? 0}
            hint={`${fleet.data?.degraded ?? 0} degraded`}
            icon={Activity}
            tone="success"
            loading={fleet.isLoading}
          />
          <StatTile
            label="Offline"
            value={fleet.data?.offline ?? 0}
            hint="No heartbeat"
            icon={AlertTriangle}
            tone={fleet.data?.offline ? "critical" : "default"}
            loading={fleet.isLoading}
          />
          <StatTile
            label="AI enabled"
            value={fleet.data?.aiEnabled ?? 0}
            hint="Analysed by AI service"
            icon={Cpu}
            tone="primary"
            loading={fleet.isLoading}
          />
          <StatTile
            label="Recording"
            value={fleet.data?.recording ?? 0}
            hint="Streams being stored"
            icon={Video}
            loading={fleet.isLoading}
          />
          <StatTile
            label="Pending review"
            value={events.data?.pendingReview ?? 0}
            hint={`${events.data?.critical ?? 0} critical today`}
            icon={ShieldAlert}
            tone="warning"
            loading={events.isLoading}
          />
        </div>
        <div className="grid gap-3 xl:grid-cols-[2fr_1fr]">
          <Panel
            title="Recent AI events"
            subtitle="AI output is advisory. Events require human review."
            actions={
              <Link to="/events" className="label-tech text-primary hover:underline">
                View all
              </Link>
            }
            bodyClassName="p-0"
          >
            <ul className="divide-y divide-border/60">
              {(recent.data ?? []).map((event) => (
                <li key={event.id} className="flex items-center gap-3 px-3 py-2.5">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <SeverityBadge severity={displaySeverity(event)} />
                      <span className="truncate text-[13px] text-foreground">
                        {eventTypeLabel(event.type)}
                      </span>
                    </div>
                    <p className="mt-0.5 truncate text-[11px] text-muted-foreground">
                      {event.cameraName} · {formatRelative(event.detectedAt)} ·{" "}
                      {event.durationSeconds}s
                    </p>
                  </div>
                  <ConfidenceMeter value={event.confidence} />
                  <StatusBadge status={event.status} />
                </li>
              ))}
              {recent.isLoading && (
                <li className="px-3 py-6 text-center text-xs text-muted-foreground">Loading…</li>
              )}
            </ul>
          </Panel>
          <div className="space-y-3">
            <Panel title="AI service" bodyClassName="space-y-2 p-3">
              <div className="flex items-center gap-2">
                <StatusDot
                  tone={ai.data?.online ? "online" : "offline"}
                  pulse={ai.data?.online === true}
                />
                <span className="text-xs text-foreground">
                  {ai.data?.online ? "Connected" : "Disconnected"}
                </span>
              </div>
              <dl className="grid grid-cols-2 gap-y-1.5 font-mono text-[11px]">
                <dt className="text-muted-foreground">Model</dt>
                <dd className="text-right text-foreground/90">{ai.data?.model ?? "—"}</dd>
                <dt className="text-muted-foreground">Device</dt>
                <dd className="text-right text-foreground/90">{ai.data?.device ?? "—"}</dd>
                <dt className="text-muted-foreground">Inference</dt>
                <dd className="text-right text-foreground/90">
                  {ai.data ? `${ai.data.inferenceFps.toFixed(1)} FPS` : "—"}
                </dd>
                <dt className="text-muted-foreground">GPU load</dt>
                <dd className="text-right text-foreground/90">{ai.data?.gpuLoadPercent ?? 0}%</dd>
                <dt className="text-muted-foreground">Queue</dt>
                <dd className="text-right text-foreground/90">{ai.data?.queueDepth ?? 0}</dd>
              </dl>
            </Panel>
            <Panel title="NVR / storage" bodyClassName="space-y-2 p-3">
              <div className="flex items-center gap-2">
                <HardDrive className="size-3.5 text-muted-foreground" />
                <span className="text-xs text-foreground">{nvr.data?.model ?? "—"}</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full bg-primary"
                  style={{ width: `${nvr.data?.storageUsedPercent ?? 0}%` }}
                />
              </div>
              <p className="font-mono text-[11px] text-muted-foreground">
                {nvr.data?.storageUsedPercent ?? 0}% used · retention {nvr.data?.retentionDays ?? 0}{" "}
                days
              </p>
            </Panel>
          </div>
        </div>
        <Panel
          title="Monitored streams"
          subtitle="Dynamic — reflects every configured camera"
          actions={
            <Link to="/monitoring" className="label-tech text-primary hover:underline">
              Open wall
            </Link>
          }
        >
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {monitored.map((camera) => (
              <CameraTile key={camera.id} camera={camera} />
            ))}
          </div>
        </Panel>
      </PageContainer>
    </>
  );
}
