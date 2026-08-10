import { createFileRoute, Link } from "@tanstack/react-router";
import { Grid2X2, Monitor, PanelLeftClose, PanelRightClose } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CameraSidebar } from "@/components/monitoring/CameraSidebar";
import { LiveEventPanel } from "@/components/monitoring/LiveEventPanel";
import {
  CameraHealthStrip,
  CameraWall,
  MainMonitoringViewport,
} from "@/components/monitoring/MainMonitoringViewport";
import { SystemStatusBar } from "@/components/monitoring/SystemStatusBar";
import { Button } from "@/components/ui/button";
import {
  useAiRules,
  useAiServiceStatus,
  useCameraSummary,
  useCameras,
  useEventsSummary,
  useNvrStatus,
  useRecentEvents,
} from "@/hooks/use-monitoring";
import { useRealtimeEvents } from "@/hooks/use-realtime-events";
import type { Camera } from "@/types";

export const Route = createFileRoute("/_authenticated/monitoring")({
  head: () => ({
    meta: [
      { title: "Live Monitoring — AI Smart Surveillance" },
      {
        name: "description",
        content:
          "Real-time multi-camera AI surveillance, detection overlays and live event review.",
      },
      { property: "og:title", content: "Live Monitoring — AI Smart Surveillance" },
      {
        property: "og:description",
        content: "Real-time multi-camera AI surveillance and intelligent event detection.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: MonitoringPage,
});

function MonitoringPage() {
  const camerasQuery = useCameras();
  const fleet = useCameraSummary();
  const eventsSummary = useEventsSummary();
  const eventsQuery = useRecentEvents(20);
  const ai = useAiServiceStatus();
  const nvr = useNvrStatus();
  const rules = useAiRules();
  useRealtimeEvents({ notify: true });
  // Everything below is persisted data only. There is no in-memory fallback:
  // when the database has no cameras or events, the UI says so.
  const cameras = useMemo(() => camerasQuery.data ?? [], [camerasQuery.data]);
  const events = useMemo(() => eventsQuery.data ?? [], [eventsQuery.data]);
  const [selectedId, setSelectedId] = useState("");
  const [mode, setMode] = useState<"single" | "wall">("single");
  const [overlays, setOverlays] = useState(true);
  const [showCameras, setShowCameras] = useState(false);
  const [showEvents, setShowEvents] = useState(true);
  useEffect(() => {
    // Keep the selection valid when cameras are added, archived or removed.
    if (cameras.length === 0) {
      if (selectedId) setSelectedId("");
      return;
    }
    if (!cameras.some((camera) => camera.id === selectedId)) setSelectedId(cameras[0]!.id);
  }, [cameras, selectedId]);
  const selected = cameras.find((camera) => camera.id === selectedId) ?? cameras[0];
  const activeRule = (rules.data ?? []).find((rule) => rule.enabled);
  const selectCamera = (camera: Camera) => {
    setSelectedId(camera.id);
    setMode("single");
    setShowCameras(false);
  };
  const selectedEvent = selected
    ? events.find((event) => event.cameraId === selected.id)
    : undefined;
  return (
    <div className="flex h-screen min-h-[640px] w-full flex-col overflow-hidden bg-background">
      <SystemStatusBar
        {...(fleet.data ? { fleet: fleet.data } : {})}
        {...(eventsSummary.data ? { events: eventsSummary.data } : {})}
        {...(ai.data ? { ai: ai.data } : {})}
        {...(nvr.data ? { nvr: nvr.data } : {})}
        onOpenCameras={() => setShowCameras(true)}
      />
      <div className="relative flex min-h-0 flex-1">
        <div
          className={`${showCameras ? "absolute inset-y-0 left-0 z-50 block w-[270px] shadow-xl" : "hidden"} lg:block`}
        >
          {activeRule ? (
            <CameraSidebar
              cameras={cameras}
              selectedId={selected?.id ?? ""}
              onSelect={selectCamera}
              rule={activeRule}
              loading={camerasQuery.isLoading}
            />
          ) : (
            <CameraSidebar
              cameras={cameras}
              selectedId={selected?.id ?? ""}
              onSelect={selectCamera}
              loading={camerasQuery.isLoading}
            />
          )}
        </div>
        {showCameras && (
          <button
            aria-label="Close cameras"
            className="absolute inset-0 z-40 bg-background/70 lg:hidden"
            onClick={() => setShowCameras(false)}
          />
        )}
        <main className="flex min-w-0 flex-1 flex-col p-2">
          <div className="grid h-9 shrink-0 grid-cols-[minmax(0,1fr)_auto] items-center border border-b-0 border-border bg-surface px-2">
            <div className="flex min-w-0 items-center gap-2">
              <span className="font-mono text-[9px] text-primary">LIVE MONITORING</span>
              <span className="truncate text-[10px] text-muted-foreground">
                {selected ? `${selected.name} · ${selected.location}` : "No camera selected"}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant={mode === "single" ? "secondary" : "ghost"}
                size="sm"
                className="h-7 px-2 font-mono text-[9px]"
                onClick={() => setMode("single")}
              >
                <Monitor className="size-3" /> 1 VIEW
              </Button>
              <Button
                variant={mode === "wall" ? "secondary" : "ghost"}
                size="sm"
                className="h-7 px-2 font-mono text-[9px]"
                onClick={() => setMode("wall")}
              >
                <Grid2X2 className="size-3" /> WALL
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="size-7 xl:hidden"
                onClick={() => setShowEvents((value) => !value)}
                aria-label="Toggle live events"
              >
                {showEvents ? <PanelRightClose /> : <PanelLeftClose />}
              </Button>
            </div>
          </div>
          {!selected ? (
            <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 border border-border bg-surface/40 text-center">
              <p className="font-mono text-[11px] uppercase text-muted-foreground">
                {camerasQuery.isLoading
                  ? "Loading cameras…"
                  : camerasQuery.isError
                    ? "Camera data unavailable"
                    : "No cameras configured"}
              </p>
              {!camerasQuery.isLoading && !camerasQuery.isError && (
                <Button asChild size="sm" variant="outline" className="font-mono text-[9px]">
                  <Link to="/cameras">CONFIGURE A CAMERA</Link>
                </Button>
              )}
            </div>
          ) : mode === "single" ? (
            <MainMonitoringViewport
              camera={selected}
              // Overlays come from the annotated AI stream or real event
              // evidence only; nothing is simulated client-side.
              detections={[]}
              {...(selectedEvent ? { event: selectedEvent } : {})}
              overlays={overlays}
              onToggleOverlays={() => setOverlays((value) => !value)}
            />
          ) : (
            <CameraWall cameras={cameras} onSelect={selectCamera} />
          )}
          {selected && (
            <CameraHealthStrip
              camera={selected}
              {...(activeRule ? { rule: activeRule } : {})}
              {...(selectedEvent ? { event: selectedEvent } : {})}
              {...(nvr.data ? { nvr: nvr.data } : {})}
            />
          )}
        </main>
        <div
          className={`${showEvents ? "absolute inset-y-0 right-0 z-40 block w-[350px] shadow-xl" : "hidden"} xl:relative xl:block`}
        >
          <LiveEventPanel
            events={events}
            loading={eventsQuery.isLoading}
            error={eventsQuery.isError}
          />
        </div>
      </div>
    </div>
  );
}
