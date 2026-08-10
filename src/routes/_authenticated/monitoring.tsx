import { createFileRoute } from "@tanstack/react-router";
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
  useOperationMode,
  useRecentEvents,
} from "@/hooks/use-monitoring";
import { useRealtimeEvents } from "@/hooks/use-realtime-events";
import { demoDetections, demoEvents, mergeDemoCameras } from "@/services/monitoring-demo-data";
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
  const opMode = useOperationMode();
  const isDemoMode = (opMode.data ?? "demo") === "demo";
  // Queries are already scoped to the active operation mode; in-memory demo
  // fallbacks are only ever added in demo mode.
  const cameras = useMemo(
    () =>
      isDemoMode
        ? mergeDemoCameras(camerasQuery.data ?? [])
        : (camerasQuery.data ?? []).filter((camera) => !camera.isDemo),
    [camerasQuery.data, isDemoMode],
  );
  const events = useMemo(() => {
    const rows = (eventsQuery.data ?? []).filter(
      (event) => event.sourceMode === (isDemoMode ? "demo" : "live"),
    );
    if (!isDemoMode || rows.length >= 3) return rows;
    return [...demoEvents, ...rows];
  }, [eventsQuery.data, isDemoMode]);
  const [selectedId, setSelectedId] = useState("");
  const [mode, setMode] = useState<"single" | "wall">("single");
  const [overlays, setOverlays] = useState(true);
  const [showCameras, setShowCameras] = useState(false);
  const [showEvents, setShowEvents] = useState(true);
  useEffect(() => {
    if (!selectedId && cameras[0]) setSelectedId(cameras[0].id);
  }, [cameras, selectedId]);
  const selected = cameras.find((camera) => camera.id === selectedId) ?? cameras[0];
  const activeRule = (rules.data ?? []).find((rule) => rule.enabled);
  const selectCamera = (camera: Camera) => {
    setSelectedId(camera.id);
    setMode("single");
    setShowCameras(false);
  };
  if (!selected)
    return (
      <div className="grid min-h-screen place-items-center font-mono text-xs text-muted-foreground">
        NO CAMERAS CONFIGURED
      </div>
    );
  const selectedEvent =
    events.find((event) => event.cameraId === selected.id) ??
    (isDemoMode && selected.isDemo ? demoEvents[0] : undefined);
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
              selectedId={selected.id}
              onSelect={selectCamera}
              rule={activeRule}
            />
          ) : (
            <CameraSidebar cameras={cameras} selectedId={selected.id} onSelect={selectCamera} />
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
                {selected.name} · {selected.location}
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
                <Grid2X2 className="size-3" /> 4 WALL
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
          {mode === "single" ? (
            <MainMonitoringViewport
              camera={selected}
              detections={
                isDemoMode && selected.isDemo && selected.status !== "offline" ? demoDetections : []
              }
              {...(selectedEvent ? { event: selectedEvent } : {})}
              live={!selected.isDemo}
              overlays={overlays}
              onToggleOverlays={() => setOverlays((value) => !value)}
            />
          ) : (
            <CameraWall cameras={cameras} onSelect={selectCamera} />
          )}
          <CameraHealthStrip
            camera={selected}
            {...(selectedEvent ? { event: selectedEvent } : {})}
          />
        </main>
        <div
          className={`${showEvents ? "absolute inset-y-0 right-0 z-40 block w-[350px] shadow-xl" : "hidden"} xl:relative xl:block`}
        >
          <LiveEventPanel events={events} />
        </div>
      </div>
    </div>
  );
}
