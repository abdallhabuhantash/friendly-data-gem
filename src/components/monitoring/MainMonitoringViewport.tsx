import { Camera as CameraIcon, Cpu, Grid2X2, Maximize2, ScanLine, VideoOff } from "lucide-react";
import { useRef } from "react";
import examHallSurveillance from "@/assets/exam-hall-surveillance.jpg";
import { LiveStreamPlayer } from "@/components/common/LiveStreamPlayer";
import { Button } from "@/components/ui/button";
import { displaySeverity } from "@/lib/event-presentation";
import { DetectionOverlayLayer } from "./DetectionOverlayLayer";
import { LiveAlertOverlay } from "./LiveAlertOverlay";
import { cn } from "@/lib/utils";
import type { Camera, DetectionEvent, DetectionOverlay } from "@/types";

export function MainMonitoringViewport({
  camera,
  detections,
  event,
  live,
  overlays,
  onToggleOverlays,
}: {
  camera: Camera;
  detections: DetectionOverlay[];
  event?: DetectionEvent;
  live: boolean;
  overlays: boolean;
  onToggleOverlays: () => void;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const capture = () => {
    const link = document.createElement("a");
    link.href = examHallSurveillance;
    link.download = `${camera.name.replaceAll(" ", "-").toLowerCase()}-snapshot.jpg`;
    link.click();
  };
  const fullscreen = () => {
    void frameRef.current?.requestFullscreen();
  };
  return (
    <div
      ref={frameRef}
      className={cn(
        "relative min-h-0 flex-1 overflow-hidden border border-primary/35 bg-background",
        event &&
          displaySeverity(event) === "critical" &&
          "animate-alert-frame border-destructive/70",
      )}
    >
      <div className="absolute inset-0">
        {camera.status === "offline" ? (
          <div className="hud-grid flex size-full flex-col items-center justify-center gap-2 text-destructive">
            <VideoOff className="size-8" />
            <span className="font-mono text-[10px] uppercase">Camera offline · No signal</span>
          </div>
        ) : (
          <>
            <img
              src={examHallSurveillance}
              alt={`Demonstration surveillance view from ${camera.name}`}
              width={1536}
              height={864}
              className="size-full object-cover saturate-[0.72] contrast-[1.08] brightness-[0.72]"
            />
            <div className="absolute inset-0 bg-background/12" />
            {live && <LiveStreamPlayer cameraId={camera.id} offline={false} />}
          </>
        )}
      </div>
      <div
        className="pointer-events-none absolute inset-0 z-10 opacity-35"
        style={{
          backgroundImage:
            "repeating-linear-gradient(to bottom, transparent 0, transparent 2px, color-mix(in oklab,var(--background) 28%,transparent) 3px)",
        }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 top-0 z-10 h-16 animate-surveillance-scan"
        style={{ background: "var(--scan-line)" }}
      />
      <span className="pointer-events-none absolute left-3 top-3 z-20 size-9 border-l-2 border-t-2 border-primary/70" />
      <span className="pointer-events-none absolute right-3 top-3 z-20 size-9 border-r-2 border-t-2 border-primary/70" />
      <span className="pointer-events-none absolute bottom-3 left-3 z-20 size-9 border-b-2 border-l-2 border-primary/70" />
      <span className="pointer-events-none absolute bottom-3 right-3 z-20 size-9 border-b-2 border-r-2 border-primary/70" />
      <DetectionOverlayLayer
        detections={detections}
        visible={overlays && camera.status !== "offline"}
      />
      <LiveAlertOverlay {...(event ? { event } : {})} camera={camera} />
      <div className="absolute left-5 top-5 z-40 flex items-center gap-2 border border-primary/40 bg-background/82 px-2 py-1.5 backdrop-blur-sm">
        <span className="size-1.5 animate-pulse-dot rounded-full bg-primary" />
        <span className="font-mono text-[9px] text-primary">AI ANALYSIS ACTIVE</span>
        <span className="border-l border-border pl-2 font-mono text-[9px] text-foreground">
          {detections.length} DETECTIONS
        </span>
      </div>
      <div className="absolute right-5 top-5 z-40 flex gap-1">
        <Button
          variant="outline"
          size="icon"
          className="size-8 bg-background/80"
          onClick={onToggleOverlays}
          aria-label="Toggle AI overlays"
        >
          <ScanLine className={cn("size-3.5", overlays && "text-primary")} />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="size-8 bg-background/80"
          onClick={capture}
          aria-label="Save snapshot"
        >
          <CameraIcon className="size-3.5" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="size-8 bg-background/80"
          onClick={fullscreen}
          aria-label="Open full screen"
        >
          <Maximize2 className="size-3.5" />
        </Button>
      </div>
      <div className="absolute bottom-5 left-5 z-40 border border-border bg-background/82 px-3 py-1.5 backdrop-blur-sm">
        <div className="flex items-center gap-2 font-mono text-[9px]">
          <span className="text-primary">CH{String(camera.channel).padStart(2, "0")}</span>
          <span className="text-foreground">{camera.name}</span>
          <span className="text-muted-foreground">{camera.location}</span>
          <span className="text-destructive">● LIVE</span>
        </div>
      </div>
      <div className="absolute bottom-5 right-5 z-40 flex items-center gap-3 border border-primary/40 bg-background/82 px-3 py-1.5 font-mono text-[9px] backdrop-blur-sm">
        <span className="flex items-center gap-1 text-primary">
          <Cpu className="size-3" /> AI
        </span>
        <span>{camera.resolution}</span>
        <span>{camera.fps} FPS</span>
        <span className="text-warning">DEMO</span>
      </div>
      {camera.status !== "offline" && (
        <div className="absolute bottom-16 left-1/2 z-30 -translate-x-1/2 bg-background/65 px-2 py-1 font-mono text-[8px] uppercase text-muted-foreground backdrop-blur-sm">
          Demo preview · Awaiting AI stream connection
        </div>
      )}
    </div>
  );
}

export function CameraHealthStrip({ camera, event }: { camera: Camera; event?: DetectionEvent }) {
  return (
    <div className="grid h-10 shrink-0 grid-cols-3 border border-t-0 border-border bg-surface sm:grid-cols-6">
      <Health
        label="Camera"
        value={camera.status}
        tone={camera.status === "online" ? "ok" : "warn"}
      />
      <Health label="AI rule" value="Mobile Phone" tone="ok" />
      <Health label="People" value="3 tracked" />
      <Health label="Phones" value="2 detected" tone="warn" />
      <Health
        label="Recording"
        value={camera.recording ? "Active" : "Stopped"}
        tone={camera.recording ? "ok" : "warn"}
      />
      <Health
        label="Last alert"
        value={event ? "Just now" : "—"}
        {...(event ? { tone: "critical" as const } : {})}
      />
    </div>
  );
}

function Health({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "ok" | "warn" | "critical";
}) {
  return (
    <div className="min-w-0 border-r border-border/70 px-2 py-1">
      <p className="truncate font-mono text-[7px] uppercase text-muted-foreground">{label}</p>
      <p
        className={cn(
          "truncate font-mono text-[9px] uppercase text-foreground",
          tone === "ok" && "text-success",
          tone === "warn" && "text-warning",
          tone === "critical" && "text-destructive",
        )}
      >
        {value}
      </p>
    </div>
  );
}

export function CameraWall({
  cameras,
  onSelect,
}: {
  cameras: Camera[];
  onSelect: (camera: Camera) => void;
}) {
  return (
    <div className="grid min-h-0 flex-1 grid-cols-2 gap-1 bg-background p-1">
      {cameras.slice(0, 4).map((camera) => (
        <button
          key={camera.id}
          type="button"
          onClick={() => onSelect(camera)}
          className="group relative min-h-0 overflow-hidden border border-border bg-surface text-left"
        >
          <img
            src={examHallSurveillance}
            alt={`Demo preview for ${camera.name}`}
            width={1536}
            height={864}
            className={cn(
              "size-full object-cover brightness-50 transition group-hover:brightness-75",
              camera.status === "offline" && "grayscale",
            )}
          />
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-background/85 px-2 py-1 font-mono text-[9px]">
            <span>
              CH{String(camera.channel).padStart(2, "0")} · {camera.name}
            </span>
            <span className={camera.status === "online" ? "text-success" : "text-destructive"}>
              {camera.status.toUpperCase()}
            </span>
          </div>
          <Grid2X2 className="absolute right-2 top-2 size-3.5 text-primary" />
        </button>
      ))}
    </div>
  );
}
