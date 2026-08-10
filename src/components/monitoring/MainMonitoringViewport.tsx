import { Cpu, Grid2X2, Maximize2, ScanLine, VideoOff } from "lucide-react";
import { useRef } from "react";
import { LiveStreamPlayer } from "@/components/common/LiveStreamPlayer";
import { Button } from "@/components/ui/button";
import { displaySeverity } from "@/lib/event-presentation";
import {
  effectiveCameraStatus,
  effectiveRecordingState,
  isCameraStale,
  recordingStateLabel,
} from "@/lib/health";
import { DetectionOverlayLayer } from "./DetectionOverlayLayer";
import { LiveAlertOverlay } from "./LiveAlertOverlay";
import { cn } from "@/lib/utils";
import type { AiRule, Camera, DetectionEvent, DetectionOverlay, NvrStatus } from "@/types";

/**
 * The viewport renders the annotated MJPEG stream from the Python AI service.
 * When that stream is unavailable it shows a waiting state — it never renders a
 * stand-in image, simulated detections or an invented recording indicator.
 */
export function MainMonitoringViewport({
  camera,
  detections,
  event,
  overlays,
  onToggleOverlays,
}: {
  camera: Camera;
  detections: DetectionOverlay[];
  event?: DetectionEvent;
  overlays: boolean;
  onToggleOverlays: () => void;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const fullscreen = () => {
    void frameRef.current?.requestFullscreen();
  };
  // Effective status: a camera whose heartbeat stopped is shown offline even if
  // the stored status column still says online.
  const status = effectiveCameraStatus(camera);
  const stale = isCameraStale(camera);
  const offline = status === "offline";
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
      <div className="hud-grid absolute inset-0 grid place-items-center">
        {offline ? (
          <div className="flex flex-col items-center gap-2 text-destructive">
            <VideoOff className="size-8" />
            <span className="font-mono text-[10px] uppercase">Camera offline · No signal</span>
          </div>
        ) : (
          <LiveStreamPlayer cameraId={camera.id} offline={false} />
        )}
      </div>
      <div
        className="pointer-events-none absolute inset-x-0 top-0 z-10 h-16 animate-surveillance-scan"
        style={{ background: "var(--scan-line)" }}
      />
      <span className="pointer-events-none absolute left-3 top-3 z-20 size-9 border-l-2 border-t-2 border-primary/70" />
      <span className="pointer-events-none absolute right-3 top-3 z-20 size-9 border-r-2 border-t-2 border-primary/70" />
      <span className="pointer-events-none absolute bottom-3 left-3 z-20 size-9 border-b-2 border-l-2 border-primary/70" />
      <span className="pointer-events-none absolute bottom-3 right-3 z-20 size-9 border-b-2 border-r-2 border-primary/70" />
      <DetectionOverlayLayer detections={detections} visible={overlays && !offline} />
      <LiveAlertOverlay {...(event ? { event } : {})} camera={camera} />
      <div className="absolute left-5 top-5 z-40 flex items-center gap-2 border border-primary/40 bg-background/82 px-2 py-1.5 backdrop-blur-sm">
        <span
          className={cn(
            "size-1.5 rounded-full",
            camera.aiEnabled && !offline ? "animate-pulse-dot bg-primary" : "bg-muted-foreground",
          )}
        />
        <span className="font-mono text-[9px] text-primary">
          {offline
            ? "AI ANALYSIS UNAVAILABLE"
            : camera.aiEnabled
              ? "AI ANALYSIS ENABLED"
              : "AI ANALYSIS OFF"}
        </span>
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
          <span className={offline ? "text-muted-foreground" : "text-destructive"}>
            {offline ? "○ OFFLINE" : status === "degraded" ? "◐ DEGRADED" : "● LIVE"}
          </span>
        </div>
      </div>
      <div className="absolute bottom-5 right-5 z-40 flex items-center gap-3 border border-primary/40 bg-background/82 px-3 py-1.5 font-mono text-[9px] backdrop-blur-sm">
        <span
          className={cn(
            "flex items-center gap-1",
            camera.aiEnabled ? "text-primary" : "text-muted-foreground",
          )}
        >
          <Cpu className="size-3" /> AI
        </span>
        <span>{camera.resolution}</span>
        <span>{stale ? "— FPS" : `${camera.fps} FPS`}</span>
        {camera.isDemo && <span className="text-warning">DEMO SOURCE</span>}
      </div>
    </div>
  );
}

export function CameraHealthStrip({
  camera,
  rule,
  event,
  nvr,
}: {
  camera: Camera;
  rule?: AiRule;
  event?: DetectionEvent;
  nvr?: NvrStatus;
}) {
  // Recording is only claimed when the camera record and the NVR heartbeat
  // agree; unknown heartbeat state is reported as unknown, never as active.
  const recording = effectiveRecordingState(camera, nvr);
  const cameraStatus = effectiveCameraStatus(camera);
  return (
    <div className="grid h-10 shrink-0 grid-cols-3 border border-t-0 border-border bg-surface sm:grid-cols-6">
      <Health
        label="Camera"
        value={cameraStatus}
        tone={cameraStatus === "online" ? "ok" : "warn"}
      />
      <Health
        label="AI rule"
        value={rule?.name ?? "None enabled"}
        {...(rule ? { tone: "ok" as const } : {})}
      />
      <Health label="AI analysis" value={camera.aiEnabled ? "Enabled" : "Off"} />
      <Health
        label="Trigger conf."
        value={
          event?.triggerConfidence != null ? `${Math.round(event.triggerConfidence * 100)}%` : "—"
        }
      />
      <Health
        label="Recording"
        value={recordingStateLabel[recording]}
        tone={recording === "active" ? "ok" : "warn"}
      />
      <Health
        label="Last alert"
        value={event ? "Reported" : "None"}
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
  if (cameras.length === 0)
    return (
      <div className="grid min-h-0 flex-1 place-items-center border border-border bg-surface/40 font-mono text-[10px] uppercase text-muted-foreground">
        No cameras configured
      </div>
    );
  return (
    <div
      className={cn(
        "grid min-h-0 flex-1 gap-1 bg-background p-1",
        cameras.length === 1 ? "grid-cols-1" : cameras.length <= 4 ? "grid-cols-2" : "grid-cols-3",
      )}
    >
      {cameras.map((camera) => (
        <button
          key={camera.id}
          type="button"
          onClick={() => onSelect(camera)}
          className="group relative min-h-0 overflow-hidden border border-border bg-surface text-left"
        >
          <div className="hud-grid absolute inset-0 grid place-items-center">
            <LiveStreamPlayer
              cameraId={camera.id}
              offline={effectiveCameraStatus(camera) === "offline"}
            />
          </div>
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-background/85 px-2 py-1 font-mono text-[9px]">
            <span className="truncate">
              CH{String(camera.channel).padStart(2, "0")} · {camera.name}
            </span>
            <span
              className={
                effectiveCameraStatus(camera) === "online" ? "text-success" : "text-destructive"
              }
            >
              {effectiveCameraStatus(camera).toUpperCase()}
            </span>
          </div>
          <Grid2X2 className="absolute right-2 top-2 size-3.5 text-primary" />
        </button>
      ))}
    </div>
  );
}
