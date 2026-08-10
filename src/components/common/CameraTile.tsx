import { Cpu, Disc, VideoOff } from "lucide-react";
import { StatusDot } from "./StatusDot";
import { LiveStreamPlayer } from "./LiveStreamPlayer";
import { cn } from "@/lib/utils";
import type { Camera } from "@/types";

export function CameraTile({ camera, live = false }: { camera: Camera; live?: boolean }) {
  const offline = camera.status === "offline";

  return (
    <figure
      className={cn(
        "panel group relative overflow-hidden",
        camera.status === "degraded" && "border-warning/50",
        offline && "border-destructive/40",
      )}
    >
      <div className="hud-grid relative flex aspect-video items-center justify-center bg-[oklch(0.14_0.02_255)]">
        {live ? (
          <LiveStreamPlayer cameraId={camera.id} offline={offline} />
        ) : offline ? (
          <div className="flex flex-col items-center gap-1 text-destructive/80">
            <VideoOff className="size-6" />
            <span className="font-mono text-[10px] uppercase tracking-[0.2em]">No signal</span>
          </div>
        ) : (
          <>
            <span className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
              RTSP stream
            </span>
            <span className="pointer-events-none absolute inset-6 rounded-[2px] border border-primary/25" />
            <span className="pointer-events-none absolute left-1/2 top-1/2 size-8 -translate-x-1/2 -translate-y-1/2 border border-primary/40" />
          </>
        )}
        <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded-[3px] bg-background/70 px-1.5 py-0.5 backdrop-blur-sm">
          <StatusDot
            tone={
              camera.status === "online"
                ? "online"
                : camera.status === "degraded"
                  ? "degraded"
                  : "offline"
            }
            pulse={camera.status === "online"}
          />
          <span className="font-mono text-[10px] uppercase tracking-[0.14em]">
            CH{String(camera.channel).padStart(2, "0")}
          </span>
        </div>
        {camera.recording && (
          <div className="absolute right-2 top-2 flex items-center gap-1 rounded-[3px] bg-background/70 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-destructive backdrop-blur-sm">
            <Disc className="size-3 animate-pulse-dot" /> REC
          </div>
        )}
        {camera.aiEnabled && (
          <div className="absolute bottom-2 left-2 flex items-center gap-1 rounded-[3px] border border-primary/40 bg-background/70 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-primary backdrop-blur-sm">
            <Cpu className="size-3" /> AI
          </div>
        )}
        <div className="absolute bottom-2 right-2 font-mono text-[10px] tabular-nums text-muted-foreground">
          {camera.resolution} · {camera.fps} FPS
        </div>
      </div>
      <figcaption className="flex items-center justify-between gap-2 border-t border-border/70 px-2.5 py-1.5">
        <div className="min-w-0">
          <p className="truncate text-xs font-medium text-foreground">{camera.name}</p>
          <p className="truncate text-[11px] text-muted-foreground">{camera.location}</p>
        </div>
        <span className="shrink-0 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
          {camera.host}
        </span>
      </figcaption>
    </figure>
  );
}