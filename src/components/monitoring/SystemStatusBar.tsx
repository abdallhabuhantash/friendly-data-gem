import { Link, useNavigate } from "@tanstack/react-router";
import { LogOut, Menu, Radio, ScanEye } from "lucide-react";
import { useEffect, useState } from "react";
import { StatusDot } from "@/components/common/StatusDot";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import {
  aiHealthState,
  componentHealthLabel,
  nvrHealthState,
  systemHealthLabel,
  systemHealthState,
} from "@/lib/health";
import type { AiServiceStatus, CameraFleetSummary, EventsSummary, NvrStatus } from "@/types";

function Clock() {
  const [date, setDate] = useState("----.--.--");
  const [time, setTime] = useState("--:--:--");
  useEffect(() => {
    const tick = () => {
      const current = new Date();
      setDate(current.toLocaleDateString("en-CA").replaceAll("-", "."));
      setTime(current.toLocaleTimeString("en-GB"));
    };
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, []);
  return (
    <div className="font-mono text-[10px] tabular-nums text-foreground">
      <span className="hidden xl:inline text-muted-foreground">{date} </span>
      {time}
    </div>
  );
}

function Metric({
  label,
  value,
  online = true,
  tone = "auto",
}: {
  label: string;
  value: string;
  online?: boolean;
  tone?: "auto" | "warning";
}) {
  return (
    <div className="flex h-full items-center gap-2 border-l border-border/70 px-3">
      <StatusDot
        tone={tone === "warning" ? "degraded" : online ? "online" : "offline"}
        pulse={online}
      />
      <div className="leading-tight">
        <p className="font-mono text-[8px] uppercase text-muted-foreground">{label}</p>
        <p className="whitespace-nowrap font-mono text-[10px] uppercase text-foreground">{value}</p>
      </div>
    </div>
  );
}

export function SystemStatusBar({
  fleet,
  events,
  ai,
  nvr,
  onOpenCameras,
}: {
  fleet?: CameraFleetSummary;
  events?: EventsSummary;
  ai?: AiServiceStatus;
  nvr?: NvrStatus;
  onOpenCameras?: () => void;
}) {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  // Every indicator is independent and only claims what has actually been reported.
  const aiState = aiHealthState(ai);
  const nvrState = nvrHealthState(nvr);
  const overall = systemHealthState({ ai, nvr, camerasOnline: fleet?.online ?? 0 });
  const capabilities = systemCapabilities({ ai, nvr, camerasOnline: fleet?.online ?? 0 });
  // Global REC is only claimed when the NVR heartbeat is actually fresh.
  const recordingActive = nvrState === "online" && nvr?.recordingActive === true;

  const handleSignOut = async () => {
    await signOut();
    await navigate({ to: "/login", replace: true });
  };
  return (
    <header className="grid h-[58px] shrink-0 grid-cols-[minmax(0,1fr)_auto] border-b border-border bg-surface">
      <div className="flex min-w-0 items-center">
        <Button
          variant="ghost"
          size="icon"
          className="mx-1 lg:hidden"
          onClick={onOpenCameras}
          aria-label="Open cameras"
        >
          <Menu />
        </Button>
        <Link
          to="/dashboard"
          className="flex h-full min-w-0 items-center gap-2 border-r border-border/70 px-3"
        >
          <span className="grid size-8 shrink-0 place-items-center border border-primary/50 bg-primary/8 text-primary">
            <ScanEye className="size-4" />
          </span>
          <span className="min-w-0 leading-tight">
            <strong className="block truncate font-mono text-[11px] text-foreground">
              AI SMART SURVEILLANCE
            </strong>
            <span className="hidden truncate text-[9px] text-muted-foreground xl:block">
              Intelligent Video Monitoring & Event Detection
            </span>
          </span>
        </Link>
        <div className="hidden h-full min-w-0 lg:flex">
          <Metric
            label="System"
            value={systemHealthLabel[overall]}
            online={overall === "ready"}
            tone={overall === "degraded" ? "warning" : "auto"}
          />
          <Metric
            label="Cameras"
            value={`${fleet?.online ?? 0} / ${fleet?.total ?? 0}`}
            online={(fleet?.online ?? 0) > 0}
          />
          <Metric
            label="AI engine"
            value={componentHealthLabel[aiState]}
            online={aiState === "active" || aiState === "demo"}
            tone={aiState === "stale" ? "warning" : "auto"}
          />
          <Metric
            label="NVR"
            value={
              nvrState === "online"
                ? recordingActive
                  ? "Recording"
                  : "Online"
                : componentHealthLabel[nvrState]
            }
            online={nvrState === "online" || nvrState === "demo"}
            tone={nvrState === "stale" ? "warning" : "auto"}
          />
          <Metric
            label="Analysis"
            value={aiState === "active" ? `${ai?.inferenceFps.toFixed(1) ?? "0.0"} FPS` : "—"}
            online={aiState === "active"}
          />
        </div>
      </div>
      <div className="flex items-center">
        <div className="hidden h-full items-center border-l border-border/70 px-3 md:flex">
          <div>
            <p className="font-mono text-[8px] uppercase text-muted-foreground">Pending review</p>
            <p className="font-mono text-[11px] text-warning">{events?.pendingReview ?? 0}</p>
          </div>
        </div>
        <div className="hidden h-full items-center gap-2 border-l border-border/70 px-3 sm:flex">
          {/* REC is only shown when recording is actually reported. */}
          {recordingActive ? (
            <>
              <Radio className="size-3.5 animate-pulse-dot text-destructive" />
              <span className="font-mono text-[10px] font-bold text-destructive">REC</span>
            </>
          ) : (
            <span className="font-mono text-[10px] uppercase text-muted-foreground">No REC</span>
          )}
          <Clock />
        </div>
        <div className="flex h-full items-center gap-2 border-l border-border/70 px-2">
          <div className="hidden text-right leading-tight xl:block">
            <p className="text-[11px] text-foreground">{user?.fullName ?? "Operator"}</p>
            <p className="font-mono text-[8px] uppercase text-muted-foreground">
              {user?.role ?? "operator"}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="size-8 text-muted-foreground hover:text-destructive"
            onClick={handleSignOut}
            aria-label="Sign out"
          >
            <LogOut className="size-3.5" />
          </Button>
        </div>
      </div>
    </header>
  );
}
