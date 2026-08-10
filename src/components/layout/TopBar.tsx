import { useNavigate } from "@tanstack/react-router";
import { LogOut } from "lucide-react";
import { useEffect, useState } from "react";
import { StatusDot } from "@/components/common/StatusDot";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { useAiServiceStatus, useEventsSummary, useNvrStatus } from "@/hooks/use-monitoring";

function Clock() {
  const [time, setTime] = useState<string>("--:--:--");
  useEffect(() => {
    const tick = () => setTime(new Date().toLocaleTimeString("en-GB"));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, []);
  return <span className="font-mono text-xs tabular-nums text-foreground/80">{time}</span>;
}

function Indicator({ label, online, detail }: { label: string; online: boolean; detail: string }) {
  return (
    <div className="flex items-center gap-2 border-l border-border/70 px-3">
      <StatusDot tone={online ? "online" : "offline"} pulse={online} />
      <div className="leading-tight">
        <p className="label-tech">{label}</p>
        <p className="font-mono text-[11px] text-foreground/80">{detail}</p>
      </div>
    </div>
  );
}

export function TopBar({ title, subtitle }: { title: string; subtitle?: string }) {
  const navigate = useNavigate();
  const { user, signOut } = useAuth();
  const aiStatus = useAiServiceStatus();
  const nvrStatus = useNvrStatus();
  const events = useEventsSummary();

  const handleSignOut = () => {
    signOut();
    void navigate({ to: "/login", replace: true });
  };

  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-surface px-4">
      <div className="min-w-0">
        <h1 className="truncate text-sm font-semibold tracking-wide text-foreground">{title}</h1>
        {subtitle && <p className="truncate text-[11px] text-muted-foreground">{subtitle}</p>}
      </div>
      <div className="flex items-center">
        <Indicator
          label="AI Service"
          online={aiStatus.data?.online ?? false}
          detail={aiStatus.data ? `${aiStatus.data.inferenceFps.toFixed(1)} FPS` : "…"}
        />
        <Indicator
          label="NVR"
          online={nvrStatus.data?.online ?? false}
          detail={
            nvrStatus.data
              ? `${nvrStatus.data.channelsUsed}/${nvrStatus.data.channelsTotal} CH`
              : "…"
          }
        />
        <div className="flex items-center gap-2 border-l border-border/70 px-3">
          <div className="leading-tight">
            <p className="label-tech">Pending review</p>
            <p className="font-mono text-[11px] text-warning">{events.data?.pendingReview ?? "-"}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 border-l border-border/70 px-3">
          <Clock />
        </div>
        <div className="flex items-center gap-3 border-l border-border/70 pl-3">
          <div className="text-right leading-tight">
            <p className="text-xs text-foreground">{user?.fullName ?? "—"}</p>
            <p className="label-tech">{user?.role ?? ""}</p>
          </div>
          <Button variant="outline" size="sm" onClick={handleSignOut} className="h-8 gap-1.5">
            <LogOut className="size-3.5" /> Sign out
          </Button>
        </div>
      </div>
    </header>
  );
}