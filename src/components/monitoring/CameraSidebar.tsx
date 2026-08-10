import { Search, Video, Waves } from "lucide-react";
import { useMemo, useState } from "react";
import { StatusDot } from "@/components/common/StatusDot";
import { cn } from "@/lib/utils";
import type { AiRule, Camera } from "@/types";

function CameraListItem({
  camera,
  selected,
  onSelect,
}: {
  camera: Camera;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "group w-full border-l-2 border-b border-sidebar-border/60 px-3 py-2.5 text-left transition-colors",
        selected
          ? "border-l-primary bg-primary/8"
          : "border-l-transparent hover:bg-sidebar-accent/50",
      )}
    >
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-2">
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
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <span className="shrink-0 font-mono text-[10px] text-primary">
              CH{String(camera.channel).padStart(2, "0")}
            </span>
            <p className="truncate text-xs font-semibold text-foreground">{camera.name}</p>
          </div>
          <p className="mt-0.5 truncate text-[10px] text-muted-foreground">{camera.location}</p>
          <div className="mt-1.5 flex items-center gap-2 font-mono text-[9px] uppercase text-muted-foreground">
            <span className={camera.aiEnabled ? "text-primary" : ""}>
              AI {camera.aiEnabled ? "active" : "off"}
            </span>
            <span className={camera.recording ? "text-destructive" : ""}>
              {camera.recording ? "● REC" : "not recording"}
            </span>
          </div>
        </div>
        {camera.isDemo && (
          <span className="border border-warning/40 bg-warning/8 px-1 py-0.5 font-mono text-[8px] text-warning">
            DEMO
          </span>
        )}
      </div>
    </button>
  );
}

export function CameraSidebar({
  cameras,
  selectedId,
  onSelect,
  rule,
}: {
  cameras: Camera[];
  selectedId: string;
  onSelect: (camera: Camera) => void;
  rule?: AiRule;
}) {
  const [filter, setFilter] = useState("");
  const visible = useMemo(
    () =>
      cameras.filter((camera) =>
        `${camera.name} ${camera.location}`.toLowerCase().includes(filter.toLowerCase()),
      ),
    [cameras, filter],
  );
  return (
    <aside className="flex min-h-0 w-full flex-col border-r border-sidebar-border bg-sidebar lg:w-[250px] lg:shrink-0">
      <div className="border-b border-sidebar-border p-3">
        <div className="flex items-center justify-between">
          <h2 className="label-tech text-foreground">Cameras</h2>
          <span className="font-mono text-[10px] text-primary">{cameras.length} CONFIGURED</span>
        </div>
        <label className="mt-2 grid h-8 grid-cols-[auto_minmax(0,1fr)] items-center gap-2 border border-input bg-background/45 px-2">
          <Search className="size-3.5 text-muted-foreground" />
          <input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Filter cameras"
            className="min-w-0 bg-transparent font-mono text-[11px] text-foreground outline-hidden placeholder:text-muted-foreground"
          />
        </label>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {visible.map((camera) => (
          <CameraListItem
            key={camera.id}
            camera={camera}
            selected={camera.id === selectedId}
            onSelect={() => onSelect(camera)}
          />
        ))}
      </div>
      <div className="border-t border-sidebar-border bg-background/20 p-3">
        <div className="mb-2 flex items-center gap-2">
          <Waves className="size-3.5 text-primary" />
          <h3 className="label-tech text-foreground">Analysis controls</h3>
        </div>
        <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1.5 font-mono text-[10px]">
          <dt className="text-muted-foreground">Analysis mode</dt>
          <dd className="text-primary">CONTINUOUS</dd>
          <dt className="text-muted-foreground">Active rule</dt>
          <dd className="max-w-28 truncate text-foreground">{rule?.name ?? "Mobile Phone"}</dd>
          <dt className="text-muted-foreground">Confidence</dt>
          <dd className="text-foreground">
            {Math.round((rule?.confidenceThreshold ?? 0.7) * 100)}%
          </dd>
          <dt className="text-muted-foreground">Inference rate</dt>
          <dd className="text-foreground">LIVE</dd>
          <dt className="text-muted-foreground">Source</dt>
          <dd className="flex items-center gap-1 text-foreground">
            <Video className="size-3" /> IP / NVR
          </dd>
        </dl>
      </div>
    </aside>
  );
}
