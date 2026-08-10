import { createFileRoute } from "@tanstack/react-router";
import { Archive, Pencil, Plus, RotateCcw } from "lucide-react";
import { useState } from "react";
import { CameraFormDialog } from "@/components/cameras/CameraFormDialog";
import { Panel } from "@/components/common/Panel";
import { StatTile } from "@/components/common/StatTile";
import { StatusDot } from "@/components/common/StatusDot";
import { PageContainer } from "@/components/layout/PageContainer";
import { TopBar } from "@/components/layout/TopBar";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  useArchiveCamera,
  useCameraSummary,
  useCameras,
  useCreateCamera,
  useRestoreCamera,
  useToggleCameraFlag,
  useUpdateCamera,
} from "@/hooks/use-monitoring";
import { formatRelative } from "@/lib/format";
import {
  effectiveCameraStatus,
  effectiveRecordingState,
  isCameraStale,
} from "@/lib/health";
import { requireAdministrator } from "@/lib/require-admin";
import type { Camera, CameraConfigInput } from "@/types";

export const Route = createFileRoute("/_authenticated/cameras")({
  beforeLoad: requireAdministrator,
  head: () => ({
    meta: [
      { title: "Cameras — Vigilant Eye AI Smart Surveillance" },
      {
        name: "description",
        content: "Configure IP cameras, AI analysis and recording per channel across the fleet.",
      },
    ],
  }),
  component: CamerasPage,
});

const sourceLabel: Record<Camera["sourceType"], string> = {
  direct_camera: "Direct camera",
  nvr_channel: "NVR channel",
  demo: "Demo",
};

function CamerasPage() {
  const cameras = useCameras("active");
  const archived = useCameras("archived");
  const fleet = useCameraSummary();
  const toggle = useToggleCameraFlag();
  const create = useCreateCamera();
  const update = useUpdateCamera();
  const archive = useArchiveCamera();
  const restore = useRestoreCamera();

  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<Camera | undefined>(undefined);
  const [archiveTarget, setArchiveTarget] = useState<Camera | null>(null);

  const submit = (config: CameraConfigInput) => {
    const done = { onSuccess: () => setFormOpen(false) };
    if (editing) update.mutate({ id: editing.id, config }, done);
    else create.mutate(config, done);
  };

  return (
    <>
      <TopBar title="Camera Management" subtitle="Administrator access" />
      <PageContainer>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          <StatTile label="Configured" value={fleet.data?.total ?? 0} />
          <StatTile label="Online" value={fleet.data?.online ?? 0} tone="success" />
          <StatTile label="Degraded" value={fleet.data?.degraded ?? 0} tone="warning" />
          <StatTile label="Offline" value={fleet.data?.offline ?? 0} tone="critical" />
          <StatTile label="AI enabled" value={fleet.data?.aiEnabled ?? 0} tone="primary" />
        </div>
        <Panel
          title="Camera channels"
          subtitle="RTSP credentials are stored server-side and never sent to the browser."
          bodyClassName="p-0"
          actions={
            <Button
              size="sm"
              className="h-8 gap-1 px-2 text-[11px]"
              onClick={() => {
                setEditing(undefined);
                setFormOpen(true);
              }}
            >
              <Plus className="size-3" /> Add camera
            </Button>
          }
        >
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="border-b border-border/70">
                <th className="label-tech px-3 py-2">Status</th>
                <th className="label-tech px-3 py-2">Name</th>
                <th className="label-tech px-3 py-2">Location</th>
                <th className="label-tech px-3 py-2">Source</th>
                <th className="label-tech px-3 py-2">Host / CH</th>
                <th className="label-tech px-3 py-2">Stream</th>
                <th className="label-tech px-3 py-2">Heartbeat</th>
                <th className="label-tech px-3 py-2">AI</th>
                <th className="label-tech px-3 py-2">Recording</th>
                <th className="label-tech px-3 py-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {(cameras.data ?? []).map((camera) => {
                const status = effectiveCameraStatus(camera);
                const stale = isCameraStale(camera);
                return (
                  <tr key={camera.id} className="hover:bg-surface-2/60">
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        <StatusDot
                          tone={
                            status === "online"
                              ? "online"
                              : status === "degraded"
                                ? "degraded"
                                : "offline"
                          }
                          pulse={status === "online"}
                        />
                        <span className="font-mono text-[11px] uppercase tracking-[0.1em] text-muted-foreground">
                          {stale ? "offline · no heartbeat" : status}
                        </span>
                      </div>
                    </td>
                    <td className="px-3 py-2 text-foreground">
                      {camera.name}
                      {camera.isDemo && (
                        <span className="ml-2 rounded-[3px] border border-border px-1 py-0.5 font-mono text-[10px] uppercase text-muted-foreground">
                          demo
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">{camera.location || "—"}</td>
                    <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                      {sourceLabel[camera.sourceType]}
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                      {camera.host}:{camera.rtspPort} · CH{String(camera.channel).padStart(2, "0")}
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px] tabular-nums text-muted-foreground">
                      {camera.resolution} @ {camera.fps}fps · {camera.streamProfile}
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                      {formatRelative(camera.lastHeartbeatAt)}
                    </td>
                    <td className="px-3 py-2">
                      <Switch
                        checked={camera.aiEnabled}
                        onCheckedChange={(value) =>
                          toggle.mutate({ id: camera.id, field: "aiEnabled", value })
                        }
                      />
                    </td>
                    <td className="px-3 py-2">
                      <span className="font-mono text-[11px] uppercase tracking-[0.1em] text-muted-foreground">
                        {effectiveRecordingState(camera, nvr.data) === "active"
                          ? "recording"
                          : effectiveRecordingState(camera, nvr.data) === "stopped"
                            ? "not recording"
                            : "unknown"}
                      </span>
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex justify-end gap-1.5">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 gap-1 px-2 text-[11px]"
                          onClick={() => {
                            setEditing(camera);
                            setFormOpen(true);
                          }}
                        >
                          <Pencil className="size-3" /> Edit
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 gap-1 px-2 text-[11px] text-warning"
                          onClick={() => setArchiveTarget(camera)}
                        >
                          <Archive className="size-3" /> Archive
                        </Button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {(cameras.data ?? []).length === 0 && (
            <p className="py-10 text-center text-xs text-muted-foreground">
              No cameras configured for this operation mode yet.
            </p>
          )}
        </Panel>

        {(archived.data ?? []).length > 0 && (
          <Panel
            title="Archived cameras"
            subtitle="Archived cameras keep their event history but are excluded from monitoring and AI rules."
            bodyClassName="p-0"
          >
            <table className="w-full text-left text-[13px]">
              <thead>
                <tr className="border-b border-border/70">
                  <th className="label-tech px-3 py-2">Name</th>
                  <th className="label-tech px-3 py-2">Location</th>
                  <th className="label-tech px-3 py-2">Host / CH</th>
                  <th className="label-tech px-3 py-2">Archived</th>
                  <th className="label-tech px-3 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {(archived.data ?? []).map((camera) => (
                  <tr key={camera.id} className="hover:bg-surface-2/60">
                    <td className="px-3 py-2 text-muted-foreground">{camera.name}</td>
                    <td className="px-3 py-2 text-muted-foreground">{camera.location || "—"}</td>
                    <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                      {camera.host}:{camera.rtspPort} · CH{String(camera.channel).padStart(2, "0")}
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                      {formatRelative(camera.updatedAt)}
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex justify-end">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 gap-1 px-2 text-[11px]"
                          onClick={() => restore.mutate(camera.id)}
                        >
                          <RotateCcw className="size-3" /> Restore
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        )}
      </PageContainer>

      <CameraFormDialog
        open={formOpen}
        {...(editing ? { camera: editing } : {})}
        pending={create.isPending || update.isPending}
        onOpenChange={setFormOpen}
        onSubmit={submit}
      />

      <AlertDialog
        open={archiveTarget !== null}
        onOpenChange={(open) => !open && setArchiveTarget(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Archive {archiveTarget?.name}?</AlertDialogTitle>
            <AlertDialogDescription>
              The camera is removed from monitoring and all AI rule assignments, but its recorded
              events stay intact for reporting. You can restore it later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (archiveTarget) archive.mutate(archiveTarget.id);
                setArchiveTarget(null);
              }}
            >
              Archive camera
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
