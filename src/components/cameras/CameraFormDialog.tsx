import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  CREDENTIALS_MESSAGE,
  rtspPreview,
  validateCameraConfig,
  type CameraFormValues,
} from "@/lib/camera-validation";
import type { Camera, CameraConfigInput } from "@/types";

const emptyValues: CameraFormValues = {
  name: "",
  location: "",
  sourceType: "direct_camera",
  host: "",
  rtspPort: 554,
  channel: 1,
  streamPath: "",
  streamProfile: "main",
  resolution: "1920x1080",
  fps: 20,
  aiEnabled: false,
};

const fromCamera = (camera: Camera): CameraFormValues => ({
  name: camera.name,
  location: camera.location,
  sourceType: camera.sourceType,
  host: camera.host,
  rtspPort: camera.rtspPort,
  channel: camera.channel,
  streamPath: camera.streamPath,
  streamProfile: camera.streamProfile,
  resolution: camera.resolution,
  fps: camera.fps,
  aiEnabled: camera.aiEnabled,
});

function Field({
  label,
  error,
  hint,
  children,
}: {
  label: string;
  error?: string | undefined;
  hint?: string | undefined;
  children: React.ReactNode;
}) {
  return (
    <label className="block space-y-1">
      <span className="label-tech text-muted-foreground">{label}</span>
      {children}
      {hint && !error && <span className="block text-[10px] text-muted-foreground">{hint}</span>}
      {error && <span className="block text-[10px] text-destructive">{error}</span>}
    </label>
  );
}

export function CameraFormDialog({
  open,
  camera,
  pending,
  onOpenChange,
  onSubmit,
}: {
  open: boolean;
  camera?: Camera;
  pending: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (config: CameraConfigInput) => void;
}) {
  const [values, setValues] = useState<CameraFormValues>(emptyValues);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!open) return;
    setValues(camera ? fromCamera(camera) : emptyValues);
    setErrors({});
  }, [open, camera]);

  const set = <K extends keyof CameraFormValues>(key: K, value: CameraFormValues[K]) =>
    setValues((current) => ({ ...current, [key]: value }));

  const preview = useMemo(
    () =>
      rtspPreview({
        host: String(values.host),
        rtspPort: values.rtspPort as number,
        streamPath: String(values.streamPath ?? ""),
      }),
    [values.host, values.rtspPort, values.streamPath],
  );

  const submit = () => {
    const result = validateCameraConfig(values);
    if (!result.ok) {
      setErrors(result.errors);
      return;
    }
    setErrors({});
    onSubmit(result.value);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl border-border bg-surface">
        <DialogHeader>
          <DialogTitle className="font-mono text-sm uppercase tracking-[0.1em]">
            {camera ? "Edit camera" : "Add camera"}
          </DialogTitle>
          <DialogDescription className="text-[11px]">
            Configuration only. Runtime status, heartbeat and recording state are reported by the AI
            service and cannot be set here.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Camera name *" error={errors["name"]}>
            <Input
              value={values.name}
              onChange={(event) => set("name", event.target.value)}
              placeholder="Front Hall Camera"
              className="h-8 text-xs"
            />
          </Field>
          <Field label="Location" error={errors["location"]}>
            <Input
              value={values.location ?? ""}
              onChange={(event) => set("location", event.target.value)}
              placeholder="Hall A"
              className="h-8 text-xs"
            />
          </Field>
          <Field
            label="Source type *"
            hint={
              values.sourceType === "nvr_channel"
                ? "Host is the NVR address; channel selects the NVR camera."
                : "Host is the camera address on the local network."
            }
          >
            <Select
              value={values.sourceType}
              onValueChange={(value) => set("sourceType", value as CameraFormValues["sourceType"])}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="direct_camera">Direct IP camera</SelectItem>
                <SelectItem value="nvr_channel">NVR channel</SelectItem>
                <SelectItem value="demo">Demo source</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field label="Host / IP *" error={errors["host"]}>
            <Input
              value={values.host}
              onChange={(event) => set("host", event.target.value)}
              placeholder="10.77.10.100"
              className="h-8 font-mono text-xs"
            />
          </Field>
          <Field label="RTSP port *" error={errors["rtspPort"]}>
            <Input
              type="number"
              value={values.rtspPort}
              onChange={(event) => set("rtspPort", event.target.value)}
              className="h-8 font-mono text-xs"
            />
          </Field>
          <Field label="Channel" error={errors["channel"]}>
            <Input
              type="number"
              value={values.channel}
              onChange={(event) => set("channel", event.target.value)}
              className="h-8 font-mono text-xs"
            />
          </Field>
          <Field label="Stream path" error={errors["streamPath"]}>
            <Input
              value={values.streamPath ?? ""}
              onChange={(event) => set("streamPath", event.target.value)}
              placeholder="/stream2"
              className="h-8 font-mono text-xs"
            />
          </Field>
          <Field label="Stream profile">
            <Select
              value={values.streamProfile}
              onValueChange={(value) =>
                set("streamProfile", value as CameraFormValues["streamProfile"])
              }
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="main">Main stream</SelectItem>
                <SelectItem value="sub">Sub stream</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field label="Resolution" error={errors["resolution"]}>
            <Input
              value={values.resolution ?? ""}
              onChange={(event) => set("resolution", event.target.value)}
              placeholder="2560x1440"
              className="h-8 font-mono text-xs"
            />
          </Field>
          <Field label="Configured FPS" error={errors["fps"]}>
            <Input
              type="number"
              value={values.fps}
              onChange={(event) => set("fps", event.target.value)}
              className="h-8 font-mono text-xs"
            />
          </Field>
          <div className="flex items-center justify-between rounded-[4px] border border-border/70 bg-surface-2/50 px-2.5 py-2 sm:col-span-2">
            <span className="label-tech text-muted-foreground">AI enabled</span>
            <Switch
              checked={Boolean(values.aiEnabled)}
              onCheckedChange={(value) => set("aiEnabled", value)}
            />
          </div>
        </div>
        <div className="space-y-1 rounded-[4px] border border-border/70 bg-background/60 p-2.5">
          <p className="label-tech text-muted-foreground">Stream reference (no credentials)</p>
          <p className="break-all font-mono text-[11px] text-primary">{preview}</p>
          <p className="text-[10px] text-muted-foreground">{CREDENTIALS_MESSAGE}</p>
        </div>
        <DialogFooter>
          <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button size="sm" onClick={submit} disabled={pending}>
            {pending ? "Saving…" : camera ? "Save changes" : "Add camera"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
