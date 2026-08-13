import { useEffect, useState } from "react";
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
import { validateExamSession } from "@/lib/exam-validation";
import type { Camera, ExamSession, ExamSessionInput } from "@/types";

const NO_CAMERA = "none";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  cameras: Camera[];
  session?: ExamSession | null;
  pending: boolean;
  onSubmit: (input: ExamSessionInput) => void;
}

interface FormValues {
  title: string;
  courseCode: string;
  locationLabel: string;
  scheduledAt: string;
  primaryCameraId: string;
  invigilators: string;
}

const emptyValues: FormValues = {
  title: "",
  courseCode: "",
  locationLabel: "",
  scheduledAt: "",
  primaryCameraId: NO_CAMERA,
  invigilators: "",
};

/** Local datetime value for `<input type="datetime-local">`. */
const toLocalInput = (iso: string | null): string => {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
};

export function ExamSessionFormDialog({
  open,
  onOpenChange,
  cameras,
  session,
  pending,
  onSubmit,
}: Props) {
  const [values, setValues] = useState<FormValues>(emptyValues);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!open) return;
    setErrors({});
    setValues(
      session
        ? {
            title: session.title,
            courseCode: session.courseCode,
            locationLabel: session.locationLabel,
            scheduledAt: toLocalInput(session.scheduledAt),
            primaryCameraId: session.primaryCameraId ?? NO_CAMERA,
            invigilators: session.invigilators.map((person) => person.fullName).join(", "),
          }
        : emptyValues,
    );
  }, [open, session]);

  const set = <K extends keyof FormValues>(key: K, value: FormValues[K]) =>
    setValues((current) => ({ ...current, [key]: value }));

  const submit = () => {
    const result = validateExamSession({
      title: values.title,
      courseCode: values.courseCode,
      locationLabel: values.locationLabel,
      scheduledAt: values.scheduledAt === "" ? null : new Date(values.scheduledAt).toISOString(),
      primaryCameraId: values.primaryCameraId === NO_CAMERA ? null : values.primaryCameraId,
      invigilatorNames: values.invigilators
        .split(",")
        .map((name) => name.trim())
        .filter((name) => name !== ""),
    });
    if (!result.ok) {
      setErrors(result.errors);
      return;
    }
    setErrors({});
    onSubmit(result.value);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{session ? "Edit exam session" : "New exam session"}</DialogTitle>
          <DialogDescription>
            Location is optional free text. No seats are registered and students never become
            application users.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-3">
          <Field label="Exam title" error={errors["title"]}>
            <Input
              value={values.title}
              onChange={(event) => set("title", event.target.value)}
              placeholder="Data Structures — Midterm"
            />
          </Field>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Course code (optional)" error={errors["courseCode"]}>
              <Input
                value={values.courseCode}
                onChange={(event) => set("courseCode", event.target.value)}
                placeholder="CS201"
              />
            </Field>
            <Field label="Hall / location label (optional)" error={errors["locationLabel"]}>
              <Input
                value={values.locationLabel}
                onChange={(event) => set("locationLabel", event.target.value)}
                placeholder="Hall B"
              />
            </Field>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Scheduled start (optional)" error={errors["scheduledAt"]}>
              <Input
                type="datetime-local"
                value={values.scheduledAt}
                onChange={(event) => set("scheduledAt", event.target.value)}
              />
            </Field>
            <Field label="Primary camera (optional)" error={errors["primaryCameraId"]}>
              <Select
                value={values.primaryCameraId}
                onValueChange={(value) => set("primaryCameraId", value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="No camera" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_CAMERA}>No camera</SelectItem>
                  {cameras.map((camera) => (
                    <SelectItem key={camera.id} value={camera.id}>
                      {camera.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>
          <Field
            label="Invigilators (comma separated, optional)"
            error={errors["invigilatorNames"]}
          >
            <Input
              value={values.invigilators}
              onChange={(event) => set("invigilators", event.target.value)}
              placeholder="Dr. Ahmad, Ms. Rana"
            />
            <p className="mt-1 text-[11px] text-muted-foreground">
              Session metadata only. The AI does not visually recognise staff.
            </p>
          </Field>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={pending}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={pending}>
            {pending ? "Saving…" : session ? "Save changes" : "Create session"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  error,
  children,
}: {
  label: string;
  error?: string | undefined;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="label-tech">{label}</span>
      <div className="mt-1">{children}</div>
      {error && <p className="mt-1 text-[11px] text-destructive">{error}</p>}
    </label>
  );
}
