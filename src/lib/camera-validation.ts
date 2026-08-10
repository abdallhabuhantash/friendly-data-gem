import { z } from "zod";
import type { CameraConfigInput } from "@/types";

/**
 * Credentials never belong in browser-visible camera configuration. They are
 * configured later in the local Python AI service environment.
 */
export const CREDENTIALS_MESSAGE =
  "Credentials are not stored here. Configure the camera username and password in the local AI service configuration.";

const hostPattern = /^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$/;

export const cameraConfigSchema = z.object({
  name: z.string().trim().min(1, { message: "Camera name is required" }).max(80),
  location: z.string().trim().max(120).default(""),
  sourceType: z.enum(["direct_camera", "nvr_channel", "demo"]),
  host: z
    .string()
    .trim()
    .min(1, { message: "Host or IP is required" })
    .max(120)
    .refine((value) => !value.includes("@") && !value.includes("://"), {
      message: CREDENTIALS_MESSAGE,
    })
    .refine((value) => hostPattern.test(value), {
      message: "Enter a valid hostname or IP address",
    }),
  rtspPort: z.coerce
    .number()
    .int({ message: "Port must be a whole number" })
    .min(1, { message: "Port must be between 1 and 65535" })
    .max(65535, { message: "Port must be between 1 and 65535" }),
  channel: z.coerce
    .number()
    .int({ message: "Channel must be a whole number" })
    .min(1, { message: "Channel must be a positive integer" })
    .max(256),
  streamPath: z
    .string()
    .trim()
    .max(120)
    .refine((value) => !value.includes("@") && !/rtsp:\/\//i.test(value), {
      message: CREDENTIALS_MESSAGE,
    })
    .refine((value) => value === "" || value.startsWith("/"), {
      message: "Stream path must start with /",
    }),
  streamProfile: z.enum(["main", "sub", "custom"]),
  resolution: z.string().trim().max(24).default("1920x1080"),
  fps: z.coerce
    .number()
    .min(1, { message: "Configured FPS must be a positive number" })
    .max(120, { message: "Configured FPS looks unrealistic" }),
  aiEnabled: z.boolean(),
});

export type CameraFormValues = Omit<
  z.input<typeof cameraConfigSchema>,
  "rtspPort" | "channel" | "fps"
> & {
  /** Kept as raw input so partially typed numbers do not reset the field. */
  rtspPort: number | string;
  channel: number | string;
  fps: number | string;
};

export function validateCameraConfig(
  values: CameraFormValues,
): { ok: true; value: CameraConfigInput } | { ok: false; errors: Record<string, string> } {
  const parsed = cameraConfigSchema.safeParse(values);
  if (parsed.success) return { ok: true, value: parsed.data as CameraConfigInput };
  const errors: Record<string, string> = {};
  parsed.error.issues.forEach((issue) => {
    const key = String(issue.path[0] ?? "form");
    if (!errors[key]) errors[key] = issue.message;
  });
  return { ok: false, errors };
}

/** Non-credential RTSP preview shown to administrators. Never includes secrets. */
export function rtspPreview(input: {
  host: string;
  rtspPort: number | string;
  streamPath: string;
}): string {
  const path = input.streamPath?.trim() ?? "";
  return `rtsp://${input.host || "host"}:${input.rtspPort || 554}${path}`;
}
