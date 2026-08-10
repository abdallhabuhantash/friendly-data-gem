import type { AiServiceStatus, Camera, CameraStatus, NvrStatus, SystemHealthState } from "@/types";

/** Heartbeat freshness thresholds. A stored `online` flag alone is never trusted. */
export const AI_HEARTBEAT_STALE_MS = 30_000;
export const NVR_HEARTBEAT_STALE_MS = 120_000;
export const CAMERA_HEARTBEAT_STALE_MS = 60_000;

export function isFresh(timestamp: string | null | undefined, thresholdMs: number): boolean {
  if (!timestamp) return false;
  const at = new Date(timestamp).getTime();
  if (Number.isNaN(at)) return false;
  return Date.now() - at <= thresholdMs;
}

/**
 * UI status for a camera. The database `status` column is preserved as the
 * reported value, but a camera whose heartbeat has stopped is shown offline.
 */
export function effectiveCameraStatus(camera: Camera): CameraStatus {
  if (!isFresh(camera.lastHeartbeatAt, CAMERA_HEARTBEAT_STALE_MS)) return "offline";
  return camera.status;
}

export function isCameraStale(camera: Camera): boolean {
  return !isFresh(camera.lastHeartbeatAt, CAMERA_HEARTBEAT_STALE_MS);
}

/** Truthful component state used by every status surface. */
export type ComponentHealth = "active" | "online" | "stale" | "offline" | "not_connected" | "demo";

export function aiHealthState(ai: AiServiceStatus | undefined): ComponentHealth {
  if (!ai || ai.neverReported) return "not_connected";
  if (ai.isDemo) return "demo";
  if (ai.online) return "active";
  if (ai.stale) return "stale";
  return "offline";
}

export function nvrHealthState(nvr: NvrStatus | undefined): ComponentHealth {
  if (!nvr || nvr.neverReported) return "not_connected";
  if (nvr.isDemo) return "demo";
  if (nvr.online) return "online";
  if (nvr.stale) return "stale";
  return "offline";
}

export const componentHealthLabel: Record<ComponentHealth, string> = {
  active: "Active",
  online: "Online",
  stale: "Stale",
  offline: "Offline",
  not_connected: "Not Connected",
  demo: "Demo",
};

/**
 * Overall posture. Component states stay independent: a missing NVR degrades
 * the system (no recording) but never marks AI inference itself offline.
 */
export function systemHealthState(input: {
  ai: AiServiceStatus | undefined;
  nvr: NvrStatus | undefined;
  camerasOnline: number;
}): SystemHealthState {
  const aiUsable = aiHealthState(input.ai) === "active";
  const cameraUsable = input.camerasOnline > 0;
  if (!aiUsable && !cameraUsable) return "not_ready";
  const nvrUsable = nvrHealthState(input.nvr) === "online";
  if (aiUsable && cameraUsable && nvrUsable) return "ready";
  return "degraded";
}

export const systemHealthLabel: Record<SystemHealthState, string> = {
  ready: "Ready",
  degraded: "Degraded",
  not_ready: "Not Ready",
};
