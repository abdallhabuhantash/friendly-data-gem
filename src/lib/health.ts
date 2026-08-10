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

/**
 * Capability-level truth. Detection and recording are INDEPENDENT capabilities:
 * a deployment with no NVR can still be fully operational for AI detection.
 * `ready` stays reserved for every supported service being online, so the
 * subtext explains which optional capability is missing instead of implying
 * that detection itself has failed.
 */
export type SystemCapabilities = {
  detection: "operational" | "no_sources" | "unavailable";
  recording: "operational" | "stopped" | "unknown" | "not_configured";
  summary: string;
};

export function systemCapabilities(input: {
  ai: AiServiceStatus | undefined;
  nvr: NvrStatus | undefined;
  camerasOnline: number;
}): SystemCapabilities {
  const aiState = aiHealthState(input.ai);
  const aiUsable = aiState === "active" || aiState === "demo";
  const detection: SystemCapabilities["detection"] = !aiUsable
    ? "unavailable"
    : input.camerasOnline > 0
      ? "operational"
      : "no_sources";

  const nvrState = nvrHealthState(input.nvr);
  const recording: SystemCapabilities["recording"] =
    nvrState === "not_connected"
      ? "not_configured"
      : nvrState === "online" || nvrState === "demo"
        ? input.nvr?.recordingActive === true
          ? "operational"
          : input.nvr?.recordingActive === false
            ? "stopped"
            : "unknown"
        : "unknown";

  const detectionText =
    detection === "operational"
      ? "Detection operational"
      : detection === "no_sources"
        ? "Detection idle — no source online"
        : "Detection unavailable";
  const recordingText =
    recording === "operational"
      ? "recording active"
      : recording === "stopped"
        ? "recording stopped"
        : recording === "not_configured"
          ? "no recording service configured"
          : "recording state unknown";

  return { detection, recording, summary: `${detectionText} · ${recordingText}` };
}


/**
 * Recording truth. The NVR heartbeat only carries a GLOBAL `recording_active`
 * flag (see docs/service-health-contract.md) — there is no per-channel signal.
 * Recording is therefore only claimed as active when a fresh camera runtime
 * report and a fresh, explicit NVR `recordingActive = true` agree; anything
 * less is `unknown` rather than a false REC claim.
 */
export type RecordingState = "active" | "stopped" | "unknown";

export function effectiveRecordingState(
  camera: Camera,
  nvr: NvrStatus | undefined | null,
): RecordingState {
  if (isCameraStale(camera)) return "unknown";
  if (!nvr || nvr.neverReported || nvr.stale || nvr.recordingActive === null) return "unknown";
  if (nvr.recordingActive === false) return "stopped";
  return camera.recording ? "active" : "stopped";
}

export const recordingStateLabel: Record<RecordingState, string> = {
  active: "Active",
  stopped: "Stopped",
  unknown: "Unknown",
};
