import type { Camera, DetectionEvent, DetectionEvidence, DetectionOverlay } from "@/types";

const now = Date.now();

export const demoCameras: Camera[] = [
  {
    id: "demo-exam-front",
    name: "Exam Hall A — Front",
    location: "North Campus · Level 2",
    host: "demo",
    channel: 1,
    status: "online",
    aiEnabled: true,
    recording: true,
    resolution: "1920×1080",
    fps: 22,
    isDemo: true,
    sourceType: "demo",
    rtspPort: 554,
    streamPath: "",
    streamProfile: "main",
    active: true,
    updatedAt: new Date(now).toISOString(),
    lastHeartbeatAt: new Date(now).toISOString(),
  },
  {
    id: "demo-exam-rear",
    name: "Exam Hall A — Rear",
    location: "North Campus · Level 2",
    host: "demo",
    channel: 2,
    status: "online",
    aiEnabled: true,
    recording: true,
    resolution: "1920×1080",
    fps: 20,
    isDemo: true,
    sourceType: "demo",
    rtspPort: 554,
    streamPath: "",
    streamProfile: "main",
    active: true,
    updatedAt: new Date(now).toISOString(),
    lastHeartbeatAt: new Date(now - 4_000).toISOString(),
  },
  {
    id: "demo-exam-b",
    name: "Exam Hall B",
    location: "North Campus · Level 3",
    host: "demo",
    channel: 3,
    status: "degraded",
    aiEnabled: true,
    recording: true,
    resolution: "1280×720",
    fps: 14,
    isDemo: true,
    sourceType: "demo",
    rtspPort: 554,
    streamPath: "",
    streamProfile: "main",
    active: true,
    updatedAt: new Date(now).toISOString(),
    lastHeartbeatAt: new Date(now - 15_000).toISOString(),
  },
  {
    id: "demo-computer-lab",
    name: "Computer Lab",
    location: "Technology Building",
    host: "demo",
    channel: 4,
    status: "offline",
    aiEnabled: false,
    recording: false,
    resolution: "1920×1080",
    fps: 0,
    isDemo: true,
    sourceType: "demo",
    rtspPort: 554,
    streamPath: "",
    streamProfile: "main",
    active: true,
    updatedAt: new Date(now).toISOString(),
    lastHeartbeatAt: new Date(now - 420_000).toISOString(),
  },
];

/** Normalized (0–1) evidence, mirroring the Python AI service payload. */
const evidenceAssociated: DetectionEvidence[] = [
  {
    objectId: "person-03",
    trackingId: "03",
    className: "person",
    confidence: 0.94,
    bbox: { x: 0.43, y: 0.38, width: 0.19, height: 0.53 },
    role: "person",
    associatedPersonTrackingId: null,
    associationConfidence: null,
  },
  {
    objectId: "phone-17",
    trackingId: null,
    className: "cell_phone",
    confidence: 0.88,
    bbox: { x: 0.505, y: 0.68, width: 0.055, height: 0.1 },
    role: "trigger_object",
    associatedPersonTrackingId: "03",
    associationConfidence: 0.91,
  },
];

const evidenceUncertain: DetectionEvidence[] = [
  {
    objectId: "person-11",
    trackingId: "11",
    className: "person",
    confidence: 0.83,
    bbox: { x: 0.7, y: 0.42, width: 0.14, height: 0.44 },
    role: "person",
    associatedPersonTrackingId: null,
    associationConfidence: null,
  },
  {
    objectId: "phone-22",
    trackingId: null,
    className: "cell_phone",
    confidence: 0.62,
    bbox: { x: 0.76, y: 0.63, width: 0.04, height: 0.08 },
    role: "trigger_object",
    associatedPersonTrackingId: "11",
    associationConfidence: 0.54,
  },
];

const evidenceUnassociated: DetectionEvidence[] = [
  {
    objectId: "phone-31",
    trackingId: null,
    className: "cell_phone",
    confidence: 0.71,
    bbox: { x: 0.18, y: 0.72, width: 0.04, height: 0.07 },
    role: "trigger_object",
    associatedPersonTrackingId: null,
    associationConfidence: null,
  },
];

/** Converts normalized evidence into percentage-based viewport overlays. */
export function overlaysFromEvidence(
  evidence: DetectionEvidence[],
  alert: boolean,
): DetectionOverlay[] {
  return evidence.map((item) => ({
    objectId: item.objectId,
    trackingId: item.trackingId,
    className: item.className === "person" ? "person" : "cell_phone",
    confidence: item.confidence,
    x: item.bbox.x * 100,
    y: item.bbox.y * 100,
    width: item.bbox.width * 100,
    height: item.bbox.height * 100,
    associatedPersonId:
      item.associatedPersonTrackingId === null
        ? null
        : (evidence.find((candidate) => candidate.trackingId === item.associatedPersonTrackingId)
            ?.objectId ?? null),
    associationConfidence: item.associationConfidence,
    alertState:
      item.associationConfidence !== null && item.associationConfidence < 0.65
        ? "uncertain"
        : alert && item.role === "trigger_object"
          ? "alert"
          : item.className === "person" && alert
            ? "alert"
            : "normal",
  }));
}

export const demoDetections: DetectionOverlay[] = [
  ...overlaysFromEvidence(evidenceAssociated, true),
  ...overlaysFromEvidence(evidenceUncertain, false),
];

export const demoEvents: DetectionEvent[] = [
  {
    id: "demo-event-critical",
    type: "suspicious_cheating_activity",
    severity: "critical",
    status: "new",
    cameraId: "demo-exam-front",
    cameraName: "Exam Hall A — Front",
    ruleId: "mobile-phone-rule",
    confidence: 0.88,
    durationSeconds: 2,
    snapshotPath: null,
    detectedAt: new Date(now - 18_000).toISOString(),
    reviewedBy: null,
    reviewedAt: null,
    note: null,
    personTrackingId: "03",
    triggerObjectClass: "cell_phone",
    triggerConfidence: 0.88,
    associationStatus: "associated",
    associationConfidence: 0.91,
    detectionDurationSeconds: 1.8,
    detectionFrameCount: 34,
    evidence: evidenceAssociated,
    sourceMode: "demo",
  },
  {
    id: "demo-event-uncertain",
    type: "possible_cheating_activity",
    severity: "warning",
    status: "under_review",
    cameraId: "demo-exam-rear",
    cameraName: "Exam Hall A — Rear",
    ruleId: "mobile-phone-rule",
    confidence: 0.62,
    durationSeconds: 1,
    snapshotPath: null,
    detectedAt: new Date(now - 165_000).toISOString(),
    reviewedBy: null,
    reviewedAt: null,
    note: null,
    personTrackingId: "11",
    triggerObjectClass: "cell_phone",
    triggerConfidence: 0.62,
    associationStatus: "uncertain",
    associationConfidence: 0.54,
    detectionDurationSeconds: 0.9,
    detectionFrameCount: 17,
    evidence: evidenceUncertain,
    sourceMode: "demo",
  },
  {
    id: "demo-event-rejected",
    type: "mobile_phone_detected",
    severity: "info",
    status: "rejected",
    cameraId: "demo-exam-b",
    cameraName: "Exam Hall B",
    ruleId: "mobile-phone-rule",
    confidence: 0.71,
    durationSeconds: 1,
    snapshotPath: null,
    detectedAt: new Date(now - 520_000).toISOString(),
    reviewedBy: "Operator",
    reviewedAt: new Date(now - 400_000).toISOString(),
    note: "Reviewed as a false positive.",
    personTrackingId: null,
    triggerObjectClass: "cell_phone",
    triggerConfidence: 0.71,
    associationStatus: "unassociated",
    associationConfidence: null,
    detectionDurationSeconds: 1.1,
    detectionFrameCount: 21,
    evidence: evidenceUnassociated,
    sourceMode: "demo",
  },
];

export function mergeDemoCameras(cameras: Camera[]): Camera[] {
  if (cameras.length >= 4) return cameras;
  const channels = new Set(cameras.map((camera) => camera.channel));
  return [...cameras, ...demoCameras.filter((camera) => !channels.has(camera.channel))].slice(0, 4);
}
