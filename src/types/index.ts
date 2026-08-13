export type UserRole = "administrator" | "operator";

/** Explicit application operating mode. Demo data is only ever used in "demo". */
export type OperationMode = "demo" | "live";

export interface AppUser {
  id: string;
  fullName: string;
  email: string;
  role: UserRole;
  status: "active" | "suspended";
  lastActiveAt: string;
}

export type CameraStatus = "online" | "offline" | "degraded";

/** How the future AI service reaches the stream. Never vendor-specific. */
export type CameraSourceType = "direct_camera" | "nvr_channel" | "demo";
export type CameraStreamProfile = "main" | "sub" | "custom";

export interface Camera {
  id: string;
  name: string;
  location: string;
  /** Host only. RTSP credentials are never exposed to the browser. */
  host: string;
  channel: number;
  sourceType: CameraSourceType;
  rtspPort: number;
  /** Non-secret stream path, e.g. "/stream2". Never contains credentials. */
  streamPath: string;
  streamProfile: CameraStreamProfile;
  /** false = archived. Archived cameras leave monitoring but keep history. */
  active: boolean;
  status: CameraStatus;
  aiEnabled: boolean;
  recording: boolean;
  resolution: string;
  fps: number;
  isDemo: boolean;
  lastHeartbeatAt: string;
  updatedAt: string;
}

/** Administrator-editable camera configuration. Runtime health is excluded. */
export interface CameraConfigInput {
  name: string;
  location: string;
  sourceType: CameraSourceType;
  host: string;
  rtspPort: number;
  channel: number;
  streamPath: string;
  streamProfile: CameraStreamProfile;
  resolution: string;
  fps: number;
  aiEnabled: boolean;
}

export type EventSeverity = "critical" | "warning" | "info";
export type EventStatus = "new" | "under_review" | "confirmed" | "rejected";

/**
 * Event types the current build knows how to present richly. The platform is
 * generic: the database accepts any identifier, so the UI must degrade
 * gracefully for unknown future types (smoking_detected, camera_offline, …).
 */
export type KnownEventType =
  "suspicious_cheating_activity" | "possible_cheating_activity" | "mobile_phone_detected";

/** Extensible: any string is valid, known values keep autocomplete. */
export type EventType = KnownEventType | (string & {});

/** What the AI knows about the phone/person relationship (never a review state). */
export type AssociationStatus = "associated" | "uncertain" | "unassociated" | "not_applicable";

/** Whether the record came from real hardware or seeded demonstration data. */
export type EventSourceMode = "live" | "demo";

/** Normalized (0–1) box relative to the frame, never pixel coordinates. */
export interface DetectionBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** One structured detection captured as evidence at event time. */
export interface DetectionEvidence {
  objectId: string;
  trackingId: string | null;
  className: string;
  confidence: number;
  bbox: DetectionBoundingBox;
  role: string;
  associatedPersonTrackingId: string | null;
  associationConfidence: number | null;
}

export interface DetectionEvent {
  id: string;
  type: EventType;
  severity: EventSeverity;
  status: EventStatus;
  cameraId: string;
  cameraName: string;
  ruleId: string;
  confidence: number;
  durationSeconds: number;
  /** Storage path inside the private snapshots bucket — never a URL. */
  snapshotPath: string | null;
  detectedAt: string;
  reviewedBy: string | null;
  /** When a human completed the review, if ever. */
  reviewedAt: string | null;
  /** Human reviewer note only — never a transport for AI evidence. */
  note: string | null;
  /** Temporary AI tracking identifier, not a real-world identity. */
  personTrackingId: string | null;
  triggerObjectClass: string | null;
  triggerConfidence: number | null;
  associationStatus: AssociationStatus;
  associationConfidence: number | null;
  /** Fractional seconds (e.g. 1.75). */
  detectionDurationSeconds: number | null;
  detectionFrameCount: number | null;
  evidence: DetectionEvidence[];
  sourceMode: EventSourceMode;
}

/** Canonical alias for the structured AI event contract. */
export type AIEvent = DetectionEvent;

export type DetectionAlertState = "normal" | "evaluating" | "alert" | "uncertain";

export interface DetectionOverlay {
  objectId: string;
  trackingId: string | null;
  className: "person" | "cell_phone";
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  associatedPersonId: string | null;
  associationConfidence: number | null;
  alertState: DetectionAlertState;
}

export interface AiRule {
  id: string;
  name: string;
  description: string;
  available: boolean;
  enabled: boolean;
  confidenceThreshold: number;
  minDurationSeconds: number;
  cooldownSeconds: number;
  severity: EventSeverity;
  cameraIds: string[];
  saveSnapshot: boolean;
  soundNotification: boolean;
  /** Minimum person-detection confidence before association is attempted. */
  personConfidenceThreshold: number;
  /** Minimum person↔trigger-object association confidence. */
  associationConfidenceThreshold: number;
  minMatchingFrames: number;
  requirePersonAssociation: boolean;
  /** Preserve very brief (possibly single-frame) visible-phone evidence. */
  instantDetectionEnabled: boolean;
  /** Stricter confidence required for single-frame instant evidence. */
  instantConfidenceThreshold: number;
}

export interface AiServiceStatus {
  online: boolean;
  version: string;
  model: string;
  device: string;
  inferenceFps: number;
  queueDepth: number;
  gpuLoadPercent: number;
  uptimeSeconds: number;
  lastPingAt: string;
  /** Heartbeat older than the freshness threshold. */
  stale: boolean;
  /** Record is a demonstration placeholder, not real hardware. */
  isDemo: boolean;
  /** No health record has ever been reported by a real service. */
  neverReported: boolean;
  /** Provider readiness flags reported by the AI service. Never contains secrets. */
  telegramConfigured: boolean;
  telegramReady: boolean;
}

/** Notification provider readiness reported by the local AI service heartbeat. */
export interface NotificationChannelReadiness {
  configured: boolean;
  ready: boolean;
}

export interface NvrStatus {
  online: boolean;
  model: string;
  channelsUsed: number;
  channelsTotal: number;
  storageUsedPercent: number;
  retentionDays: number;
  lastSyncAt: string;
  stale: boolean;
  isDemo: boolean;
  neverReported: boolean;
  /**
   * Evidence-based recording state reported by the NVR/service heartbeat.
   * null = never reported. The UI must not claim recording when unknown.
   */
  recordingActive: boolean | null;
}

/** Predictable overall posture derived from independent component health. */
export type SystemHealthState = "ready" | "degraded" | "not_ready";

export interface CameraFleetSummary {
  total: number;
  online: number;
  offline: number;
  degraded: number;
  aiEnabled: number;
  /** Only cameras whose recording state is provably active. */
  recording: number;
  /** Cameras whose recording state cannot be proven (stale camera or NVR). */
  recordingUnknown: number;
}

export interface EventsSummary {
  today: number;
  critical: number;
  pendingReview: number;
  confirmed: number;
  rejected: number;
}

export interface ReportPoint {
  label: string;
  events: number;
  confirmed: number;
}

export interface ReportSummary {
  range: "7d" | "30d";
  mode: OperationMode;
  totalEvents: number;
  timeline: ReportPoint[];
  byCamera: { cameraName: string; events: number }[];
  bySeverity: { critical: number; warning: number; info: number };
  byType: { type: string; events: number }[];
  confirmed: number;
  rejected: number;
  pending: number;
  averageConfidence: number;
  confirmationRate: number;
  /** Null when no completed reviews exist in range. */
  averageReviewMinutes: number | null;
}

export interface SystemSettings {
  operationMode: OperationMode;
  aiServiceUrl: string;
  websocketUrl: string;
  retentionDays: number;
  snapshotStorage: "local" | "cloud";
  soundAlerts: boolean;
  autoAcknowledgeMinutes: number;
  timezone: string;
}

export interface AuthSession {
  user: AppUser;
  issuedAt: string;
}

/**
 * Exam session lifecycle. `active` is only ever set by the future Start Exam
 * Session runtime action; this foundation never claims monitoring has begun.
 */
export type ExamSessionStatus = "draft" | "ready" | "active" | "ended" | "archived";

export interface ExamSession {
  id: string;
  title: string;
  /** Optional free text, e.g. "CS201". */
  courseCode: string;
  /** Optional free text hall label. No seat/hall registration exists. */
  locationLabel: string;
  scheduledAt: string | null;
  status: ExamSessionStatus;
  startedAt: string | null;
  endedAt: string | null;
  createdBy: string | null;
  createdAt: string;
  updatedAt: string;
  /** Linked existing cameras. Credentials are never included. */
  cameraIds: string[];
  primaryCameraId: string | null;
  /** Session metadata only — no visual staff recognition exists. */
  invigilators: ExamInvigilator[];
  rosterCount: number;
}

export interface ExamSessionInput {
  title: string;
  courseCode: string;
  locationLabel: string;
  scheduledAt: string | null;
  primaryCameraId: string | null;
  invigilatorNames: string[];
}

export interface ExamInvigilator {
  id: string;
  fullName: string;
  role: string;
}

/**
 * A real student record scoped to one exam session. Students are never
 * application users and never carry an AI tracking or subject identity.
 */
export interface RosterStudent {
  id: string;
  examSessionId: string;
  universityId: string;
  fullName: string;
  createdAt: string;
  updatedAt: string;
}

export interface RosterStudentInput {
  universityId: string;
  fullName: string;
}

/**
 * Truthful tracking state of one anonymous subject, as reported by the AI
 * service. `uncertain` and `conflict` are preserved deliberately: the system
 * never upgrades ambiguous tracking evidence into a confident claim.
 */
export type SubjectTrackingStatus =
  "stable" | "temporarily_lost" | "uncertain" | "conflict" | "ended";

/**
 * An anonymous exam-session subject (S001, S002, …).
 *
 * This is NOT an identity: it carries no name, no university ID, no face, no
 * biometric signature and no seat. Resolution to a roster student is manual,
 * optional and on demand.
 */
export interface SessionSubject {
  id: string;
  examSessionId: string;
  subjectNumber: number;
  /** Deterministic per-session label, e.g. "S017". */
  label: string;
  cameraId: string | null;
  trackingStatus: SubjectTrackingStatus;
  firstSeenAt: string;
  lastSeenAt: string;
  endedAt: string | null;
  /** How often a lost raw track had to be recovered for this subject. */
  reassociationCount: number;
  /** Confidence of the most recent recovery, or null when never recovered. */
  lastAssociationConfidence: number | null;
}
