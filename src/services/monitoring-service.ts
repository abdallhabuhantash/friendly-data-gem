/**
 * Service layer. Every screen reads through these functions only.
 * All data now comes from the cloud backend; detection events are written
 * by the external Python AI service using its service credentials.
 */
import { supabase } from "@/integrations/supabase/client";
import {
  AI_HEARTBEAT_STALE_MS,
  NVR_HEARTBEAT_STALE_MS,
  effectiveCameraStatus,
  effectiveRecordingState,
  isFresh,
} from "@/lib/health";
import { effectiveSeverity } from "@/lib/event-presentation";
import { addDays, startOfZonedDay, zonedShortDateLabel, zonedWeekdayLabel } from "@/lib/time-zone";
import type { Json, Tables, TablesInsert, TablesUpdate } from "@/integrations/supabase/types";
import type {
  AiRule,
  AiServiceStatus,
  AppUser,
  AssociationStatus,
  Camera,
  CameraConfigInput,
  CameraFleetSummary,
  DetectionEvent,
  DetectionEvidence,
  EventSourceMode,
  EventStatus,
  EventsSummary,
  NvrStatus,
  OperationMode,
  ReportSummary,
  SystemSettings,
} from "@/types";

// Row shapes come from the generated backend types; mappers translate them
// into the UI-facing domain types declared in `@/types`.
type CameraRow = Tables<"cameras">;
type EventRow = Tables<"events">;
type RuleRow = Tables<"ai_rules">;
type ProfileRow = Tables<"profiles">;
type ServiceHealthRow = Tables<"service_health">;

/** Narrows a jsonb column to a readable key/value bag. */
const jsonRecord = (value: Json | null | undefined): Record<string, Json | undefined> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, Json | undefined>)
    : {};

const numberOrNull = (value: Json | null | undefined): number | null =>
  value === null || value === undefined ? null : Number(value);

const stringOrNull = (value: Json | null | undefined): string | null =>
  typeof value === "string" ? value : null;

/** Maps the jsonb evidence array into the typed frontend contract. */
const toEvidence = (value: Json | null | undefined): DetectionEvidence[] => {
  if (!Array.isArray(value)) return [];
  return value.map((entry) => {
    const item = jsonRecord(entry);
    const bbox = jsonRecord(item["bbox"]);
    return {
      objectId: stringOrNull(item["object_id"]) ?? "",
      trackingId: stringOrNull(item["tracking_id"]),
      className: stringOrNull(item["class_name"]) ?? "object",
      confidence: Number(item["confidence"] ?? 0),
      bbox: {
        x: Number(bbox["x"] ?? 0),
        y: Number(bbox["y"] ?? 0),
        width: Number(bbox["width"] ?? 0),
        height: Number(bbox["height"] ?? 0),
      },
      role: stringOrNull(item["role"]) ?? "object",
      associatedPersonTrackingId: stringOrNull(item["associated_person_tracking_id"]),
      associationConfidence: numberOrNull(item["association_confidence"] ?? null),
    };
  });
};

const fail = (error: { message: string } | null): void => {
  if (error) throw new Error(error.message);
};

const toCamera = (row: CameraRow): Camera => ({
  id: row.id as string,
  name: row.name as string,
  location: row.location as string,
  host: row.host as string,
  channel: row.channel as number,
  sourceType: (row.source_type as Camera["sourceType"]) ?? "direct_camera",
  rtspPort: Number(row.rtsp_port ?? 554),
  streamPath: (row.stream_path as string) ?? "",
  streamProfile: (row.stream_profile as Camera["streamProfile"]) ?? "main",
  active: row.active !== false,
  status: row.status as Camera["status"],
  aiEnabled: row.ai_enabled as boolean,
  recording: row.recording as boolean,
  resolution: row.resolution as string,
  fps: row.fps as number,
  isDemo: row.is_demo as boolean,
  lastHeartbeatAt: row.last_heartbeat_at as string,
  updatedAt: (row.updated_at as string) ?? (row.created_at as string),
});

const toEvent = (row: EventRow): DetectionEvent => ({
  id: row.id as string,
  type: row.type as DetectionEvent["type"],
  severity: row.severity as DetectionEvent["severity"],
  status: row.status as EventStatus,
  cameraId: (row.camera_id as string) ?? "",
  cameraName: row.camera_name as string,
  ruleId: (row.rule_id as string) ?? "",
  confidence: Number(row.confidence ?? 0),
  durationSeconds: row.duration_seconds as number,
  snapshotPath: (row.snapshot_path as string) ?? null,
  detectedAt: row.detected_at as string,
  reviewedBy: (row.reviewed_by as string) ?? null,
  reviewedAt: (row.reviewed_at as string) ?? null,
  note: (row.note as string) ?? null,
  personTrackingId: row.person_tracking_id ?? null,
  triggerObjectClass: row.trigger_object_class ?? null,
  triggerConfidence: row.trigger_confidence === null ? null : Number(row.trigger_confidence),
  associationStatus: (row.association_status as AssociationStatus) ?? "not_applicable",
  associationConfidence:
    row.association_confidence === null ? null : Number(row.association_confidence),
  detectionDurationSeconds:
    row.detection_duration_seconds === null
      ? Number(row.duration_seconds ?? 0)
      : Number(row.detection_duration_seconds),
  detectionFrameCount: row.detection_frame_count ?? null,
  evidence: toEvidence(row.evidence),
  sourceMode: (row.source_mode as EventSourceMode) ?? "live",
});

const toRule = (row: RuleRow, cameraIds: string[]): AiRule => ({
  id: row.id as string,
  name: row.name as string,
  description: row.description as string,
  available: row.available as boolean,
  enabled: row.enabled as boolean,
  confidenceThreshold: Number(row.confidence_threshold ?? 0),
  minDurationSeconds: Number(row.min_duration_seconds ?? 0),
  cooldownSeconds: row.cooldown_seconds as number,
  severity: row.severity as AiRule["severity"],
  cameraIds,
  saveSnapshot: row.save_snapshot as boolean,
  soundNotification: row.sound_notification as boolean,
  personConfidenceThreshold: Number(row.person_confidence_threshold ?? 0.6),
  associationConfidenceThreshold: Number(row.association_confidence_threshold ?? 0.65),
  minMatchingFrames: Number(row.min_matching_frames ?? 5),
  requirePersonAssociation: Boolean(row.require_person_association),
});

/** Which archive state a camera listing should include. */
export type CameraScope = "active" | "archived" | "all";

const toCameraRowPatch = (input: CameraConfigInput): TablesUpdate<"cameras"> => ({
  name: input.name,
  location: input.location,
  source_type: input.sourceType,
  // `is_demo` is always derived from the source type so demo/live filtering
  // can never drift out of sync after an edit.
  is_demo: input.sourceType === "demo",
  host: input.host,
  rtsp_port: input.rtspPort,
  channel: input.channel,
  stream_path: input.streamPath,
  stream_profile: input.streamProfile,
  resolution: input.resolution,
  fps: input.fps,
  ai_enabled: input.aiEnabled,
});

export const camerasService = {
  list: async (mode?: OperationMode, scope: CameraScope = "active"): Promise<Camera[]> => {
    const { data, error } = await supabase.from("cameras").select("*").order("channel");
    fail(error);
    let cameras = (data ?? []).map(toCamera);
    if (scope === "active") cameras = cameras.filter((camera) => camera.active);
    if (scope === "archived") cameras = cameras.filter((camera) => !camera.active);
    if (mode === "live") return cameras.filter((camera) => !camera.isDemo);
    if (mode === "demo") return cameras.filter((camera) => camera.isDemo);
    return cameras;
  },
  get: async (id: string): Promise<Camera | undefined> => {
    const { data } = await supabase.from("cameras").select("*").eq("id", id).maybeSingle();
    return data ? toCamera(data) : undefined;
  },
  /**
   * Fleet counters use the same heartbeat-aware status as the table rows, so a
   * stale camera can never be counted as online. Archived cameras are excluded.
   */
  summary: async (mode?: OperationMode): Promise<CameraFleetSummary> => {
    const cameras = await camerasService.list(mode, "active");
    const statuses = cameras.map(effectiveCameraStatus);
    // Recording counts go through the shared recording helper: the NVR flag is
    // global, so a camera is only counted recording when both reports agree.
    const nvr = await systemService.nvrStatus(mode ?? "live");
    const recordingStates = cameras.map((camera) => effectiveRecordingState(camera, nvr));
    return {
      total: cameras.length,
      online: statuses.filter((status) => status === "online").length,
      offline: statuses.filter((status) => status === "offline").length,
      degraded: statuses.filter((status) => status === "degraded").length,
      aiEnabled: cameras.filter((c) => c.aiEnabled).length,
      recording: recordingStates.filter((state) => state === "active").length,
      recordingUnknown: recordingStates.filter((state) => state === "unknown").length,
    };
  },
  toggleAi: async (id: string, enabled: boolean): Promise<void> => {
    const { error } = await supabase.from("cameras").update({ ai_enabled: enabled }).eq("id", id);
    fail(error);
  },
  /**
   * Creates configuration only. Runtime health (status, heartbeat, recording)
   * stays at its defaults until the AI service reports it.
   */
  create: async (input: CameraConfigInput): Promise<void> => {
    const payload: TablesInsert<"cameras"> = {
      ...toCameraRowPatch(input),
      name: input.name,
      is_demo: input.sourceType === "demo",
      active: true,
      status: "offline",
    };
    const { error } = await supabase.from("cameras").insert(payload);
    fail(error);
  },
  update: async (id: string, input: CameraConfigInput): Promise<void> => {
    const { error } = await supabase.from("cameras").update(toCameraRowPatch(input)).eq("id", id);
    fail(error);
  },
  /**
   * Archive instead of delete: historical events keep referencing the camera.
   * Rule assignments are dropped so the future AI service ignores it.
   */
  archive: async (id: string): Promise<void> => {
    const { error } = await supabase
      .from("cameras")
      .update({ active: false, ai_enabled: false })
      .eq("id", id);
    fail(error);
    const unlink = await supabase.from("ai_rule_cameras").delete().eq("camera_id", id);
    fail(unlink.error);
  },
  /** Restores configuration only — rules must be reassigned explicitly. */
  restore: async (id: string): Promise<void> => {
    const { error } = await supabase.from("cameras").update({ active: true }).eq("id", id);
    fail(error);
  },
};

export const eventsService = {
  list: async (mode?: OperationMode): Promise<DetectionEvent[]> => {
    let query = supabase.from("events").select("*");
    if (mode) query = query.eq("source_mode", mode);
    const { data, error } = await query.order("detected_at", { ascending: false }).limit(300);
    fail(error);
    return (data ?? []).map(toEvent);
  },
  recent: async (limit: number, mode?: OperationMode): Promise<DetectionEvent[]> => {
    let query = supabase.from("events").select("*");
    if (mode) query = query.eq("source_mode", mode);
    const { data, error } = await query.order("detected_at", { ascending: false }).limit(limit);
    fail(error);
    return (data ?? []).map(toEvent);
  },
  /**
   * Events belonging to the current Asia/Amman calendar day, scoped to the
   * active operation mode. Filtered at the database level.
   */
  today: async (mode?: OperationMode): Promise<DetectionEvent[]> => {
    const dayStart = startOfZonedDay();
    const dayEnd = startOfZonedDay(addDays(dayStart, 1));
    let query = supabase
      .from("events")
      .select("*")
      .gte("detected_at", dayStart.toISOString())
      .lt("detected_at", dayEnd.toISOString());
    if (mode) query = query.eq("source_mode", mode);
    const { data, error } = await query.order("detected_at", { ascending: false });
    fail(error);
    return (data ?? []).map(toEvent);
  },
  /**
   * "Today" counters use the Asia/Amman calendar day; critical uses the shared
   * effective severity so uncertain associations are never counted as critical.
   */
  summary: async (mode?: OperationMode): Promise<EventsSummary> => {
    const [today, events] = await Promise.all([
      eventsService.today(mode),
      eventsService.list(mode),
    ]);
    return {
      today: today.length,
      critical: today.filter((e) => effectiveSeverity(e) === "critical").length,
      pendingReview: events.filter((e) => e.status === "new" || e.status === "under_review").length,
      confirmed: events.filter((e) => e.status === "confirmed").length,
      rejected: events.filter((e) => e.status === "rejected").length,
    };
  },
  /**
   * Review goes through a secure database function. Detection evidence
   * (type, confidence, camera, rule, timestamps, snapshot) cannot be changed
   * by any signed-in user; only the review outcome is writable.
   */
  review: async (id: string, status: EventStatus, note?: string): Promise<void> => {
    const { error } = await supabase.rpc("review_event", {
      _event_id: id,
      _status: status,
      ...(note === undefined ? {} : { _note: note }),
    });
    fail(error);
  },
  /** Temporary signed URL for a private snapshot. Never persisted anywhere. */
  createSnapshotSignedUrl: async (path: string | null): Promise<string | null> => {
    if (!path) return null;
    const { data } = await supabase.storage.from("snapshots").createSignedUrl(path, 300);
    return data?.signedUrl ?? null;
  },
};

export const rulesService = {
  list: async (): Promise<AiRule[]> => {
    const [rules, links] = await Promise.all([
      supabase.from("ai_rules").select("*").order("created_at"),
      supabase.from("ai_rule_cameras").select("*"),
    ]);
    fail(rules.error);
    fail(links.error);
    return (rules.data ?? []).map((row) =>
      toRule(
        row,
        (links.data ?? [])
          .filter((link) => link.rule_id === row.id)
          .map((link) => link.camera_id as string),
      ),
    );
  },
  update: async (id: string, patch: Partial<AiRule>): Promise<void> => {
    const payload: TablesUpdate<"ai_rules"> = {};
    if (patch.enabled !== undefined) payload.enabled = patch.enabled;
    if (patch.confidenceThreshold !== undefined)
      payload.confidence_threshold = patch.confidenceThreshold;
    if (patch.minDurationSeconds !== undefined)
      payload.min_duration_seconds = patch.minDurationSeconds;
    if (patch.cooldownSeconds !== undefined) payload.cooldown_seconds = patch.cooldownSeconds;
    if (patch.severity !== undefined) payload.severity = patch.severity;
    if (patch.saveSnapshot !== undefined) payload.save_snapshot = patch.saveSnapshot;
    if (patch.soundNotification !== undefined) payload.sound_notification = patch.soundNotification;
    if (patch.personConfidenceThreshold !== undefined)
      payload.person_confidence_threshold = patch.personConfidenceThreshold;
    if (patch.associationConfidenceThreshold !== undefined)
      payload.association_confidence_threshold = patch.associationConfidenceThreshold;
    if (patch.minMatchingFrames !== undefined)
      payload.min_matching_frames = patch.minMatchingFrames;
    if (patch.requirePersonAssociation !== undefined)
      payload.require_person_association = patch.requirePersonAssociation;
    if (Object.keys(payload).length === 0) return;
    const { error } = await supabase.from("ai_rules").update(payload).eq("id", id);
    fail(error);
  },
  /** Persists rule→camera assignment through the existing join table. */
  setCameras: async (ruleId: string, cameraIds: string[]): Promise<void> => {
    const remove = await supabase.from("ai_rule_cameras").delete().eq("rule_id", ruleId);
    fail(remove.error);
    if (cameraIds.length === 0) return;
    const { error } = await supabase
      .from("ai_rule_cameras")
      .insert(cameraIds.map((cameraId) => ({ rule_id: ruleId, camera_id: cameraId })));
    fail(error);
  },
};

const toUser = (row: ProfileRow, role: string | undefined): AppUser => ({
  id: row.id as string,
  fullName: (row.full_name as string) || (row.email as string),
  email: row.email as string,
  role: (role as AppUser["role"]) ?? "operator",
  status: row.status as AppUser["status"],
  lastActiveAt: row.last_active_at as string,
});

export const usersService = {
  list: async (): Promise<AppUser[]> => {
    const [profiles, roles] = await Promise.all([
      supabase.from("profiles").select("*").order("created_at"),
      supabase.from("user_roles").select("*"),
    ]);
    fail(profiles.error);
    fail(roles.error);
    return (profiles.data ?? []).map((row) =>
      toUser(
        row,
        (roles.data ?? []).find((entry) => entry.user_id === row.id)?.role as string | undefined,
      ),
    );
  },
  byId: async (id: string): Promise<AppUser | null> => {
    const [profile, roles] = await Promise.all([
      supabase.from("profiles").select("*").eq("id", id).maybeSingle(),
      supabase.from("user_roles").select("role").eq("user_id", id),
    ]);
    if (!profile.data) return null;
    return toUser(profile.data, (roles.data ?? [])[0]?.role as string | undefined);
  },
};

export const systemService = {
  operationMode: async (): Promise<OperationMode> => {
    const { data } = await supabase.from("system_settings").select("*").maybeSingle();
    return (data?.operation_mode as OperationMode) ?? "demo";
  },
  aiStatus: async (mode: OperationMode): Promise<AiServiceStatus> => {
    const { data } = await supabase
      .from("service_health")
      .select("*")
      .eq("service", "ai")
      .maybeSingle();
    const row: ServiceHealthRow | null = data;
    const isDemo = Boolean(row?.is_demo);
    // In live mode a demo placeholder is not a connected service.
    const usable = row && (mode === "demo" || !isDemo);
    const payload = usable ? jsonRecord(row.payload) : {};
    const lastPingAt = (row?.updated_at as string) ?? null;
    const stale = usable ? !isFresh(lastPingAt, AI_HEARTBEAT_STALE_MS) : true;
    // Notification readiness is only trusted while the heartbeat is fresh.
    const notificationChannels = jsonRecord(payload["notification_channels"]);
    const telegram = jsonRecord(notificationChannels["telegram"]);
    return {
      // A stored `online` flag is only believed while the heartbeat is fresh.
      online: Boolean(usable && row.online) && !stale,
      version: (payload["version"] as string) ?? "—",
      model: (payload["model"] as string) ?? "—",
      device: (payload["device"] as string) ?? "—",
      inferenceFps: Number(payload["inference_fps"] ?? 0),
      queueDepth: Number(payload["queue_depth"] ?? 0),
      gpuLoadPercent: Number(payload["gpu_load_percent"] ?? 0),
      uptimeSeconds: Number(payload["uptime_seconds"] ?? 0),
      lastPingAt: lastPingAt ?? "",
      stale,
      isDemo: Boolean(usable && isDemo),
      neverReported: !row || (mode === "live" && isDemo),
      telegramConfigured: !stale && telegram["configured"] === true,
      telegramReady: !stale && telegram["ready"] === true,
    };
  },
  nvrStatus: async (mode: OperationMode): Promise<NvrStatus> => {
    const { data } = await supabase
      .from("service_health")
      .select("*")
      .eq("service", "nvr")
      .maybeSingle();
    const row: ServiceHealthRow | null = data;
    const isDemo = Boolean(row?.is_demo);
    const usable = row && (mode === "demo" || !isDemo);
    const payload = usable ? jsonRecord(row.payload) : {};
    const lastSyncAt = (row?.updated_at as string) ?? null;
    const stale = usable ? !isFresh(lastSyncAt, NVR_HEARTBEAT_STALE_MS) : true;
    // Recording is only claimed when the heartbeat explicitly reports it.
    const reportedRecording = payload["recording_active"];
    const recordingActive =
      usable && !stale && typeof reportedRecording === "boolean" ? reportedRecording : null;
    return {
      online: Boolean(usable && row.online) && !stale,
      recordingActive,
      model: (payload["model"] as string) ?? "—",
      channelsUsed: Number(payload["channels_used"] ?? 0),
      channelsTotal: Number(payload["channels_total"] ?? 0),
      storageUsedPercent: Number(payload["storage_used_percent"] ?? 0),
      retentionDays: Number(payload["retention_days"] ?? 0),
      lastSyncAt: lastSyncAt ?? "",
      stale,
      isDemo: Boolean(usable && isDemo),
      neverReported: !row || (mode === "live" && isDemo),
    };
  },
  settings: async (): Promise<SystemSettings> => {
    const { data } = await supabase.from("system_settings").select("*").maybeSingle();
    return {
      operationMode: (data?.operation_mode as OperationMode) ?? "demo",
      aiServiceUrl: (data?.ai_service_url as string) ?? "",
      websocketUrl: (data?.websocket_url as string) ?? "",
      retentionDays: Number(data?.retention_days ?? 30),
      snapshotStorage: (data?.snapshot_storage as SystemSettings["snapshotStorage"]) ?? "cloud",
      soundAlerts: Boolean(data?.sound_alerts),
      autoAcknowledgeMinutes: Number(data?.auto_acknowledge_minutes ?? 30),
      timezone: (data?.timezone as string) ?? "Asia/Amman",
    };
  },
  updateSettings: async (patch: Partial<SystemSettings>): Promise<void> => {
    const payload: TablesInsert<"system_settings"> = {
      id: true,
      updated_at: new Date().toISOString(),
    };
    if (patch.operationMode !== undefined) payload.operation_mode = patch.operationMode;
    if (patch.aiServiceUrl !== undefined) payload.ai_service_url = patch.aiServiceUrl;
    if (patch.websocketUrl !== undefined) payload.websocket_url = patch.websocketUrl;
    if (patch.retentionDays !== undefined) payload["retention_days"] = patch.retentionDays;
    if (patch.snapshotStorage !== undefined) payload.snapshot_storage = patch.snapshotStorage;
    if (patch.soundAlerts !== undefined) payload.sound_alerts = patch.soundAlerts;
    if (patch.autoAcknowledgeMinutes !== undefined)
      payload.auto_acknowledge_minutes = patch.autoAcknowledgeMinutes;
    if (patch.timezone !== undefined) payload.timezone = patch.timezone;
    const { error } = await supabase.from("system_settings").upsert(payload);
    fail(error);
  },
};

/**
 * Average minutes between detection and a completed human review.
 * Null when no completed review exists; malformed/negative spans are ignored.
 */
export function averageReviewMinutes(events: DetectionEvent[]): number | null {
  const spans = events
    .filter((event) => event.status === "confirmed" || event.status === "rejected")
    .map((event) => {
      if (!event.reviewedAt) return null;
      const detected = new Date(event.detectedAt).getTime();
      const reviewed = new Date(event.reviewedAt).getTime();
      if (!Number.isFinite(detected) || !Number.isFinite(reviewed)) return null;
      const minutes = (reviewed - detected) / 60_000;
      return minutes < 0 ? null : minutes;
    })
    .filter((value): value is number => value !== null);
  if (spans.length === 0) return null;
  return spans.reduce((sum, value) => sum + value, 0) / spans.length;
}

export const reportsService = {
  /** Every report metric is scoped to one operation mode; demo and live never mix. */
  summary: async (range: "7d" | "30d", mode: OperationMode): Promise<ReportSummary> => {
    // Exact calendar-day coverage: 7d = 7 days, 30d = 30 days, both ending today.
    const totalDays = range === "7d" ? 7 : 30;
    const bucketDays = range === "7d" ? 1 : 6; // 7×1 or 5×6 — always exactly totalDays.
    const buckets = totalDays / bucketDays;
    const todayStart = startOfZonedDay();
    const rangeStart = startOfZonedDay(addDays(todayStart, -(totalDays - 1)));
    const rangeEnd = startOfZonedDay(addDays(todayStart, 1));

    const { data, error } = await supabase
      .from("events")
      .select("*")
      .eq("source_mode", mode)
      .gte("detected_at", rangeStart.toISOString())
      .lt("detected_at", rangeEnd.toISOString())
      .order("detected_at");
    fail(error);
    const events = (data ?? []).map(toEvent);

    const bucketBounds = Array.from({ length: buckets }, (_, index) => {
      const from = startOfZonedDay(addDays(rangeStart, index * bucketDays));
      const to = startOfZonedDay(addDays(from, bucketDays));
      return { from, to };
    });

    const timeline = bucketBounds.map(({ from, to }) => {
      const inBucket = events.filter((event) => {
        const at = new Date(event.detectedAt).getTime();
        return at >= from.getTime() && at < to.getTime();
      });
      return {
        label: range === "7d" ? zonedWeekdayLabel(from) : zonedShortDateLabel(from),
        events: inBucket.length,
        confirmed: inBucket.filter((event) => event.status === "confirmed").length,
      };
    });

    const byCameraMap = new Map<string, number>();
    const byTypeMap = new Map<string, number>();
    const bySeverity = { critical: 0, warning: 0, info: 0 };
    events.forEach((event) => {
      byCameraMap.set(event.cameraName, (byCameraMap.get(event.cameraName) ?? 0) + 1);
      // Unknown/future event types are counted safely by their raw identifier.
      byTypeMap.set(event.type, (byTypeMap.get(event.type) ?? 0) + 1);
      bySeverity[effectiveSeverity(event)] += 1;
    });

    const confirmed = events.filter((event) => event.status === "confirmed").length;
    const rejected = events.filter((event) => event.status === "rejected").length;
    const pending = events.filter(
      (event) => event.status === "new" || event.status === "under_review",
    ).length;
    const reviewedCount = confirmed + rejected;

    return {
      range,
      mode,
      totalEvents: events.length,
      timeline,
      byCamera: [...byCameraMap.entries()]
        .map(([cameraName, count]) => ({ cameraName, events: count }))
        .sort((a, b) => b.events - a.events),
      byType: [...byTypeMap.entries()]
        .map(([type, count]) => ({ type, events: count }))
        .sort((a, b) => b.events - a.events),
      bySeverity,
      confirmed,
      rejected,
      pending,
      averageConfidence: events.length
        ? events.reduce((sum, event) => sum + event.confidence, 0) / events.length
        : 0,
      confirmationRate: reviewedCount ? confirmed / reviewedCount : 0,
      averageReviewMinutes: averageReviewMinutes(events),
    };
  },
};
