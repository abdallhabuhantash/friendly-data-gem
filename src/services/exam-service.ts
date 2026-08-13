/**
 * Exam session + roster reads and writes. Row shapes come from the generated
 * backend types; mappers translate them into the domain types in `@/types`.
 * Writes are additionally gated by row level security (administrator only).
 */
import { supabase } from "@/integrations/supabase/client";
import type { Tables } from "@/integrations/supabase/types";
import type {
  ExamInvigilator,
  ExamSession,
  ExamSessionInput,
  ExamSessionStatus,
  RosterStudent,
  RosterStudentInput,
  SessionSubject,
  SubjectLifecycle,
  SubjectTrackAssociation,
} from "@/types";

type SessionRow = Tables<"exam_sessions">;
type CameraLinkRow = Tables<"exam_session_cameras">;
type InvigilatorRow = Tables<"exam_invigilators">;
type RosterRow = Tables<"exam_roster_students">;

const fail = (error: { message: string } | null): void => {
  if (error) throw new Error(error.message);
};

const toInvigilator = (row: InvigilatorRow): ExamInvigilator => ({
  id: row.id,
  fullName: row.full_name,
  role: row.role,
});

const toRosterStudent = (row: RosterRow): RosterStudent => ({
  id: row.id,
  examSessionId: row.exam_session_id,
  universityId: row.university_id,
  fullName: row.full_name,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
});

const toSession = (
  row: SessionRow,
  links: CameraLinkRow[],
  invigilators: InvigilatorRow[],
  rosterCount: number,
): ExamSession => ({
  id: row.id,
  title: row.title,
  courseCode: row.course_code,
  locationLabel: row.location_label,
  scheduledAt: row.scheduled_at,
  status: row.status as ExamSessionStatus,
  startedAt: row.started_at,
  endedAt: row.ended_at,
  createdBy: row.created_by,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
  cameraIds: links.map((link) => link.camera_id),
  primaryCameraId: links.find((link) => link.is_primary)?.camera_id ?? null,
  invigilators: invigilators.map(toInvigilator),
  rosterCount,
});

const groupBy = <T>(rows: T[], key: (row: T) => string): Map<string, T[]> => {
  const map = new Map<string, T[]>();
  rows.forEach((row) => {
    const id = key(row);
    const bucket = map.get(id);
    if (bucket) bucket.push(row);
    else map.set(id, [row]);
  });
  return map;
};

export const examSessionsService = {
  async list(): Promise<ExamSession[]> {
    const sessions = await supabase
      .from("exam_sessions")
      .select("*")
      .order("created_at", { ascending: false });
    fail(sessions.error);
    const rows = sessions.data ?? [];
    if (rows.length === 0) return [];
    const ids = rows.map((row) => row.id);

    const [links, invigilators, roster] = await Promise.all([
      supabase.from("exam_session_cameras").select("*").in("exam_session_id", ids),
      supabase
        .from("exam_invigilators")
        .select("*")
        .in("exam_session_id", ids)
        .order("created_at", { ascending: true }),
      supabase.from("exam_roster_students").select("id,exam_session_id").in("exam_session_id", ids),
    ]);
    fail(links.error);
    fail(invigilators.error);
    fail(roster.error);

    const linksBySession = groupBy(links.data ?? [], (row) => row.exam_session_id);
    const staffBySession = groupBy(invigilators.data ?? [], (row) => row.exam_session_id);
    const rosterBySession = groupBy(roster.data ?? [], (row) => row.exam_session_id);

    return rows.map((row) =>
      toSession(
        row,
        linksBySession.get(row.id) ?? [],
        staffBySession.get(row.id) ?? [],
        (rosterBySession.get(row.id) ?? []).length,
      ),
    );
  },

  async byId(id: string): Promise<ExamSession | null> {
    const session = await supabase.from("exam_sessions").select("*").eq("id", id).maybeSingle();
    fail(session.error);
    if (!session.data) return null;
    const [links, invigilators, roster] = await Promise.all([
      supabase.from("exam_session_cameras").select("*").eq("exam_session_id", id),
      supabase
        .from("exam_invigilators")
        .select("*")
        .eq("exam_session_id", id)
        .order("created_at", { ascending: true }),
      supabase.from("exam_roster_students").select("id").eq("exam_session_id", id),
    ]);
    fail(links.error);
    fail(invigilators.error);
    fail(roster.error);
    return toSession(
      session.data,
      links.data ?? [],
      invigilators.data ?? [],
      (roster.data ?? []).length,
    );
  },

  async create(input: ExamSessionInput): Promise<ExamSession> {
    const created = await supabase
      .from("exam_sessions")
      .insert({
        title: input.title,
        course_code: input.courseCode,
        location_label: input.locationLabel,
        scheduled_at: input.scheduledAt,
        status: "draft",
      })
      .select("*")
      .single();
    fail(created.error);
    const row = created.data as SessionRow;

    if (input.primaryCameraId) {
      const link = await supabase
        .from("exam_session_cameras")
        .insert({ exam_session_id: row.id, camera_id: input.primaryCameraId, is_primary: true });
      fail(link.error);
    }
    if (input.invigilatorNames.length > 0) {
      const staff = await supabase.from("exam_invigilators").insert(
        input.invigilatorNames.map((fullName) => ({
          exam_session_id: row.id,
          full_name: fullName,
          role: "invigilator",
        })),
      );
      fail(staff.error);
    }
    const session = await examSessionsService.byId(row.id);
    if (!session) throw new Error("The exam session could not be read back.");
    return session;
  },

  async update(id: string, input: ExamSessionInput): Promise<void> {
    const updated = await supabase
      .from("exam_sessions")
      .update({
        title: input.title,
        course_code: input.courseCode,
        location_label: input.locationLabel,
        scheduled_at: input.scheduledAt,
      })
      .eq("id", id);
    fail(updated.error);

    const cleared = await supabase
      .from("exam_session_cameras")
      .delete()
      .eq("exam_session_id", id)
      .eq("is_primary", true);
    fail(cleared.error);
    if (input.primaryCameraId) {
      const link = await supabase
        .from("exam_session_cameras")
        .upsert({ exam_session_id: id, camera_id: input.primaryCameraId, is_primary: true });
      fail(link.error);
    }

    const removedStaff = await supabase
      .from("exam_invigilators")
      .delete()
      .eq("exam_session_id", id);
    fail(removedStaff.error);
    if (input.invigilatorNames.length > 0) {
      const staff = await supabase.from("exam_invigilators").insert(
        input.invigilatorNames.map((fullName) => ({
          exam_session_id: id,
          full_name: fullName,
          role: "invigilator",
        })),
      );
      fail(staff.error);
    }
  },

  /**
   * Configuration-only status change. `ready` means "configured"; it never
   * starts monitoring — that is the future Start Exam Session action.
   */
  async setConfiguredStatus(id: string, status: Extract<ExamSessionStatus, "draft" | "ready">) {
    const updated = await supabase.from("exam_sessions").update({ status }).eq("id", id);
    fail(updated.error);
  },

  async remove(id: string): Promise<void> {
    const deleted = await supabase.from("exam_sessions").delete().eq("id", id);
    fail(deleted.error);
  },
};

export const rosterService = {
  async list(examSessionId: string): Promise<RosterStudent[]> {
    const response = await supabase
      .from("exam_roster_students")
      .select("*")
      .eq("exam_session_id", examSessionId)
      .order("university_id", { ascending: true });
    fail(response.error);
    return (response.data ?? []).map(toRosterStudent);
  },

  async add(examSessionId: string, input: RosterStudentInput): Promise<void> {
    const inserted = await supabase.from("exam_roster_students").insert({
      exam_session_id: examSessionId,
      university_id: input.universityId,
      full_name: input.fullName,
    });
    if (inserted.error) {
      if (inserted.error.code === "23505") {
        throw new Error("That university ID already exists in this exam session.");
      }
      throw new Error(inserted.error.message);
    }
  },

  async update(id: string, input: RosterStudentInput): Promise<void> {
    const updated = await supabase
      .from("exam_roster_students")
      .update({ university_id: input.universityId, full_name: input.fullName })
      .eq("id", id);
    if (updated.error) {
      if (updated.error.code === "23505") {
        throw new Error("That university ID already exists in this exam session.");
      }
      throw new Error(updated.error.message);
    }
  },

  async remove(id: string): Promise<void> {
    const deleted = await supabase.from("exam_roster_students").delete().eq("id", id);
    fail(deleted.error);
  },

  /**
   * Inserts pre-validated rows in one statement. Existing students are never
   * overwritten: a conflict fails the whole insert and is reported truthfully.
   */
  async importMany(examSessionId: string, rows: readonly RosterStudentInput[]): Promise<number> {
    if (rows.length === 0) return 0;
    const inserted = await supabase
      .from("exam_roster_students")
      .insert(
        rows.map((row) => ({
          exam_session_id: examSessionId,
          university_id: row.universityId,
          full_name: row.fullName,
        })),
      )
      .select("id");
    if (inserted.error) {
      if (inserted.error.code === "23505") {
        throw new Error(
          "Import cancelled: one or more university IDs already exist in this exam session. No students were added.",
        );
      }
      throw new Error(inserted.error.message);
    }
    return (inserted.data ?? []).length;
  },
};

/**
 * Read-only access to anonymous monitored subjects. Only the local AI service
 * writes this data; the console never invents or edits a subject.
 */
export const sessionSubjectsService = {
  async list(examSessionId: string): Promise<SessionSubject[]> {
    const response = await supabase
      .from("session_subjects")
      .select("*")
      .eq("exam_session_id", examSessionId)
      .order("subject_number", { ascending: true });
    fail(response.error);
    return (response.data ?? []).map((row) => ({
      id: row.id,
      examSessionId: row.exam_session_id,
      subjectNumber: row.subject_number,
      label: row.subject_label,
      cameraId: row.camera_id,
      lifecycle: (row.lifecycle_status ?? "active") as SubjectLifecycle,
      association: (row.track_association ?? "unresolved") as SubjectTrackAssociation,
      firstSeenAt: row.first_seen_at,
      lastSeenAt: row.last_seen_at,
      endedAt: row.ended_at,
      recoveryCount: row.reassociation_count ?? 0,
      lastAssociationConfidence:
        row.last_association_confidence === null ? null : Number(row.last_association_confidence),
    }));
  },
};
