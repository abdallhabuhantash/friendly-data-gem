import { z } from "zod";
import type { ExamSessionInput, RosterStudentInput } from "@/types";

/** Pure validation for exam session and roster input. No database access. */

export const examSessionSchema = z.object({
  title: z.string().trim().min(1, { message: "Exam title is required" }).max(120),
  courseCode: z.string().trim().max(40).default(""),
  locationLabel: z.string().trim().max(120).default(""),
  scheduledAt: z
    .string()
    .trim()
    .max(40)
    .nullable()
    .default(null)
    .refine((value) => value === null || value === "" || !Number.isNaN(Date.parse(value)), {
      message: "Enter a valid date and time",
    }),
  primaryCameraId: z.string().uuid().nullable().default(null),
  invigilatorNames: z.array(z.string().trim().max(120)).default([]),
});

export const rosterStudentSchema = z.object({
  universityId: z
    .string()
    .trim()
    .min(1, { message: "University / student ID is required" })
    .max(64),
  fullName: z.string().trim().min(1, { message: "Full name is required" }).max(120),
});

type Result<T> = { ok: true; value: T } | { ok: false; errors: Record<string, string> };

const collect = (issues: z.ZodIssue[]): Record<string, string> => {
  const errors: Record<string, string> = {};
  issues.forEach((issue) => {
    const key = String(issue.path[0] ?? "form");
    if (!errors[key]) errors[key] = issue.message;
  });
  return errors;
};

export function validateExamSession(values: unknown): Result<ExamSessionInput> {
  const parsed = examSessionSchema.safeParse(values);
  if (!parsed.success) return { ok: false, errors: collect(parsed.error.issues) };
  const data = parsed.data;
  return {
    ok: true,
    value: {
      title: data.title,
      courseCode: data.courseCode,
      locationLabel: data.locationLabel,
      scheduledAt: data.scheduledAt === "" ? null : data.scheduledAt,
      primaryCameraId: data.primaryCameraId,
      invigilatorNames: data.invigilatorNames.filter((name) => name.trim() !== ""),
    },
  };
}

export function validateRosterStudent(values: unknown): Result<RosterStudentInput> {
  const parsed = rosterStudentSchema.safeParse(values);
  if (!parsed.success) return { ok: false, errors: collect(parsed.error.issues) };
  return { ok: true, value: parsed.data };
}

/** Case-insensitive duplicate check inside one exam session. */
export function isDuplicateUniversityId(
  universityId: string,
  existing: readonly string[],
): boolean {
  const key = universityId.trim().toLowerCase();
  return existing.some((value) => value.trim().toLowerCase() === key);
}

export const EXAM_STATUS_LABELS: Record<string, string> = {
  draft: "Draft",
  ready: "Ready",
  active: "Active",
  ended: "Ended",
  archived: "Archived",
};
