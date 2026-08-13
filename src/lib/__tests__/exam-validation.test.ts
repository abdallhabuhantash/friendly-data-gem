import { describe, expect, it } from "vitest";
import {
  isDuplicateUniversityId,
  validateExamSession,
  validateRosterStudent,
} from "@/lib/exam-validation";

describe("validateExamSession", () => {
  const base = {
    title: "Data Structures — Midterm",
    courseCode: "CS201",
    locationLabel: "Hall B",
    scheduledAt: "2026-09-01T09:00:00.000Z",
    primaryCameraId: null,
    invigilatorNames: ["Dr. Ahmad", " "],
  };

  it("accepts a valid session and drops blank invigilator names", () => {
    const result = validateExamSession(base);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.invigilatorNames).toEqual(["Dr. Ahmad"]);
      expect(result.value.title).toBe("Data Structures — Midterm");
    }
  });

  it("requires a title", () => {
    const result = validateExamSession({ ...base, title: "   " });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors["title"]).toBeTruthy();
  });

  it("allows a session with no location, schedule or camera", () => {
    const result = validateExamSession({
      title: "Quiz",
      courseCode: "",
      locationLabel: "",
      scheduledAt: null,
      primaryCameraId: null,
      invigilatorNames: [],
    });
    expect(result.ok).toBe(true);
  });

  it("rejects an unparsable scheduled time", () => {
    const result = validateExamSession({ ...base, scheduledAt: "not-a-date" });
    expect(result.ok).toBe(false);
  });
});

describe("validateRosterStudent", () => {
  it("accepts a student with an id and a name", () => {
    const result = validateRosterStudent({ universityId: " 20211234 ", fullName: " Sara " });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toEqual({ universityId: "20211234", fullName: "Sara" });
    }
  });

  it("requires the university id", () => {
    const result = validateRosterStudent({ universityId: "", fullName: "Sara" });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors["universityId"]).toBeTruthy();
  });

  it("requires the full name", () => {
    const result = validateRosterStudent({ universityId: "1", fullName: "  " });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors["fullName"]).toBeTruthy();
  });
});

describe("isDuplicateUniversityId", () => {
  it("detects a duplicate inside the same session, ignoring case and padding", () => {
    expect(isDuplicateUniversityId(" ab12 ", ["AB12"])).toBe(true);
  });

  it("treats the same id in a different session roster as valid", () => {
    // A different exam session passes its own roster ids, which do not contain
    // this student, so the same university id is accepted there.
    expect(isDuplicateUniversityId("20211234", ["20219999"])).toBe(false);
    expect(isDuplicateUniversityId("20211234", [])).toBe(false);
  });
});
