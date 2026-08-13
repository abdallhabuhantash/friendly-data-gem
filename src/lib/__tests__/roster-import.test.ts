import { describe, expect, it } from "vitest";
import * as XLSX from "xlsx";
import {
  buildRosterImportPlan,
  detectColumnMapping,
  parseSheetBuffer,
  type SheetTable,
} from "@/lib/roster-import";

const sheetBuffer = (rows: unknown[][], bookType: "csv" | "xlsx" = "xlsx"): Uint8Array => {
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet(rows), "Roster");
  return new Uint8Array(XLSX.write(workbook, { type: "array", bookType }));
};

const table = (headers: string[], rows: string[][]): SheetTable => ({
  headers,
  rows,
  sheetName: "Roster",
});

describe("parseSheetBuffer", () => {
  it("reads an XLSX roster into headers and rows", () => {
    const parsed = parseSheetBuffer(
      sheetBuffer([
        ["University ID", "Full Name"],
        ["20211234", "Sara Khalid"],
        ["20215678", "Omar Nasser"],
      ]),
    );
    expect(parsed.headers).toEqual(["University ID", "Full Name"]);
    expect(parsed.rows).toEqual([
      ["20211234", "Sara Khalid"],
      ["20215678", "Omar Nasser"],
    ]);
  });

  it("reads a CSV roster and drops blank rows", () => {
    const parsed = parseSheetBuffer(
      sheetBuffer(
        [["Student ID", "Name"], ["1", "A"], ["", ""], ["2", "B"]],
        "csv",
      ),
    );
    expect(parsed.rows).toEqual([
      ["1", "A"],
      ["2", "B"],
    ]);
  });

  it("returns an empty table for an empty sheet instead of inventing rows", () => {
    const parsed = parseSheetBuffer(sheetBuffer([]));
    expect(parsed.headers).toEqual([]);
    expect(parsed.rows).toEqual([]);
  });
});

describe("detectColumnMapping", () => {
  it("detects common header spellings", () => {
    expect(detectColumnMapping(["Full Name", "University ID"])).toEqual({
      universityIdColumn: 1,
      fullNameColumn: 0,
    });
    expect(detectColumnMapping(["student_id", "student name"])).toEqual({
      universityIdColumn: 0,
      fullNameColumn: 1,
    });
  });

  it("leaves unknown headers unmapped rather than guessing", () => {
    expect(detectColumnMapping(["col a", "col b"])).toEqual({
      universityIdColumn: null,
      fullNameColumn: null,
    });
  });

  it("never maps the same column to both fields", () => {
    const mapping = detectColumnMapping(["student id"]);
    expect(mapping.universityIdColumn).toBe(0);
    expect(mapping.fullNameColumn).toBeNull();
  });
});

describe("buildRosterImportPlan", () => {
  const mapping = { universityIdColumn: 0, fullNameColumn: 1 };

  it("accepts clean rows", () => {
    const plan = buildRosterImportPlan(
      table(["id", "name"], [["1", "A"], ["2", "B"]]),
      mapping,
      [],
    );
    expect(plan.counts).toMatchObject({ total: 2, valid: 2 });
    expect(plan.valid).toEqual([
      { universityId: "1", fullName: "A" },
      { universityId: "2", fullName: "B" },
    ]);
  });

  it("rejects malformed rows with missing id or name", () => {
    const plan = buildRosterImportPlan(
      table(["id", "name"], [["", "A"], ["2", ""], ["3", "C"]]),
      mapping,
      [],
    );
    expect(plan.counts).toMatchObject({
      total: 3,
      valid: 1,
      missingUniversityId: 1,
      missingFullName: 1,
    });
    expect(plan.valid).toEqual([{ universityId: "3", fullName: "C" }]);
    expect(plan.rows[0]?.issues).toContain("missing_university_id");
    expect(plan.rows[1]?.issues).toContain("missing_full_name");
  });

  it("flags duplicate ids inside the file and keeps the first occurrence", () => {
    const plan = buildRosterImportPlan(
      table(["id", "name"], [["1", "A"], ["1", "Duplicate"]]),
      mapping,
      [],
    );
    expect(plan.counts.duplicateInFile).toBe(1);
    expect(plan.valid).toEqual([{ universityId: "1", fullName: "A" }]);
  });

  it("flags ids already present in the exam session", () => {
    const plan = buildRosterImportPlan(
      table(["id", "name"], [["1", "A"], ["2", "B"]]),
      mapping,
      ["1"],
    );
    expect(plan.counts.alreadyInSession).toBe(1);
    expect(plan.valid).toEqual([{ universityId: "2", fullName: "B" }]);
  });

  it("treats existing-id comparison case-insensitively", () => {
    const plan = buildRosterImportPlan(table(["id", "name"], [["ab12", "A"]]), mapping, ["AB12"]);
    expect(plan.counts.valid).toBe(0);
    expect(plan.rows[0]?.issues).toContain("already_in_session");
  });

  it("produces nothing when the mapping is incomplete", () => {
    const plan = buildRosterImportPlan(
      table(["id", "name"], [["1", "A"]]),
      { universityIdColumn: 0, fullNameColumn: null },
      [],
    );
    expect(plan.valid).toEqual([]);
    expect(plan.rows[0]?.issues).toContain("missing_full_name");
  });

  it("never fabricates rows for an empty file", () => {
    const plan = buildRosterImportPlan(table(["id", "name"], []), mapping, []);
    expect(plan.rows).toEqual([]);
    expect(plan.valid).toEqual([]);
    expect(plan.counts.total).toBe(0);
  });
});
