import * as XLSX from "xlsx";
import type { RosterStudentInput } from "@/types";

/**
 * Pure spreadsheet roster import logic.
 *
 * Nothing here touches the database and nothing here invents rows: an
 * unreadable or ambiguous file produces explicit issues, never fabricated
 * students.
 */

export interface SheetTable {
  /** Header labels exactly as they appear in the file. */
  headers: string[];
  /** Data rows aligned to `headers`; missing cells are empty strings. */
  rows: string[][];
  sheetName: string;
}

export interface ColumnMapping {
  /** Index into `SheetTable.headers`, or null when unmapped. */
  universityIdColumn: number | null;
  fullNameColumn: number | null;
}

export type RosterRowIssue =
  "missing_university_id" | "missing_full_name" | "duplicate_in_file" | "already_in_session";

export interface RosterRowResult {
  /** 1-based data row number as the user sees it in the preview. */
  rowNumber: number;
  universityId: string;
  fullName: string;
  issues: RosterRowIssue[];
}

export interface RosterImportPlan {
  rows: RosterRowResult[];
  /** Rows safe to insert, in file order. */
  valid: RosterStudentInput[];
  counts: {
    total: number;
    valid: number;
    missingUniversityId: number;
    missingFullName: number;
    duplicateInFile: number;
    alreadyInSession: number;
  };
}

const cell = (value: unknown): string =>
  value === null || value === undefined ? "" : String(value).trim();

const ID_HEADERS = [
  "university id",
  "universityid",
  "university_id",
  "student id",
  "studentid",
  "student_id",
  "id",
  "number",
  "reg no",
  "registration",
  "matric",
  "الرقم الجامعي",
];

const NAME_HEADERS = [
  "full name",
  "fullname",
  "full_name",
  "name",
  "student name",
  "student_name",
  "الاسم",
];

/** Parses CSV or XLSX bytes into a header/rows table. Never throws on data. */
export function parseSheetBuffer(data: ArrayBuffer | Uint8Array): SheetTable {
  const workbook = XLSX.read(data, { type: "array" });
  const sheetName = workbook.SheetNames[0];
  if (!sheetName) return { headers: [], rows: [], sheetName: "" };
  const sheet = workbook.Sheets[sheetName];
  if (!sheet) return { headers: [], rows: [], sheetName };
  const matrix = XLSX.utils.sheet_to_json<unknown[]>(sheet, {
    header: 1,
    blankrows: false,
    defval: "",
    raw: false,
  });
  if (matrix.length === 0) return { headers: [], rows: [], sheetName };
  const headers = (matrix[0] ?? []).map(cell);
  const width = headers.length;
  const rows = matrix
    .slice(1)
    .map((row) => Array.from({ length: width }, (_, index) => cell(row?.[index])))
    .filter((row) => row.some((value) => value !== ""));
  return { headers, rows, sheetName };
}

export async function parseSpreadsheetFile(file: File): Promise<SheetTable> {
  const buffer = await file.arrayBuffer();
  return parseSheetBuffer(buffer);
}

/**
 * Best-effort header detection. The user always confirms or overrides the
 * mapping before anything is imported, so guessing here is safe.
 */
export function detectColumnMapping(headers: string[]): ColumnMapping {
  const normalized = headers.map((header) => header.toLowerCase().trim());
  const find = (candidates: string[]): number | null => {
    for (const candidate of candidates) {
      const exact = normalized.indexOf(candidate);
      if (exact >= 0) return exact;
    }
    for (const candidate of candidates) {
      const partial = normalized.findIndex((header) => header !== "" && header.includes(candidate));
      if (partial >= 0) return partial;
    }
    return null;
  };
  const universityIdColumn = find(ID_HEADERS);
  let fullNameColumn = find(NAME_HEADERS);
  if (fullNameColumn !== null && fullNameColumn === universityIdColumn) fullNameColumn = null;
  return { universityIdColumn, fullNameColumn };
}

/**
 * Validates every row before any database write. Rows with issues are
 * reported and excluded; they are never silently accepted or repaired.
 */
export function buildRosterImportPlan(
  table: SheetTable,
  mapping: ColumnMapping,
  existingUniversityIds: readonly string[],
): RosterImportPlan {
  const existing = new Set(existingUniversityIds.map((value) => value.trim().toLowerCase()));
  const seen = new Set<string>();
  const rows: RosterRowResult[] = [];

  table.rows.forEach((row, index) => {
    const universityId =
      mapping.universityIdColumn === null ? "" : (row[mapping.universityIdColumn] ?? "").trim();
    const fullName =
      mapping.fullNameColumn === null ? "" : (row[mapping.fullNameColumn] ?? "").trim();
    const issues: RosterRowIssue[] = [];
    if (universityId === "") issues.push("missing_university_id");
    if (fullName === "") issues.push("missing_full_name");
    const key = universityId.toLowerCase();
    if (universityId !== "") {
      if (seen.has(key)) issues.push("duplicate_in_file");
      else seen.add(key);
      if (existing.has(key)) issues.push("already_in_session");
    }
    rows.push({ rowNumber: index + 1, universityId, fullName, issues });
  });

  const counts = {
    total: rows.length,
    valid: rows.filter((row) => row.issues.length === 0).length,
    missingUniversityId: rows.filter((row) => row.issues.includes("missing_university_id")).length,
    missingFullName: rows.filter((row) => row.issues.includes("missing_full_name")).length,
    duplicateInFile: rows.filter((row) => row.issues.includes("duplicate_in_file")).length,
    alreadyInSession: rows.filter((row) => row.issues.includes("already_in_session")).length,
  };

  return {
    rows,
    valid: rows
      .filter((row) => row.issues.length === 0)
      .map((row) => ({ universityId: row.universityId, fullName: row.fullName })),
    counts,
  };
}

export const ROSTER_ISSUE_LABELS: Record<RosterRowIssue, string> = {
  missing_university_id: "Missing student ID",
  missing_full_name: "Missing full name",
  duplicate_in_file: "Duplicate ID inside this file",
  already_in_session: "Already in this exam session",
};
