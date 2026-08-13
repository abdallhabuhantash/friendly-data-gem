import { useMemo, useState, type ChangeEvent } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ROSTER_ISSUE_LABELS,
  buildRosterImportPlan,
  detectColumnMapping,
  parseSpreadsheetFile,
  type ColumnMapping,
  type SheetTable,
} from "@/lib/roster-import";
import type { RosterStudentInput } from "@/types";

const UNMAPPED = "unmapped";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  existingUniversityIds: readonly string[];
  pending: boolean;
  onConfirm: (rows: RosterStudentInput[]) => void;
}

export function RosterImportDialog({
  open,
  onOpenChange,
  existingUniversityIds,
  pending,
  onConfirm,
}: Props) {
  const [table, setTable] = useState<SheetTable | null>(null);
  const [fileName, setFileName] = useState("");
  const [mapping, setMapping] = useState<ColumnMapping>({
    universityIdColumn: null,
    fullNameColumn: null,
  });
  const [parseError, setParseError] = useState<string | null>(null);
  const [parsing, setParsing] = useState(false);

  const reset = () => {
    setTable(null);
    setFileName("");
    setMapping({ universityIdColumn: null, fullNameColumn: null });
    setParseError(null);
  };

  const pickFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setParsing(true);
    setParseError(null);
    try {
      const parsed = await parseSpreadsheetFile(file);
      if (parsed.headers.length === 0 || parsed.rows.length === 0) {
        setTable(null);
        setParseError("No data rows were found in this file. Nothing was imported.");
        return;
      }
      setTable(parsed);
      setFileName(file.name);
      setMapping(detectColumnMapping(parsed.headers));
    } catch {
      setTable(null);
      setParseError("This file could not be read as a CSV or XLSX spreadsheet.");
    } finally {
      setParsing(false);
    }
  };

  const plan = useMemo(
    () => (table ? buildRosterImportPlan(table, mapping, existingUniversityIds) : null),
    [table, mapping, existingUniversityIds],
  );

  const mappingComplete = mapping.universityIdColumn !== null && mapping.fullNameColumn !== null;
  const canImport = Boolean(plan) && mappingComplete && (plan?.counts.valid ?? 0) > 0 && !pending;

  const columnSelect = (
    label: string,
    value: number | null,
    onChange: (next: number | null) => void,
  ) => (
    <label className="block">
      <span className="label-tech">{label}</span>
      <Select
        value={value === null ? UNMAPPED : String(value)}
        onValueChange={(next) => onChange(next === UNMAPPED ? null : Number(next))}
      >
        <SelectTrigger className="mt-1">
          <SelectValue placeholder="Select column" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={UNMAPPED}>Not mapped</SelectItem>
          {(table?.headers ?? []).map((header, index) => (
            <SelectItem key={`${header}-${index}`} value={String(index)}>
              {header === "" ? `Column ${index + 1}` : header}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  );

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) reset();
        onOpenChange(next);
      }}
    >
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Import roster from spreadsheet</DialogTitle>
          <DialogDescription>
            CSV or XLSX. The file is parsed in your browser, every row is validated before import,
            and existing students are never overwritten.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3">
          <input
            type="file"
            accept=".csv,.xlsx,.xls,text/csv"
            onChange={pickFile}
            className="block w-full rounded-[3px] border border-border bg-surface-2 px-2 py-1.5 text-[13px]"
          />
          {parsing && <p className="text-xs text-muted-foreground">Reading file…</p>}
          {parseError && <p className="text-xs text-destructive">{parseError}</p>}

          {table && (
            <>
              <p className="font-mono text-[11px] text-muted-foreground">
                {fileName} — sheet “{table.sheetName}”, {table.rows.length} data rows
              </p>
              <div className="grid gap-3 sm:grid-cols-2">
                {columnSelect("Student ID column", mapping.universityIdColumn, (next) =>
                  setMapping((current) => ({ ...current, universityIdColumn: next })),
                )}
                {columnSelect("Full name column", mapping.fullNameColumn, (next) =>
                  setMapping((current) => ({ ...current, fullNameColumn: next })),
                )}
              </div>
              {!mappingComplete && (
                <p className="text-xs text-warning-foreground">
                  Map both the student ID and the full name column to continue.
                </p>
              )}

              {plan && mappingComplete && (
                <>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                    <Count label="Valid" value={plan.counts.valid} />
                    <Count label="Missing ID" value={plan.counts.missingUniversityId} />
                    <Count label="Missing name" value={plan.counts.missingFullName} />
                    <Count label="Dupes in file" value={plan.counts.duplicateInFile} />
                    <Count label="Already added" value={plan.counts.alreadyInSession} />
                  </div>
                  <div className="max-h-64 overflow-auto rounded-[3px] border border-border/70">
                    <table className="w-full text-left text-[12px]">
                      <thead className="sticky top-0 bg-surface-2">
                        <tr className="border-b border-border/70">
                          <th className="label-tech px-2 py-1.5">Row</th>
                          <th className="label-tech px-2 py-1.5">Student ID</th>
                          <th className="label-tech px-2 py-1.5">Full name</th>
                          <th className="label-tech px-2 py-1.5">Result</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/50">
                        {plan.rows.map((row) => (
                          <tr key={row.rowNumber}>
                            <td className="px-2 py-1 font-mono text-[11px] text-muted-foreground">
                              {row.rowNumber}
                            </td>
                            <td className="px-2 py-1 font-mono text-[11px]">
                              {row.universityId || "—"}
                            </td>
                            <td className="px-2 py-1">{row.fullName || "—"}</td>
                            <td className="px-2 py-1">
                              {row.issues.length === 0 ? (
                                <span className="text-success-foreground">Will be imported</span>
                              ) : (
                                <span className="text-destructive">
                                  {row.issues
                                    .map((issue) => ROSTER_ISSUE_LABELS[issue])
                                    .join(" · ")}
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[11px] text-muted-foreground">
                    Only the {plan.counts.valid} valid row(s) are imported. Rows with issues are
                    skipped and never modified in the database.
                  </p>
                </>
              )}
            </>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => {
              reset();
              onOpenChange(false);
            }}
            disabled={pending}
          >
            Cancel
          </Button>
          <Button
            disabled={!canImport}
            onClick={() => {
              if (plan) onConfirm(plan.valid);
            }}
          >
            {pending ? "Importing…" : `Import ${plan?.counts.valid ?? 0} student(s)`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function Count({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-[3px] border border-border/70 bg-surface-2 px-2 py-1.5">
      <p className="label-tech">{label}</p>
      <p className="font-mono text-[15px] text-foreground">{value}</p>
    </div>
  );
}
