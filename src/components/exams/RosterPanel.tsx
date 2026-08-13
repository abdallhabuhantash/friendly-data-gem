import { useMemo, useState } from "react";
import { Pencil, Trash2, Upload, UserPlus } from "lucide-react";
import { toast } from "sonner";
import { Panel } from "@/components/common/Panel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RosterImportDialog } from "@/components/exams/RosterImportDialog";
import { RosterStudentDialog } from "@/components/exams/RosterStudentDialog";
import {
  useAddRosterStudent,
  useImportRoster,
  useRemoveRosterStudent,
  useRoster,
  useUpdateRosterStudent,
} from "@/hooks/use-exams";
import type { RosterStudent, RosterStudentInput } from "@/types";

interface Props {
  examSessionId: string;
  canEdit: boolean;
}

export function RosterPanel({ examSessionId, canEdit }: Props) {
  const roster = useRoster(examSessionId);
  const [search, setSearch] = useState("");
  const [studentDialog, setStudentDialog] = useState<{ open: boolean; student: RosterStudent | null }>(
    { open: false, student: null },
  );
  const [importOpen, setImportOpen] = useState(false);

  const add = useAddRosterStudent(examSessionId);
  const update = useUpdateRosterStudent(examSessionId);
  const remove = useRemoveRosterStudent(examSessionId);
  const importRoster = useImportRoster(examSessionId);

  const students = roster.data ?? [];
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (query === "") return students;
    return students.filter(
      (student) =>
        student.universityId.toLowerCase().includes(query) ||
        student.fullName.toLowerCase().includes(query),
    );
  }, [students, search]);

  const existingIds = useMemo(
    () =>
      students
        .filter((student) => student.id !== studentDialog.student?.id)
        .map((student) => student.universityId),
    [students, studentDialog.student?.id],
  );

  const submitStudent = async (input: RosterStudentInput) => {
    try {
      if (studentDialog.student) {
        await update.mutateAsync({ id: studentDialog.student.id, input });
        toast.success("Student updated");
      } else {
        await add.mutateAsync(input);
        toast.success("Student added");
      }
      setStudentDialog({ open: false, student: null });
    } catch (caught) {
      toast.error(caught instanceof Error ? caught.message : "Could not save the student.");
    }
  };

  const confirmImport = async (rows: RosterStudentInput[]) => {
    try {
      const inserted = await importRoster.mutateAsync(rows);
      toast.success(`Imported ${inserted} student(s)`);
      setImportOpen(false);
    } catch (caught) {
      toast.error(caught instanceof Error ? caught.message : "The import failed.");
    }
  };

  return (
    <>
      <Panel
        title="Roster"
        subtitle={
          roster.isLoading
            ? "Loading roster…"
            : `${students.length} student record(s) in this exam session`
        }
        bodyClassName="p-0"
        actions={
          canEdit ? (
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => setImportOpen(true)}
                disabled={importRoster.isPending}
              >
                <Upload className="mr-1 size-3.5" /> Import
              </Button>
              <Button size="sm" onClick={() => setStudentDialog({ open: true, student: null })}>
                <UserPlus className="mr-1 size-3.5" /> Add student
              </Button>
            </div>
          ) : null
        }
      >
        <div className="border-b border-border/70 p-3">
          <Input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search by student ID or name"
            className="max-w-sm"
          />
        </div>

        {roster.isLoading && <p className="p-3 text-xs text-muted-foreground">Loading roster…</p>}
        {roster.isError && (
          <p className="p-3 text-xs text-destructive">
            The roster could not be loaded. {(roster.error as Error).message}
          </p>
        )}
        {!roster.isLoading && !roster.isError && students.length === 0 && (
          <p className="p-3 text-xs text-muted-foreground">
            No students yet. Add students manually or import a CSV/XLSX roster.
          </p>
        )}
        {!roster.isLoading && students.length > 0 && filtered.length === 0 && (
          <p className="p-3 text-xs text-muted-foreground">No student matches this search.</p>
        )}

        {filtered.length > 0 && (
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="border-b border-border/70">
                <th className="label-tech px-3 py-2">University ID</th>
                <th className="label-tech px-3 py-2">Full name</th>
                {canEdit && <th className="label-tech px-3 py-2 text-right">Actions</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {filtered.map((student) => (
                <tr key={student.id} className="hover:bg-surface-2/60">
                  <td className="px-3 py-2 font-mono text-[12px] text-muted-foreground">
                    {student.universityId}
                  </td>
                  <td className="px-3 py-2 text-foreground">{student.fullName}</td>
                  {canEdit && (
                    <td className="px-3 py-2 text-right">
                      <div className="flex justify-end gap-1">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setStudentDialog({ open: true, student })}
                        >
                          <Pencil className="size-3.5" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={remove.isPending}
                          onClick={async () => {
                            try {
                              await remove.mutateAsync(student.id);
                              toast.success("Student removed");
                            } catch (caught) {
                              toast.error(
                                caught instanceof Error
                                  ? caught.message
                                  : "Could not remove the student.",
                              );
                            }
                          }}
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      </div>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>

      <RosterStudentDialog
        open={studentDialog.open}
        onOpenChange={(open) => setStudentDialog((current) => ({ ...current, open }))}
        student={studentDialog.student}
        existingUniversityIds={existingIds}
        pending={add.isPending || update.isPending}
        onSubmit={submitStudent}
      />
      <RosterImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        existingUniversityIds={students.map((student) => student.universityId)}
        pending={importRoster.isPending}
        onConfirm={confirmImport}
      />
    </>
  );
}
