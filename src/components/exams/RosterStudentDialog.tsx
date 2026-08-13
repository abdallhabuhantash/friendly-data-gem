import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { isDuplicateUniversityId, validateRosterStudent } from "@/lib/exam-validation";
import type { RosterStudent, RosterStudentInput } from "@/types";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  student?: RosterStudent | null;
  /** University IDs already in this session, excluding the edited student. */
  existingUniversityIds: readonly string[];
  pending: boolean;
  onSubmit: (input: RosterStudentInput) => void;
}

export function RosterStudentDialog({
  open,
  onOpenChange,
  student,
  existingUniversityIds,
  pending,
  onSubmit,
}: Props) {
  const [universityId, setUniversityId] = useState("");
  const [fullName, setFullName] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!open) return;
    setErrors({});
    setUniversityId(student?.universityId ?? "");
    setFullName(student?.fullName ?? "");
  }, [open, student]);

  const submit = () => {
    const result = validateRosterStudent({ universityId, fullName });
    if (!result.ok) {
      setErrors(result.errors);
      return;
    }
    if (isDuplicateUniversityId(result.value.universityId, existingUniversityIds)) {
      setErrors({ universityId: "This university ID is already in this exam session." });
      return;
    }
    setErrors({});
    onSubmit(result.value);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{student ? "Edit student" : "Add student"}</DialogTitle>
          <DialogDescription>
            Roster records are exam metadata. Students are not application users and are never
            identified visually.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-3">
          <label className="block">
            <span className="label-tech">University / student ID</span>
            <Input
              className="mt-1"
              value={universityId}
              onChange={(event) => setUniversityId(event.target.value)}
              placeholder="20211234"
            />
            {errors["universityId"] && (
              <p className="mt-1 text-[11px] text-destructive">{errors["universityId"]}</p>
            )}
          </label>
          <label className="block">
            <span className="label-tech">Full name</span>
            <Input
              className="mt-1"
              value={fullName}
              onChange={(event) => setFullName(event.target.value)}
              placeholder="Sara Khalid"
            />
            {errors["fullName"] && (
              <p className="mt-1 text-[11px] text-destructive">{errors["fullName"]}</p>
            )}
          </label>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={pending}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={pending}>
            {pending ? "Saving…" : student ? "Save changes" : "Add student"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
