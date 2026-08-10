import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

/** Which irreversible-feeling review decision is awaiting confirmation. */
export type ReviewDecision = "confirmed" | "rejected";

const COPY: Record<ReviewDecision, { title: string; question: string; action: string }> = {
  confirmed: {
    title: "Confirm Event",
    question: "Confirm this event?",
    action: "Confirm Event",
  },
  rejected: {
    title: "Reject Event",
    question: "Reject this event?",
    action: "Reject Event",
  },
};

/**
 * Lightweight confirmation for human review decisions. Wording stays advisory:
 * an event is confirmed or rejected, never "confirmed cheating".
 */
export function ReviewConfirmDialog({
  decision,
  onOpenChange,
  onConfirm,
}: {
  decision: ReviewDecision | null;
  onOpenChange: (open: boolean) => void;
  onConfirm: (decision: ReviewDecision) => void;
}) {
  const copy = decision ? COPY[decision] : null;
  return (
    <AlertDialog open={decision !== null} onOpenChange={onOpenChange}>
      <AlertDialogContent className="border-border bg-surface">
        <AlertDialogHeader>
          <AlertDialogTitle className="font-mono text-sm uppercase tracking-[0.1em]">
            {copy?.title ?? ""}
          </AlertDialogTitle>
          <AlertDialogDescription className="text-[11px]">
            {copy?.question ?? ""} This records a human review decision against the AI detection.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel className="h-8 text-[11px]">Cancel</AlertDialogCancel>
          <AlertDialogAction
            className="h-8 text-[11px]"
            onClick={() => decision && onConfirm(decision)}
          >
            {copy?.action ?? ""}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
