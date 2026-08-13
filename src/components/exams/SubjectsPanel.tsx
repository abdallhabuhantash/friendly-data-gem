import { Panel } from "@/components/common/Panel";
import { useSessionSubjects } from "@/hooks/use-exams";
import type {
  ExamSession,
  SessionSubject,
  SubjectLifecycle,
  SubjectTrackAssociation,
} from "@/types";

const LIFECYCLE_LABEL: Record<SubjectLifecycle, string> = {
  active: "Active",
  temporarily_lost: "Temporarily lost",
  lost: "Lost",
  ended: "Ended",
};

const LIFECYCLE_TONE: Record<SubjectLifecycle, string> = {
  active: "border-success/40 text-success",
  temporarily_lost: "border-warning/40 text-warning",
  lost: "border-warning/40 text-warning",
  ended: "border-border text-muted-foreground",
};

const ASSOCIATION_LABEL: Record<SubjectTrackAssociation, string> = {
  confirmed: "Confirmed",
  provisional: "Provisional",
  unresolved: "Unresolved",
  conflict: "Conflicting evidence",
};

const ASSOCIATION_TONE: Record<SubjectTrackAssociation, string> = {
  confirmed: "border-success/40 text-success",
  provisional: "border-warning/40 text-warning",
  unresolved: "border-border text-muted-foreground",
  conflict: "border-destructive/40 text-destructive",
};

/**
 * Anonymous monitored subjects (S001, S002, …) of one exam session.
 *
 * These labels are NOT identities: no name, no university ID, no face, no seat.
 * Resolving a subject to a roster student stays a manual, on-demand action that
 * this panel deliberately does not perform or suggest.
 */
export function SubjectsPanel({ session }: { session: ExamSession }) {
  const subjects = useSessionSubjects(session.id);
  const rows = subjects.data ?? [];
  const armed = session.status === "active";

  return (
    <Panel
      title="Monitored subjects"
      subtitle="Anonymous, per-session labels only. No name, face, biometric or seat is used."
    >
      {subjects.isError && (
        <p className="text-xs text-destructive">
          Subjects could not be loaded. {(subjects.error as Error).message}
        </p>
      )}

      {!armed && rows.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Monitoring has not been started for this session, so no subject exists yet.
        </p>
      )}

      {armed && rows.length === 0 && !subjects.isLoading && (
        <p className="text-xs text-muted-foreground">
          Monitoring is armed. No person has been observed long enough to earn a subject label yet.
        </p>
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-border/70 text-left">
                <th className="label-tech py-1.5 pr-3">Subject</th>
                <th className="label-tech py-1.5 pr-3">Subject state</th>
                <th className="label-tech py-1.5 pr-3">Track association</th>
                <th className="label-tech py-1.5 pr-3">First seen</th>
                <th className="label-tech py-1.5 pr-3">Last seen</th>
                <th className="label-tech py-1.5 pr-3">Track recoveries</th>
                <th className="label-tech py-1.5">Recovery confidence</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((subject) => (
                <SubjectRow key={subject.id} subject={subject} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="mt-2 text-[11px] text-muted-foreground">
        A subject label belongs to one person for the whole session: it is never renumbered, never
        transferred and never reused. A lost subject keeps its label reserved. Ambiguous tracking is
        reported as “Unresolved” or “Conflicting evidence” instead of being guessed.
      </p>
    </Panel>
  );
}

function SubjectRow({ subject }: { subject: SessionSubject }) {
  return (
    <tr className="border-b border-border/40 last:border-0">
      <td className="py-1.5 pr-3 font-mono text-[12px] text-foreground">{subject.label}</td>
      <td className="py-1.5 pr-3">
        <span
          className={`inline-flex rounded-[3px] border px-1.5 py-0.5 font-mono text-[10px] uppercase ${LIFECYCLE_TONE[subject.lifecycle]}`}
        >
          {LIFECYCLE_LABEL[subject.lifecycle]}
        </span>
      </td>
      <td className="py-1.5 pr-3">
        <span
          className={`inline-flex rounded-[3px] border px-1.5 py-0.5 font-mono text-[10px] uppercase ${ASSOCIATION_TONE[subject.association]}`}
        >
          {ASSOCIATION_LABEL[subject.association]}
        </span>
      </td>
      <td className="py-1.5 pr-3 text-muted-foreground">
        {new Date(subject.firstSeenAt).toLocaleTimeString()}
      </td>
      <td className="py-1.5 pr-3 text-muted-foreground">
        {new Date(subject.lastSeenAt).toLocaleTimeString()}
      </td>
      <td className="py-1.5 pr-3 text-muted-foreground">{subject.recoveryCount}</td>
      <td className="py-1.5 text-muted-foreground">
        {subject.lastAssociationConfidence === null
          ? "—"
          : `${Math.round(subject.lastAssociationConfidence * 100)}%`}
      </td>
    </tr>
  );
}
