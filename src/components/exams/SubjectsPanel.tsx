import { Panel } from "@/components/common/Panel";
import { useSessionSubjects } from "@/hooks/use-exams";
import type { ExamSession, SessionSubject, SubjectTrackingStatus } from "@/types";

const STATUS_LABEL: Record<SubjectTrackingStatus, string> = {
  stable: "Tracked",
  temporarily_lost: "Temporarily lost",
  uncertain: "Uncertain",
  conflict: "Conflicting evidence",
  ended: "Ended",
};

const STATUS_TONE: Record<SubjectTrackingStatus, string> = {
  stable: "border-success/40 text-success",
  temporarily_lost: "border-warning/40 text-warning",
  uncertain: "border-warning/40 text-warning",
  conflict: "border-destructive/40 text-destructive",
  ended: "border-border text-muted-foreground",
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
                <th className="label-tech py-1.5 pr-3">Tracking</th>
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
        Identity stays UNRESOLVED unless a reviewer resolves it manually. Ambiguous tracking is
        reported as “Uncertain” instead of being guessed.
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
          className={`inline-flex rounded-[3px] border px-1.5 py-0.5 font-mono text-[10px] uppercase ${STATUS_TONE[subject.trackingStatus]}`}
        >
          {STATUS_LABEL[subject.trackingStatus]}
        </span>
      </td>
      <td className="py-1.5 pr-3 text-muted-foreground">
        {new Date(subject.firstSeenAt).toLocaleTimeString()}
      </td>
      <td className="py-1.5 pr-3 text-muted-foreground">
        {new Date(subject.lastSeenAt).toLocaleTimeString()}
      </td>
      <td className="py-1.5 pr-3 text-muted-foreground">{subject.reassociationCount}</td>
      <td className="py-1.5 text-muted-foreground">
        {subject.lastAssociationConfidence === null
          ? "—"
          : `${Math.round(subject.lastAssociationConfidence * 100)}%`}
      </td>
    </tr>
  );
}
