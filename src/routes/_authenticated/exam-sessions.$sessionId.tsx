import { Link, createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { ArrowLeft, Pencil } from "lucide-react";
import { toast } from "sonner";
import { Panel } from "@/components/common/Panel";
import { PageContainer } from "@/components/layout/PageContainer";
import { TopBar } from "@/components/layout/TopBar";
import { Button } from "@/components/ui/button";
import { ExamSessionFormDialog } from "@/components/exams/ExamSessionFormDialog";
import { RosterPanel } from "@/components/exams/RosterPanel";
import { useAuth } from "@/hooks/use-auth";
import { useCameras } from "@/hooks/use-monitoring";
import { useExamSession, useSetExamConfiguredStatus, useUpdateExamSession } from "@/hooks/use-exams";
import { EXAM_STATUS_LABELS } from "@/lib/exam-validation";
import type { ExamSession, ExamSessionInput } from "@/types";

export const Route = createFileRoute("/_authenticated/exam-sessions/$sessionId")({
  head: () => ({
    meta: [
      { title: "Exam Session — Vigilant Eye AI Smart Surveillance" },
      {
        name: "description",
        content: "Exam session overview and student roster management for anonymous monitoring.",
      },
      { property: "og:title", content: "Exam Session — Vigilant Eye" },
      {
        property: "og:description",
        content: "Configured exam session details and roster records.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: ExamSessionDetailPage,
});

function ExamSessionDetailPage() {
  const { sessionId } = Route.useParams();
  const session = useExamSession(sessionId);
  const cameras = useCameras("all");
  const { isAdministrator } = useAuth();
  const update = useUpdateExamSession(sessionId);
  const setStatus = useSetExamConfiguredStatus(sessionId);
  const [editing, setEditing] = useState(false);

  const submit = async (input: ExamSessionInput) => {
    try {
      await update.mutateAsync(input);
      toast.success("Exam session updated");
      setEditing(false);
    } catch (caught) {
      toast.error(caught instanceof Error ? caught.message : "Could not save the exam session.");
    }
  };

  const data = session.data ?? null;

  return (
    <>
      <TopBar title={data?.title ?? "Exam session"} subtitle="Overview and roster" />
      <PageContainer>
        <div>
          <Link
            to="/exam-sessions"
            className="inline-flex items-center gap-1 font-mono text-[11px] text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="size-3" /> All exam sessions
          </Link>
        </div>

        {session.isLoading && (
          <p className="text-xs text-muted-foreground">Loading exam session…</p>
        )}
        {session.isError && (
          <p className="text-xs text-destructive">
            The exam session could not be loaded. {(session.error as Error).message}
          </p>
        )}
        {!session.isLoading && !session.isError && data === null && (
          <p className="text-xs text-muted-foreground">This exam session does not exist.</p>
        )}

        {data && (
          <>
            <Panel
              title="Overview"
              subtitle="Configured information only. Monitoring has not been started."
              actions={
                isAdministrator ? (
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" onClick={() => setEditing(true)}>
                      <Pencil className="mr-1 size-3.5" /> Edit
                    </Button>
                    {data.status === "draft" && (
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={setStatus.isPending}
                        onClick={async () => {
                          try {
                            await setStatus.mutateAsync("ready");
                            toast.success("Session marked as configured (Ready)");
                          } catch (caught) {
                            toast.error(
                              caught instanceof Error
                                ? caught.message
                                : "Could not update the status.",
                            );
                          }
                        }}
                      >
                        Mark configured
                      </Button>
                    )}
                    {data.status === "ready" && (
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={setStatus.isPending}
                        onClick={async () => {
                          try {
                            await setStatus.mutateAsync("draft");
                            toast.success("Session returned to draft");
                          } catch (caught) {
                            toast.error(
                              caught instanceof Error
                                ? caught.message
                                : "Could not update the status.",
                            );
                          }
                        }}
                      >
                        Return to draft
                      </Button>
                    )}
                  </div>
                ) : null
              }
            >
              <Overview session={data} cameras={cameras.data ?? []} />
            </Panel>

            <RosterPanel examSessionId={data.id} canEdit={isAdministrator} />

            <ExamSessionFormDialog
              open={editing}
              onOpenChange={setEditing}
              cameras={cameras.data ?? []}
              session={data}
              pending={update.isPending}
              onSubmit={submit}
            />
          </>
        )}
      </PageContainer>
    </>
  );
}

function Overview({
  session,
  cameras,
}: {
  session: ExamSession;
  cameras: { id: string; name: string; location: string }[];
}) {
  const camera = cameras.find((entry) => entry.id === session.primaryCameraId) ?? null;
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <Fact label="Status" value={EXAM_STATUS_LABELS[session.status] ?? session.status} />
      <Fact label="Course code" value={session.courseCode === "" ? "—" : session.courseCode} />
      <Fact
        label="Hall / location"
        value={session.locationLabel === "" ? "—" : session.locationLabel}
      />
      <Fact
        label="Scheduled start"
        value={session.scheduledAt ? new Date(session.scheduledAt).toLocaleString() : "Not scheduled"}
      />
      <Fact
        label="Primary camera"
        value={camera ? `${camera.name}${camera.location ? ` — ${camera.location}` : ""}` : "None"}
      />
      <Fact label="Roster students" value={String(session.rosterCount)} />
      <Fact
        label="Invigilators"
        value={
          session.invigilators.length === 0
            ? "None recorded"
            : session.invigilators.map((person) => person.fullName).join(", ")
        }
      />
      <Fact
        label="Started at"
        value={session.startedAt ? new Date(session.startedAt).toLocaleString() : "Not started"}
      />
      <Fact
        label="Ended at"
        value={session.endedAt ? new Date(session.endedAt).toLocaleString() : "—"}
      />
      <p className="sm:col-span-2 lg:col-span-3 text-[11px] text-muted-foreground">
        Subject identity (S001, S002, …) and monitoring arming are not part of this configuration
        step. Invigilator names are metadata; the system performs no facial or biometric
        recognition.
      </p>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[3px] border border-border/70 bg-surface-2 px-2.5 py-2">
      <p className="label-tech">{label}</p>
      <p className="mt-0.5 text-[13px] text-foreground">{value}</p>
    </div>
  );
}
