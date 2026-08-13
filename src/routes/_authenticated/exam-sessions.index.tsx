import { Link, createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { CalendarClock, Cctv, MapPin, Plus, Users } from "lucide-react";
import { toast } from "sonner";
import { Panel } from "@/components/common/Panel";
import { PageContainer } from "@/components/layout/PageContainer";
import { TopBar } from "@/components/layout/TopBar";
import { Button } from "@/components/ui/button";
import { ExamSessionFormDialog } from "@/components/exams/ExamSessionFormDialog";
import { useAuth } from "@/hooks/use-auth";
import { useCameras } from "@/hooks/use-monitoring";
import { useCreateExamSession, useExamSessions } from "@/hooks/use-exams";
import { EXAM_STATUS_LABELS } from "@/lib/exam-validation";
import type { ExamSession, ExamSessionInput } from "@/types";

export const Route = createFileRoute("/_authenticated/exam-sessions/")({
  head: () => ({
    meta: [
      { title: "Exam Sessions — Vigilant Eye AI Smart Surveillance" },
      {
        name: "description",
        content:
          "Configure exam sessions, invigilators and student rosters for anonymous exam monitoring.",
      },
      { property: "og:title", content: "Exam Sessions — Vigilant Eye" },
      {
        property: "og:description",
        content: "Exam session configuration and student roster management.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: ExamSessionsPage,
});

function ExamSessionsPage() {
  const sessions = useExamSessions();
  const cameras = useCameras("all");
  const { isAdministrator } = useAuth();
  const create = useCreateExamSession();
  const [open, setOpen] = useState(false);

  const submit = async (input: ExamSessionInput) => {
    try {
      await create.mutateAsync(input);
      toast.success("Exam session created");
      setOpen(false);
    } catch (caught) {
      toast.error(caught instanceof Error ? caught.message : "Could not create the exam session.");
    }
  };

  return (
    <>
      <TopBar title="Exam Sessions" subtitle="Session configuration and rosters" />
      <PageContainer>
        <Panel
          title="Exam sessions"
          subtitle="Students are roster records only — never application users, seats or recognised faces."
          bodyClassName="p-0"
          actions={
            isAdministrator ? (
              <Button size="sm" onClick={() => setOpen(true)}>
                <Plus className="mr-1 size-3.5" /> New session
              </Button>
            ) : null
          }
        >
          {sessions.isLoading && (
            <p className="p-3 text-xs text-muted-foreground">Loading exam sessions…</p>
          )}
          {sessions.isError && (
            <p className="p-3 text-xs text-destructive">
              Exam sessions could not be loaded. {(sessions.error as Error).message}
            </p>
          )}
          {!sessions.isLoading && !sessions.isError && (sessions.data ?? []).length === 0 && (
            <p className="p-3 text-xs text-muted-foreground">
              No exam sessions are configured yet.
              {isAdministrator ? " Create one to begin." : ""}
            </p>
          )}
          {(sessions.data ?? []).length > 0 && (
            <ul className="divide-y divide-border/50">
              {(sessions.data ?? []).map((session) => (
                <li key={session.id}>
                  <Link
                    to="/exam-sessions/$sessionId"
                    params={{ sessionId: session.id }}
                    className="block px-3 py-2.5 hover:bg-surface-2/60"
                  >
                    <SessionRow session={session} cameraName={cameraLabel(session, cameras.data)} />
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </PageContainer>

      <ExamSessionFormDialog
        open={open}
        onOpenChange={setOpen}
        cameras={cameras.data ?? []}
        pending={create.isPending}
        onSubmit={submit}
      />
    </>
  );
}

function cameraLabel(session: ExamSession, cameras: { id: string; name: string }[] | undefined) {
  if (!session.primaryCameraId) return null;
  return cameras?.find((camera) => camera.id === session.primaryCameraId)?.name ?? "Linked camera";
}

function SessionRow({ session, cameraName }: { session: ExamSession; cameraName: string | null }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div className="min-w-0">
        <p className="truncate text-[13px] text-foreground">
          {session.title}
          {session.courseCode !== "" && (
            <span className="ml-2 font-mono text-[11px] text-muted-foreground">
              {session.courseCode}
            </span>
          )}
        </p>
        <div className="mt-1 flex flex-wrap items-center gap-3 font-mono text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <CalendarClock className="size-3" />
            {session.scheduledAt ? new Date(session.scheduledAt).toLocaleString() : "Not scheduled"}
          </span>
          <span className="inline-flex items-center gap-1">
            <MapPin className="size-3" />
            {session.locationLabel === "" ? "No location" : session.locationLabel}
          </span>
          <span className="inline-flex items-center gap-1">
            <Cctv className="size-3" />
            {cameraName ?? "No camera"}
          </span>
          <span className="inline-flex items-center gap-1">
            <Users className="size-3" />
            {session.rosterCount} student(s)
          </span>
        </div>
      </div>
      <span className="rounded-[3px] border border-primary/40 bg-primary/10 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] text-primary">
        {EXAM_STATUS_LABELS[session.status] ?? session.status}
      </span>
    </div>
  );
}
