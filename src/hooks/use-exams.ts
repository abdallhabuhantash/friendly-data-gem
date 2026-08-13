import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import { useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import {
  examSessionsService,
  rosterService,
  sessionSubjectsService,
} from "@/services/exam-service";
import { endExamSession, startExamSession } from "@/lib/exam-runtime.functions";
import type { ExamSessionInput, ExamSessionStatus, RosterStudentInput } from "@/types";

export const useExamSessions = () =>
  useQuery({ queryKey: ["exam-sessions"], queryFn: examSessionsService.list });

export const useExamSession = (id: string) =>
  useQuery({
    queryKey: ["exam-sessions", id],
    queryFn: () => examSessionsService.byId(id),
    enabled: id !== "",
  });

export const useRoster = (examSessionId: string) =>
  useQuery({
    queryKey: ["exam-roster", examSessionId],
    queryFn: () => rosterService.list(examSessionId),
    enabled: examSessionId !== "",
  });

const useInvalidate = () => {
  const queryClient = useQueryClient();
  return (examSessionId?: string) => {
    void queryClient.invalidateQueries({ queryKey: ["exam-sessions"] });
    if (examSessionId)
      void queryClient.invalidateQueries({ queryKey: ["exam-roster", examSessionId] });
  };
};

export const useCreateExamSession = () => {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (input: ExamSessionInput) => examSessionsService.create(input),
    onSuccess: () => invalidate(),
  });
};

export const useUpdateExamSession = (id: string) => {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (input: ExamSessionInput) => examSessionsService.update(id, input),
    onSuccess: () => invalidate(),
  });
};

export const useSetExamConfiguredStatus = (id: string) => {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (status: Extract<ExamSessionStatus, "draft" | "ready">) =>
      examSessionsService.setConfiguredStatus(id, status),
    onSuccess: () => invalidate(),
  });
};

export const useAddRosterStudent = (examSessionId: string) => {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (input: RosterStudentInput) => rosterService.add(examSessionId, input),
    onSuccess: () => invalidate(examSessionId),
  });
};

export const useUpdateRosterStudent = (examSessionId: string) => {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: ({ id, input }: { id: string; input: RosterStudentInput }) =>
      rosterService.update(id, input),
    onSuccess: () => invalidate(examSessionId),
  });
};

export const useRemoveRosterStudent = (examSessionId: string) => {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (id: string) => rosterService.remove(id),
    onSuccess: () => invalidate(examSessionId),
  });
};

export const useImportRoster = (examSessionId: string) => {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (rows: RosterStudentInput[]) => rosterService.importMany(examSessionId, rows),
    onSuccess: () => invalidate(examSessionId),
  });
};

/**
 * Anonymous monitored subjects of one exam session, kept live by the same
 * realtime channel the AI service writes through. Read-only by design.
 */
export const useSessionSubjects = (examSessionId: string) => {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["session-subjects", examSessionId],
    queryFn: () => sessionSubjectsService.list(examSessionId),
    enabled: examSessionId !== "",
  });

  useEffect(() => {
    if (examSessionId === "") return;
    const channel = supabase
      .channel(`session-subjects-${examSessionId}`)
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: "session_subjects",
          filter: `exam_session_id=eq.${examSessionId}`,
        },
        () => {
          void queryClient.invalidateQueries({ queryKey: ["session-subjects", examSessionId] });
        },
      )
      .subscribe();
    return () => {
      void supabase.removeChannel(channel);
    };
  }, [examSessionId, queryClient]);

  return query;
};

/**
 * Start Exam Session: arms monitoring through the local AI service. The status
 * only becomes `active` because the AI service confirmed and wrote it.
 */
export const useStartExamSession = (examSessionId: string) => {
  const invalidate = useInvalidate();
  const start = useServerFn(startExamSession);
  return useMutation({
    mutationFn: () => start({ data: { examSessionId } }),
    onSuccess: () => invalidate(examSessionId),
  });
};

/** End Exam Session: disarms monitoring and closes every anonymous subject. */
export const useEndExamSession = (examSessionId: string) => {
  const invalidate = useInvalidate();
  const end = useServerFn(endExamSession);
  return useMutation({
    mutationFn: () => end({ data: { examSessionId } }),
    onSuccess: () => invalidate(examSessionId),
  });
};
