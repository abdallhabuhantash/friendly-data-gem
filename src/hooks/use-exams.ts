import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { examSessionsService, rosterService } from "@/services/exam-service";
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
