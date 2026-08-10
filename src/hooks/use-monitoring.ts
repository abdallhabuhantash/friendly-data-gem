import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type CameraScope,
  camerasService,
  eventsService,
  reportsService,
  rulesService,
  systemService,
  usersService,
} from "@/services/monitoring-service";
import type {
  AiRule,
  CameraConfigInput,
  EventStatus,
  OperationMode,
  SystemSettings,
} from "@/types";

const LIVE_REFRESH_MS = 15_000;
const HEARTBEAT_REFRESH_MS = 10_000;

/** Explicit demo/live switch, stored in system settings. */
export const useOperationMode = () =>
  useQuery<OperationMode>({
    queryKey: ["system", "mode"],
    queryFn: systemService.operationMode,
    refetchInterval: LIVE_REFRESH_MS,
  });

/**
 * Demo and live records never mix: every monitoring read is scoped to the
 * active operation mode, including the summary counters.
 */
const useScopedMode = () => {
  const mode = useOperationMode();
  return { mode: mode.data ?? "demo", ready: mode.data !== undefined } as const;
};

export const useCameras = (scope: CameraScope = "active") => {
  const { mode, ready } = useScopedMode();
  return useQuery({
    queryKey: ["cameras", mode, scope],
    queryFn: () => camerasService.list(mode, scope),
    enabled: ready,
    refetchInterval: HEARTBEAT_REFRESH_MS,
  });
};

export const useCameraSummary = () => {
  const { mode, ready } = useScopedMode();
  return useQuery({
    queryKey: ["cameras", "summary", mode],
    queryFn: () => camerasService.summary(mode),
    enabled: ready,
    refetchInterval: LIVE_REFRESH_MS,
  });
};

export const useEvents = () => {
  const { mode, ready } = useScopedMode();
  return useQuery({
    queryKey: ["events", mode],
    queryFn: () => eventsService.list(mode),
    enabled: ready,
    refetchInterval: LIVE_REFRESH_MS,
  });
};

export const useRecentEvents = (limit = 5) => {
  const { mode, ready } = useScopedMode();
  return useQuery({
    queryKey: ["events", "recent", limit, mode],
    queryFn: () => eventsService.recent(limit, mode),
    enabled: ready,
    refetchInterval: LIVE_REFRESH_MS,
  });
};

export const useEventsSummary = () => {
  const { mode, ready } = useScopedMode();
  return useQuery({
    queryKey: ["events", "summary", mode],
    queryFn: () => eventsService.summary(mode),
    enabled: ready,
    refetchInterval: LIVE_REFRESH_MS,
  });
};

export const useAiRules = () => useQuery({ queryKey: ["ai-rules"], queryFn: rulesService.list });

export const useUsers = () => useQuery({ queryKey: ["users"], queryFn: usersService.list });

export const useAiServiceStatus = () => {
  const mode = useOperationMode();
  return useQuery({
    queryKey: ["system", "ai", mode.data ?? "demo"],
    queryFn: () => systemService.aiStatus(mode.data ?? "demo"),
    enabled: mode.data !== undefined,
    refetchInterval: HEARTBEAT_REFRESH_MS,
  });
};

export const useNvrStatus = () => {
  const mode = useOperationMode();
  return useQuery({
    queryKey: ["system", "nvr", mode.data ?? "demo"],
    queryFn: () => systemService.nvrStatus(mode.data ?? "demo"),
    enabled: mode.data !== undefined,
    refetchInterval: HEARTBEAT_REFRESH_MS,
  });
};

export const useSystemSettings = () =>
  useQuery({ queryKey: ["system", "settings"], queryFn: systemService.settings });

/** Reports are cached per operation mode so demo data never leaks into live. */
export const useReportSummary = (range: "7d" | "30d") => {
  const { mode, ready } = useScopedMode();
  return useQuery({
    queryKey: ["reports", range, mode],
    queryFn: () => reportsService.summary(range, mode),
    enabled: ready,
  });
};

export function useReviewEvent() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: { id: string; status: EventStatus; note?: string | undefined }) =>
      eventsService.review(input.id, input.status, input.note),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["events"] });
      void queryClient.invalidateQueries({ queryKey: ["reports"] });
    },
  });
}

export function useUpdateRule() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: { id: string; patch: Partial<AiRule> }) =>
      rulesService.update(input.id, input.patch),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["ai-rules"] });
    },
  });
}

export function useToggleCameraFlag() {
  const queryClient = useQueryClient();
  return useMutation({
    // Recording state is reported by the NVR / AI service, never set from the browser.
    mutationFn: (input: { id: string; field: "aiEnabled"; value: boolean }) =>
      camerasService.toggleAi(input.id, input.value),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["cameras"] });
    },
  });
}

/** Assigns which cameras a detection rule applies to. */
export function useSetRuleCameras() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: { ruleId: string; cameraIds: string[] }) =>
      rulesService.setCameras(input.ruleId, input.cameraIds),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["ai-rules"] });
    },
  });
}

export function useUpdateSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (patch: Partial<SystemSettings>) => systemService.updateSettings(patch),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["system", "settings"] });
      void queryClient.invalidateQueries({ queryKey: ["system"] });
    },
  });
}

/** Camera configuration mutations. RLS restricts these to administrators. */
function useCameraMutation<TInput>(fn: (input: TInput) => Promise<void>) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["cameras"] });
      void queryClient.invalidateQueries({ queryKey: ["ai-rules"] });
    },
  });
}

export const useCreateCamera = () =>
  useCameraMutation((input: CameraConfigInput) => camerasService.create(input));

export const useUpdateCamera = () =>
  useCameraMutation((input: { id: string; config: CameraConfigInput }) =>
    camerasService.update(input.id, input.config),
  );

export const useArchiveCamera = () => useCameraMutation((id: string) => camerasService.archive(id));

export const useRestoreCamera = () => useCameraMutation((id: string) => camerasService.restore(id));

/**
 * Temporary signed URL for a private snapshot. Nothing is persisted; the URL
 * is refetched before the five-minute signature expires.
 */
export const useEventSnapshot = (snapshotPath: string | null, enabled = true) =>
  useQuery({
    queryKey: ["snapshot", snapshotPath],
    queryFn: () => eventsService.createSnapshotSignedUrl(snapshotPath),
    enabled: enabled && Boolean(snapshotPath),
    staleTime: 240_000,
    refetchInterval: 240_000,
    retry: 1,
  });
