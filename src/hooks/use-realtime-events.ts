import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";

const LABEL: Record<string, string> = {
  suspicious_cheating_activity: "Suspicious Cheating Activity",
  possible_cheating_activity: "Possible Cheating Activity",
  mobile_phone_detected: "Mobile Phone Detected",
};

/**
 * Single realtime subscription on detection events written by the Python AI
 * service. Keeps every events-derived query fresh and raises an alert toast.
 */
export function useRealtimeEvents({ notify = false }: { notify?: boolean } = {}) {
  const queryClient = useQueryClient();

  useEffect(() => {
    const channel = supabase
      .channel("events-stream")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "events" },
        (payload) => {
          void queryClient.invalidateQueries({ queryKey: ["events"] });
          void queryClient.invalidateQueries({ queryKey: ["reports"] });
          if (notify && payload.eventType === "INSERT") {
            const row = payload.new as { type?: string; camera_name?: string };
            toast.warning(LABEL[row.type ?? ""] ?? "Detection event", {
              description: `${row.camera_name ?? "Camera"} — requires human review`,
            });
          }
        },
      )
      .subscribe();

    return () => {
      void supabase.removeChannel(channel);
    };
  }, [queryClient, notify]);
}
