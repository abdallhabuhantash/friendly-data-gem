import { useServerFn } from "@tanstack/react-start";
import { useQuery } from "@tanstack/react-query";
import { VideoOff } from "lucide-react";
import { useState } from "react";
import { createStreamTicket } from "@/lib/stream-ticket.functions";

/**
 * Renders the annotated MJPEG stream produced by the Python AI service.
 * The browser only ever talks to the app's own proxy route, never to the
 * camera or NVR.
 */
export function LiveStreamPlayer({ cameraId, offline }: { cameraId: string; offline: boolean }) {
  const issueTicket = useServerFn(createStreamTicket);
  const [failed, setFailed] = useState(false);

  const ticket = useQuery({
    queryKey: ["stream-ticket", cameraId],
    queryFn: () => issueTicket({ data: { cameraId } }),
    enabled: !offline,
    refetchInterval: 4 * 60_000,
    retry: false,
  });

  if (offline || failed || ticket.isError || !ticket.data) {
    return (
      <div className="flex flex-col items-center gap-1 text-muted-foreground">
        <VideoOff className="size-6" />
        <span className="font-mono text-[10px] uppercase tracking-[0.2em]">
          {offline ? "No signal" : "Awaiting AI service"}
        </span>
      </div>
    );
  }

  return (
    <img
      src={`/api/stream/${cameraId}?t=${encodeURIComponent(ticket.data.ticket)}`}
      alt="Annotated live camera stream with AI detection overlays"
      className="absolute inset-0 size-full object-cover"
      onError={() => setFailed(true)}
    />
  );
}
