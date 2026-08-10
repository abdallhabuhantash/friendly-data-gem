import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

/**
 * Issues a short-lived ticket so an <img> tag can pull the annotated stream
 * through the server proxy. The RTSP URL and camera credentials never leave
 * the server.
 */
export const createStreamTicket = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data) => z.object({ cameraId: z.string().uuid() }).parse(data))
  .handler(async ({ data }) => {
    const { signStreamTicket } = await import("./stream-ticket.server");
    return { ticket: signStreamTicket(data.cameraId) };
  });
