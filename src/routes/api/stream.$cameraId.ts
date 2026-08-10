import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/api/stream/$cameraId")({
  server: {
    handlers: {
      GET: async ({ request, params }) => {
        const { verifyStreamTicket } = await import("@/lib/stream-ticket.server");
        const ticket = new URL(request.url).searchParams.get("t");
        if (!verifyStreamTicket(params.cameraId, ticket)) {
          return new Response("Unauthorized", { status: 401 });
        }

        const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
        const { data: settings } = await supabaseAdmin
          .from("system_settings")
          .select("ai_service_url")
          .maybeSingle();

        const base = (settings?.ai_service_url ?? "").replace(/\/$/, "");
        // The Python AI service is optional (not running in preview/demo).
        // Return 4xx instead of 5xx so it is treated as "no stream yet",
        // not as an application error.
        if (!base) return new Response("AI service is not configured", { status: 404 });

        const headers: Record<string, string> = {};
        const serviceKey = process.env["AI_SERVICE_KEY"];
        if (serviceKey) headers["X-Service-Key"] = serviceKey;

        try {
          const upstream = await fetch(`${base}/stream/${params.cameraId}`, { headers });
          if (!upstream.ok || !upstream.body) {
            return new Response("Stream unavailable", { status: 404 });
          }
          return new Response(upstream.body, {
            status: 200,
            headers: {
              "content-type":
                upstream.headers.get("content-type") ??
                "multipart/x-mixed-replace; boundary=frame",
              "cache-control": "no-store",
            },
          });
        } catch {
          return new Response("Stream unreachable", { status: 404 });
        }
      },
    },
  },
});
