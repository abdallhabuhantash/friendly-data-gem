import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

/**
 * Start / End Exam Session.
 *
 * Arming is deliberately explicit: handing out exam papers at the beginning of
 * an exam looks exactly like a paper exchange, so monitoring stays unarmed
 * until an administrator starts the session (see
 * `docs/exam-session-identity-contract.md` §11).
 *
 * The web app never fakes an armed session: it asks the local AI service, and
 * the session only becomes `active` because the AI service confirmed it and
 * wrote that state itself. If the AI service is not reachable or not
 * configured, the truthful failure is reported and nothing changes.
 */

const input = z.object({ examSessionId: z.string().uuid() });

type AiRuntimeReply = {
  armed: boolean;
  examSessionId: string;
  cameras: string[];
  startedAt: string | null;
  endedAt: string | null;
};

async function aiServiceCall(
  path: string,
): Promise<{ ok: true; body: AiRuntimeReply } | { ok: false; message: string }> {
  const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
  const { data: settings } = await supabaseAdmin
    .from("system_settings")
    .select("ai_service_url")
    .maybeSingle();

  const base = (settings?.ai_service_url ?? "").trim().replace(/\/$/, "");
  if (base === "") {
    return { ok: false, message: "The AI service endpoint is not configured in Settings." };
  }
  let origin: URL;
  try {
    origin = new URL(base);
  } catch {
    return { ok: false, message: "The configured AI service endpoint is not a valid URL." };
  }
  if (origin.protocol !== "http:" && origin.protocol !== "https:") {
    return { ok: false, message: "The AI service endpoint must be an http(s) URL." };
  }

  const serviceKey = process.env["AI_SERVICE_KEY"];
  if (!serviceKey) {
    return { ok: false, message: "AI_SERVICE_KEY is not set, so the AI service stays closed." };
  }

  try {
    const response = await fetch(`${base}${path}`, {
      method: "POST",
      headers: { "X-Service-Key": serviceKey },
    });
    const text = await response.text();
    if (!response.ok) {
      let detail = text;
      try {
        detail = String((JSON.parse(text) as { detail?: string }).detail ?? text);
      } catch {
        /* plain-text error body */
      }
      return { ok: false, message: detail || `AI service returned ${response.status}.` };
    }
    const parsed = (text === "" ? {} : JSON.parse(text)) as Record<string, unknown>;
    return {
      ok: true,
      body: {
        armed: parsed["armed"] === true,
        examSessionId: String(parsed["exam_session_id"] ?? ""),
        cameras: Array.isArray(parsed["cameras"]) ? (parsed["cameras"] as string[]) : [],
        startedAt: typeof parsed["started_at"] === "string" ? parsed["started_at"] : null,
        endedAt: typeof parsed["ended_at"] === "string" ? parsed["ended_at"] : null,
      },
    };
  } catch {
    return { ok: false, message: "The AI service is unreachable." };
  }
}

async function assertAdministrator(context: { supabase: unknown; userId: string }) {
  const supabase = context.supabase as {
    rpc: (
      name: string,
      args: Record<string, unknown>,
    ) => Promise<{ data: unknown; error: { message: string } | null }>;
  };
  const { data, error } = await supabase.rpc("has_role", {
    _user_id: context.userId,
    _role: "administrator",
  });
  if (error || data !== true)
    throw new Error("Only administrators can start or end an exam session.");
}

export const startExamSession = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data) => input.parse(data))
  .handler(async ({ data, context }) => {
    await assertAdministrator(context);
    const result = await aiServiceCall(`/exam-sessions/${data.examSessionId}/arm`);
    if (!result.ok) throw new Error(`Monitoring was not started. ${result.message}`);
    return result.body;
  });

export const endExamSession = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data) => input.parse(data))
  .handler(async ({ data, context }) => {
    await assertAdministrator(context);
    const result = await aiServiceCall(`/exam-sessions/${data.examSessionId}/end`);
    if (!result.ok) throw new Error(`The session was not ended. ${result.message}`);
    return result.body;
  });
