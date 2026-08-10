import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

/**
 * One-time bootstrap so the very first administrator account can exist.
 * It refuses to run as soon as any account exists — this is not registration.
 */
export const needsBootstrap = createServerFn({ method: "GET" }).handler(async () => {
  const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
  const { count } = await supabaseAdmin
    .from("profiles")
    .select("id", { count: "exact", head: true });
  return { needsBootstrap: (count ?? 0) === 0 };
});

export const createFirstAdministrator = createServerFn({ method: "POST" })
  .inputValidator((data) =>
    z
      .object({
        email: z.string().email(),
        password: z.string().min(8),
        fullName: z.string().min(1),
      })
      .parse(data),
  )
  .handler(async ({ data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const { count } = await supabaseAdmin
      .from("profiles")
      .select("id", { count: "exact", head: true });
    if ((count ?? 0) > 0) throw new Error("An account already exists.");

    const { error } = await supabaseAdmin.auth.admin.createUser({
      email: data.email,
      password: data.password,
      email_confirm: true,
      user_metadata: { full_name: data.fullName, role: "administrator" },
    });
    if (error) throw new Error(error.message);
    return { ok: true };
  });
