import { redirect } from "@tanstack/react-router";
import { supabase } from "@/integrations/supabase/client";

/**
 * Route-level administrator gate. Backed by row level security: the database
 * rejects administrator-only writes even if this check is bypassed.
 */
export async function requireAdministrator() {
  const { data: userData } = await supabase.auth.getUser();
  const userId = userData.user?.id;
  if (!userId) throw redirect({ to: "/login" });
  const { data: isAdmin, error } = await supabase.rpc("has_role", {
    _user_id: userId,
    _role: "administrator",
  });
  if (error || !isAdmin) throw redirect({ to: "/dashboard" });
}
