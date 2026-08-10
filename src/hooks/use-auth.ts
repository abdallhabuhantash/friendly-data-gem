import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "@/integrations/supabase/client";
import { authService } from "@/services/auth-service";
import { usersService } from "@/services/monitoring-service";
import type { AppUser, UserRole } from "@/types";

export function useAuth() {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const { data: sub } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next);
      setReady(true);
    });
    void supabase.auth.getSession().then(({ data }) => {
      setSession(data.session ?? null);
      setReady(true);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  const userId = session?.user.id ?? null;

  const profile = useQuery<AppUser | null>({
    queryKey: ["profile", userId],
    queryFn: () => usersService.byId(userId as string),
    enabled: Boolean(userId),
  });

  const user = profile.data ?? null;
  const role: UserRole | null = user?.role ?? null;

  return {
    ready,
    session,
    user,
    role,
    isAdministrator: role === "administrator",
    isAuthenticated: session !== null,
    signIn: authService.signIn,
    signOut: authService.signOut,
  };
}
