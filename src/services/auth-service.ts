/**
 * Authentication is handled by Lovable Cloud (email + password).
 * Nothing about the session is stored by the app itself.
 */
import { supabase } from "@/integrations/supabase/client";

export const authService = {
  getUser: async () => {
    const { data } = await supabase.auth.getUser();
    return data.user ?? null;
  },
  getSession: async () => {
    const { data } = await supabase.auth.getSession();
    return data.session ?? null;
  },
  signIn: async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    });
    if (error) throw new Error(error.message);
    return data.user;
  },
  signOut: async () => {
    await supabase.auth.signOut();
  },
};
