
-- Helper: does this user hold any assigned role?
CREATE OR REPLACE FUNCTION public.has_any_role(_user_id uuid)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (SELECT 1 FROM public.user_roles WHERE user_id = _user_id);
$$;

REVOKE ALL ON FUNCTION public.has_any_role(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.has_any_role(uuid) TO authenticated;

-- Restrict configuration/operational reads to role holders
DROP POLICY IF EXISTS cameras_select ON public.cameras;
CREATE POLICY cameras_select ON public.cameras FOR SELECT TO authenticated
  USING (public.has_any_role(auth.uid()));

DROP POLICY IF EXISTS ai_rules_select ON public.ai_rules;
CREATE POLICY ai_rules_select ON public.ai_rules FOR SELECT TO authenticated
  USING (public.has_any_role(auth.uid()));

DROP POLICY IF EXISTS ai_rule_cameras_select ON public.ai_rule_cameras;
CREATE POLICY ai_rule_cameras_select ON public.ai_rule_cameras FOR SELECT TO authenticated
  USING (public.has_any_role(auth.uid()));

DROP POLICY IF EXISTS service_health_select ON public.service_health;
CREATE POLICY service_health_select ON public.service_health FOR SELECT TO authenticated
  USING (public.has_any_role(auth.uid()));

DROP POLICY IF EXISTS system_settings_select ON public.system_settings;
CREATE POLICY system_settings_select ON public.system_settings FOR SELECT TO authenticated
  USING (public.has_any_role(auth.uid()));

DROP POLICY IF EXISTS events_select ON public.events;
CREATE POLICY events_select ON public.events FOR SELECT TO authenticated
  USING (public.has_any_role(auth.uid()));

-- Snapshots: role holders only, and only files referenced by a real event
DROP POLICY IF EXISTS snapshots_read_authenticated ON storage.objects;
CREATE POLICY snapshots_read_authenticated ON storage.objects FOR SELECT TO authenticated
  USING (
    bucket_id = 'snapshots'
    AND public.has_any_role(auth.uid())
    AND EXISTS (SELECT 1 FROM public.events e WHERE e.snapshot_path = storage.objects.name)
  );

-- Lock down SECURITY DEFINER function execution
REVOKE ALL ON FUNCTION public.handle_new_user() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.update_updated_at_column() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.has_role(uuid, public.app_role) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.has_role(uuid, public.app_role) TO authenticated;
REVOKE ALL ON FUNCTION public.is_admin() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.is_admin() TO authenticated;
REVOKE ALL ON FUNCTION public.review_event(uuid, text, text) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.review_event(uuid, text, text) TO authenticated;
