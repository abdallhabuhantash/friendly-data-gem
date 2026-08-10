-- 1. Operating mode
ALTER TABLE public.system_settings
  ADD COLUMN IF NOT EXISTS operation_mode text NOT NULL DEFAULT 'demo';

ALTER TABLE public.system_settings
  DROP CONSTRAINT IF EXISTS system_settings_operation_mode_check;
ALTER TABLE public.system_settings
  ADD CONSTRAINT system_settings_operation_mode_check
  CHECK (operation_mode IN ('demo','live'));

-- 2. Demo marker on service health so seeded rows are never mistaken for real hardware
ALTER TABLE public.service_health
  ADD COLUMN IF NOT EXISTS is_demo boolean NOT NULL DEFAULT false;

-- 3. Profiles: admins see all, operators see only themselves
DROP POLICY IF EXISTS profiles_select_all ON public.profiles;
CREATE POLICY profiles_select_self_or_admin ON public.profiles
  FOR SELECT TO authenticated
  USING (id = auth.uid() OR public.is_admin());

-- 4. User roles: admins see all, users read only their own role
DROP POLICY IF EXISTS user_roles_select ON public.user_roles;
CREATE POLICY user_roles_select_self_or_admin ON public.user_roles
  FOR SELECT TO authenticated
  USING (user_id = auth.uid() OR public.is_admin());

-- 5. Events: remove broad UPDATE, replace with a narrow review RPC
DROP POLICY IF EXISTS events_review ON public.events;

CREATE OR REPLACE FUNCTION public.review_event(
  _event_id uuid,
  _status text,
  _note text DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  _reviewer text;
BEGIN
  IF auth.uid() IS NULL THEN
    RAISE EXCEPTION 'Authentication required';
  END IF;
  IF _status NOT IN ('new','under_review','confirmed','rejected') THEN
    RAISE EXCEPTION 'Invalid review status: %', _status;
  END IF;

  SELECT COALESCE(NULLIF(p.full_name, ''), p.email)
    INTO _reviewer
    FROM public.profiles p
   WHERE p.id = auth.uid();

  UPDATE public.events
     SET status = _status,
         reviewed_by = COALESCE(_reviewer, 'Reviewer'),
         reviewed_at = now(),
         note = COALESCE(_note, note)
   WHERE id = _event_id;
END;
$$;

REVOKE ALL ON FUNCTION public.review_event(uuid, text, text) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.review_event(uuid, text, text) TO authenticated;