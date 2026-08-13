CREATE TABLE public.exam_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title text NOT NULL,
  course_code text NOT NULL DEFAULT '',
  location_label text NOT NULL DEFAULT '',
  scheduled_at timestamptz,
  status text NOT NULL DEFAULT 'draft',
  started_at timestamptz,
  ended_at timestamptz,
  created_by uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT exam_sessions_title_not_blank CHECK (btrim(title) <> ''),
  CONSTRAINT exam_sessions_status_valid CHECK (status IN ('draft','ready','active','ended','archived'))
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.exam_sessions TO authenticated;
GRANT ALL ON public.exam_sessions TO service_role;
ALTER TABLE public.exam_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY exam_sessions_select ON public.exam_sessions FOR SELECT TO authenticated USING (public.has_any_role(auth.uid()));
CREATE POLICY exam_sessions_admin_write ON public.exam_sessions FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE TABLE public.exam_session_cameras (
  exam_session_id uuid NOT NULL REFERENCES public.exam_sessions(id) ON DELETE CASCADE,
  camera_id uuid NOT NULL REFERENCES public.cameras(id) ON DELETE CASCADE,
  is_primary boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (exam_session_id, camera_id)
);

CREATE UNIQUE INDEX exam_session_cameras_one_primary
  ON public.exam_session_cameras (exam_session_id)
  WHERE is_primary;

GRANT SELECT, INSERT, UPDATE, DELETE ON public.exam_session_cameras TO authenticated;
GRANT ALL ON public.exam_session_cameras TO service_role;
ALTER TABLE public.exam_session_cameras ENABLE ROW LEVEL SECURITY;
CREATE POLICY exam_session_cameras_select ON public.exam_session_cameras FOR SELECT TO authenticated USING (public.has_any_role(auth.uid()));
CREATE POLICY exam_session_cameras_admin_write ON public.exam_session_cameras FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE TABLE public.exam_invigilators (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  exam_session_id uuid NOT NULL REFERENCES public.exam_sessions(id) ON DELETE CASCADE,
  full_name text NOT NULL,
  role text NOT NULL DEFAULT 'invigilator',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT exam_invigilators_name_not_blank CHECK (btrim(full_name) <> '')
);

CREATE INDEX exam_invigilators_session_idx ON public.exam_invigilators (exam_session_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.exam_invigilators TO authenticated;
GRANT ALL ON public.exam_invigilators TO service_role;
ALTER TABLE public.exam_invigilators ENABLE ROW LEVEL SECURITY;
CREATE POLICY exam_invigilators_select ON public.exam_invigilators FOR SELECT TO authenticated USING (public.has_any_role(auth.uid()));
CREATE POLICY exam_invigilators_admin_write ON public.exam_invigilators FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE TABLE public.exam_roster_students (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  exam_session_id uuid NOT NULL REFERENCES public.exam_sessions(id) ON DELETE CASCADE,
  university_id text NOT NULL,
  full_name text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT exam_roster_students_university_id_not_blank CHECK (btrim(university_id) <> ''),
  CONSTRAINT exam_roster_students_full_name_not_blank CHECK (btrim(full_name) <> ''),
  CONSTRAINT exam_roster_students_unique_per_session UNIQUE (exam_session_id, university_id)
);

CREATE INDEX exam_roster_students_session_idx ON public.exam_roster_students (exam_session_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.exam_roster_students TO authenticated;
GRANT ALL ON public.exam_roster_students TO service_role;
ALTER TABLE public.exam_roster_students ENABLE ROW LEVEL SECURITY;
CREATE POLICY exam_roster_students_select ON public.exam_roster_students FOR SELECT TO authenticated USING (public.has_any_role(auth.uid()));
CREATE POLICY exam_roster_students_admin_write ON public.exam_roster_students FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE TRIGGER update_exam_sessions_updated_at BEFORE UPDATE ON public.exam_sessions FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_exam_invigilators_updated_at BEFORE UPDATE ON public.exam_invigilators FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_exam_roster_students_updated_at BEFORE UPDATE ON public.exam_roster_students FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();