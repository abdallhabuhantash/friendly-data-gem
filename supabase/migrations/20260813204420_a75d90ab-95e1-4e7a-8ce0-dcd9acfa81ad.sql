CREATE TABLE public.session_subjects (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  exam_session_id uuid NOT NULL REFERENCES public.exam_sessions(id) ON DELETE CASCADE,
  subject_number integer NOT NULL,
  subject_label text NOT NULL GENERATED ALWAYS AS ('S' || lpad(subject_number::text, 3, '0')) STORED,
  camera_id uuid REFERENCES public.cameras(id) ON DELETE SET NULL,
  tracking_status text NOT NULL DEFAULT 'stable',
  first_seen_at timestamptz NOT NULL DEFAULT now(),
  last_seen_at timestamptz NOT NULL DEFAULT now(),
  ended_at timestamptz,
  anchor_x numeric,
  anchor_y numeric,
  anchor_width numeric,
  anchor_height numeric,
  anchor_updated_at timestamptz,
  last_association_confidence numeric,
  reassociation_count integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT session_subjects_number_positive CHECK (subject_number >= 1),
  CONSTRAINT session_subjects_status_valid CHECK (
    tracking_status IN ('stable','temporarily_lost','uncertain','conflict','ended')
  ),
  CONSTRAINT session_subjects_unique_number UNIQUE (exam_session_id, subject_number)
);

CREATE INDEX session_subjects_session_idx ON public.session_subjects (exam_session_id);
CREATE INDEX session_subjects_active_idx
  ON public.session_subjects (exam_session_id)
  WHERE ended_at IS NULL;

GRANT SELECT ON public.session_subjects TO authenticated;
GRANT ALL ON public.session_subjects TO service_role;
ALTER TABLE public.session_subjects ENABLE ROW LEVEL SECURITY;
CREATE POLICY session_subjects_select ON public.session_subjects
  FOR SELECT TO authenticated USING (public.has_any_role(auth.uid()));

CREATE TABLE public.session_subject_tracks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_subject_id uuid NOT NULL REFERENCES public.session_subjects(id) ON DELETE CASCADE,
  exam_session_id uuid NOT NULL REFERENCES public.exam_sessions(id) ON DELETE CASCADE,
  raw_tracking_id text NOT NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  ended_at timestamptz,
  association_method text NOT NULL DEFAULT 'initial',
  association_confidence numeric,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT session_subject_tracks_raw_id_not_blank CHECK (btrim(raw_tracking_id) <> ''),
  CONSTRAINT session_subject_tracks_method_valid CHECK (
    association_method IN ('initial','short_gap_reassociation')
  )
);

CREATE INDEX session_subject_tracks_subject_idx
  ON public.session_subject_tracks (session_subject_id, started_at);
CREATE UNIQUE INDEX session_subject_tracks_one_open_per_session
  ON public.session_subject_tracks (exam_session_id, raw_tracking_id)
  WHERE ended_at IS NULL;

GRANT SELECT ON public.session_subject_tracks TO authenticated;
GRANT ALL ON public.session_subject_tracks TO service_role;
ALTER TABLE public.session_subject_tracks ENABLE ROW LEVEL SECURITY;
CREATE POLICY session_subject_tracks_select ON public.session_subject_tracks
  FOR SELECT TO authenticated USING (public.has_any_role(auth.uid()));

CREATE TRIGGER update_session_subjects_updated_at
  BEFORE UPDATE ON public.session_subjects
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

ALTER PUBLICATION supabase_realtime ADD TABLE public.session_subjects;
ALTER TABLE public.session_subjects REPLICA IDENTITY FULL;