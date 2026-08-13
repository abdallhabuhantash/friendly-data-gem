-- Anonymous exam subject identity: lifecycle and current track association are
-- separate concepts. Subject existence outlives any raw tracker binding.
ALTER TABLE public.session_subjects
  DROP CONSTRAINT session_subjects_status_valid;

ALTER TABLE public.session_subjects
  ADD COLUMN lifecycle_status text NOT NULL DEFAULT 'active',
  ADD COLUMN track_association text NOT NULL DEFAULT 'unresolved',
  ADD COLUMN active_raw_tracking_id text,
  ADD COLUMN last_bbox_x numeric,
  ADD COLUMN last_bbox_y numeric,
  ADD COLUMN last_bbox_width numeric,
  ADD COLUMN last_bbox_height numeric,
  ADD COLUMN velocity_x numeric,
  ADD COLUMN velocity_y numeric,
  ADD COLUMN motion_updated_at timestamptz,
  ADD CONSTRAINT session_subjects_lifecycle_valid CHECK (
    lifecycle_status IN ('active','temporarily_lost','lost','ended')
  ),
  ADD CONSTRAINT session_subjects_association_valid CHECK (
    track_association IN ('confirmed','provisional','unresolved','conflict')
  );

-- The old single overloaded status column and the fixed seat-like anchor are
-- replaced by the two columns above plus mobility-aware motion state.
ALTER TABLE public.session_subjects
  DROP COLUMN tracking_status,
  DROP COLUMN anchor_x,
  DROP COLUMN anchor_y,
  DROP COLUMN anchor_width,
  DROP COLUMN anchor_height,
  DROP COLUMN anchor_updated_at;

CREATE INDEX session_subjects_lifecycle_idx
  ON public.session_subjects (exam_session_id, lifecycle_status);

-- Monotonic, race-free subject numbering per exam session. Numbers are never
-- reused inside one session; a different session starts again at 1.
CREATE TABLE public.session_subject_sequences (
  exam_session_id uuid PRIMARY KEY REFERENCES public.exam_sessions(id) ON DELETE CASCADE,
  next_number integer NOT NULL DEFAULT 1,
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT session_subject_sequences_next_positive CHECK (next_number >= 1)
);

-- No `authenticated` grant on purpose: a browser must never allocate identities.
GRANT ALL ON public.session_subject_sequences TO service_role;
ALTER TABLE public.session_subject_sequences ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.allocate_session_subject_number(_exam_session_id uuid)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  _number integer;
BEGIN
  -- The row lock taken by ON CONFLICT DO UPDATE makes concurrent allocation
  -- safe; GREATEST(...) also protects against a sequence row that lags behind
  -- subjects already persisted (for example after a service restart).
  INSERT INTO public.session_subject_sequences AS s (exam_session_id, next_number)
  VALUES (_exam_session_id, 2)
  ON CONFLICT (exam_session_id) DO UPDATE
    SET next_number = GREATEST(
          s.next_number,
          COALESCE(
            (SELECT max(subject_number) + 1
               FROM public.session_subjects
              WHERE exam_session_id = _exam_session_id),
            1
          )
        ) + 1,
        updated_at = now()
  RETURNING next_number - 1 INTO _number;
  RETURN _number;
END;
$$;

REVOKE ALL ON FUNCTION public.allocate_session_subject_number(uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.allocate_session_subject_number(uuid) TO service_role;

-- Identity immutability is enforced here, not only by UI convention.
CREATE OR REPLACE FUNCTION public.guard_session_subject_identity()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
  IF NEW.subject_number IS DISTINCT FROM OLD.subject_number
     OR NEW.exam_session_id IS DISTINCT FROM OLD.exam_session_id THEN
    RAISE EXCEPTION
      'Exam subject identity is immutable: subject_number and exam_session_id cannot change';
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER guard_session_subject_identity
  BEFORE UPDATE ON public.session_subjects
  FOR EACH ROW EXECUTE FUNCTION public.guard_session_subject_identity();

-- Losing tracking never deletes a subject: ended exams stay auditable. Rows may
-- only disappear together with their exam session.
CREATE OR REPLACE FUNCTION public.guard_session_subject_retention()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
  IF EXISTS (SELECT 1 FROM public.exam_sessions WHERE id = OLD.exam_session_id) THEN
    RAISE EXCEPTION
      'Exam subject history cannot be deleted while its exam session exists';
  END IF;
  RETURN OLD;
END;
$$;

CREATE TRIGGER guard_session_subject_retention
  BEFORE DELETE ON public.session_subjects
  FOR EACH ROW EXECUTE FUNCTION public.guard_session_subject_retention();

-- Append-only raw track history gains audit detail.
ALTER TABLE public.session_subject_tracks
  DROP CONSTRAINT session_subject_tracks_method_valid;

ALTER TABLE public.session_subject_tracks
  ADD COLUMN association_state text NOT NULL DEFAULT 'confirmed',
  ADD COLUMN start_reason text,
  ADD COLUMN end_reason text,
  ADD CONSTRAINT session_subject_tracks_method_valid CHECK (
    association_method IN ('initial','short_gap_reassociation','restored_after_restart')
  ),
  ADD CONSTRAINT session_subject_tracks_state_valid CHECK (
    association_state IN ('confirmed','provisional')
  );