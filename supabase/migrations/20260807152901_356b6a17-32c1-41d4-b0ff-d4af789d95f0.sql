-- 1. Make event type generic: drop restrictive type CHECK constraints on events
DO $$
DECLARE c record;
BEGIN
  FOR c IN
    SELECT con.conname
    FROM pg_constraint con
    JOIN pg_class rel ON rel.oid = con.conrelid
    JOIN pg_namespace ns ON ns.oid = rel.relnamespace
    WHERE ns.nspname = 'public' AND rel.relname = 'events' AND con.contype = 'c'
      AND pg_get_constraintdef(con.oid) ILIKE '%type%'
      AND pg_get_constraintdef(con.oid) ILIKE '%cheating%'
  LOOP
    EXECUTE format('ALTER TABLE public.events DROP CONSTRAINT %I', c.conname);
  END LOOP;
END $$;

-- 2. Structured primary detection fields
ALTER TABLE public.events
  ADD COLUMN IF NOT EXISTS person_tracking_id text,
  ADD COLUMN IF NOT EXISTS trigger_object_class text,
  ADD COLUMN IF NOT EXISTS trigger_confidence numeric,
  ADD COLUMN IF NOT EXISTS association_status text NOT NULL DEFAULT 'not_applicable',
  ADD COLUMN IF NOT EXISTS association_confidence numeric,
  ADD COLUMN IF NOT EXISTS detection_duration_seconds numeric,
  ADD COLUMN IF NOT EXISTS detection_frame_count integer,
  ADD COLUMN IF NOT EXISTS evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS source_mode text NOT NULL DEFAULT 'live';

-- Backfill fractional duration from the legacy integer column
UPDATE public.events
   SET detection_duration_seconds = duration_seconds
 WHERE detection_duration_seconds IS NULL;

ALTER TABLE public.events
  DROP CONSTRAINT IF EXISTS events_association_status_check,
  DROP CONSTRAINT IF EXISTS events_source_mode_check,
  DROP CONSTRAINT IF EXISTS events_trigger_confidence_check,
  DROP CONSTRAINT IF EXISTS events_association_confidence_check,
  DROP CONSTRAINT IF EXISTS events_evidence_is_array_check,
  DROP CONSTRAINT IF EXISTS events_detection_duration_check;

ALTER TABLE public.events
  ADD CONSTRAINT events_association_status_check
    CHECK (association_status IN ('associated','uncertain','unassociated','not_applicable')),
  ADD CONSTRAINT events_source_mode_check
    CHECK (source_mode IN ('live','demo')),
  ADD CONSTRAINT events_trigger_confidence_check
    CHECK (trigger_confidence IS NULL OR (trigger_confidence >= 0 AND trigger_confidence <= 1)),
  ADD CONSTRAINT events_association_confidence_check
    CHECK (association_confidence IS NULL OR (association_confidence >= 0 AND association_confidence <= 1)),
  ADD CONSTRAINT events_evidence_is_array_check
    CHECK (jsonb_typeof(evidence) = 'array'),
  ADD CONSTRAINT events_detection_duration_check
    CHECK (detection_duration_seconds IS NULL OR detection_duration_seconds >= 0);

-- 3. AI rules: fractional duration + association engine configuration
ALTER TABLE public.ai_rules
  ALTER COLUMN min_duration_seconds TYPE numeric USING min_duration_seconds::numeric;

ALTER TABLE public.ai_rules
  ADD COLUMN IF NOT EXISTS person_confidence_threshold numeric NOT NULL DEFAULT 0.60,
  ADD COLUMN IF NOT EXISTS association_confidence_threshold numeric NOT NULL DEFAULT 0.65,
  ADD COLUMN IF NOT EXISTS min_matching_frames integer NOT NULL DEFAULT 5,
  ADD COLUMN IF NOT EXISTS require_person_association boolean NOT NULL DEFAULT true;

ALTER TABLE public.ai_rules
  DROP CONSTRAINT IF EXISTS ai_rules_person_confidence_check,
  DROP CONSTRAINT IF EXISTS ai_rules_association_confidence_check,
  DROP CONSTRAINT IF EXISTS ai_rules_confidence_threshold_range_check,
  DROP CONSTRAINT IF EXISTS ai_rules_min_duration_check,
  DROP CONSTRAINT IF EXISTS ai_rules_min_matching_frames_check;

ALTER TABLE public.ai_rules
  ADD CONSTRAINT ai_rules_person_confidence_check
    CHECK (person_confidence_threshold >= 0 AND person_confidence_threshold <= 1),
  ADD CONSTRAINT ai_rules_association_confidence_check
    CHECK (association_confidence_threshold >= 0 AND association_confidence_threshold <= 1),
  ADD CONSTRAINT ai_rules_confidence_threshold_range_check
    CHECK (confidence_threshold >= 0 AND confidence_threshold <= 1),
  ADD CONSTRAINT ai_rules_min_duration_check
    CHECK (min_duration_seconds >= 0),
  ADD CONSTRAINT ai_rules_min_matching_frames_check
    CHECK (min_matching_frames >= 1);

COMMENT ON COLUMN public.ai_rules.confidence_threshold IS
  'Trigger object confidence threshold (e.g. cell_phone). Single source of truth; no separate trigger_confidence_threshold column.';
COMMENT ON COLUMN public.events.person_tracking_id IS
  'Temporary AI tracking identifier only. Never a real-world/biometric identity.';

-- Recommended starting values for the active mobile phone rule
UPDATE public.ai_rules
   SET confidence_threshold = 0.55,
       person_confidence_threshold = 0.60,
       association_confidence_threshold = 0.65,
       min_duration_seconds = 1.50,
       min_matching_frames = 5,
       cooldown_seconds = 20,
       require_person_association = true
 WHERE available = true;