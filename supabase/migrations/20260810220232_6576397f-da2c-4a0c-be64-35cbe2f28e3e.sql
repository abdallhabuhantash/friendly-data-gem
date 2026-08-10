ALTER TABLE public.ai_rules
  ADD COLUMN IF NOT EXISTS instant_detection_enabled boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS instant_confidence_threshold numeric NOT NULL DEFAULT 0.85;

ALTER TABLE public.ai_rules
  DROP CONSTRAINT IF EXISTS ai_rules_instant_confidence_threshold_check;

ALTER TABLE public.ai_rules
  ADD CONSTRAINT ai_rules_instant_confidence_threshold_check
    CHECK (instant_confidence_threshold >= 0 AND instant_confidence_threshold <= 1);

COMMENT ON COLUMN public.ai_rules.instant_detection_enabled IS
  'Preserve very brief (possibly single-frame) visible-phone evidence as a mobile_phone_detected warning, independently of temporal confirmation.';
COMMENT ON COLUMN public.ai_rules.instant_confidence_threshold IS
  'Stricter trigger-object confidence required for single-frame instant evidence; effective value is never below confidence_threshold.';