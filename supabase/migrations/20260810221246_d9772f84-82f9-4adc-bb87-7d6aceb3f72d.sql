UPDATE public.ai_rules
SET instant_confidence_threshold = confidence_threshold
WHERE instant_confidence_threshold < confidence_threshold;

ALTER TABLE public.ai_rules
ADD CONSTRAINT ai_rules_instant_threshold_not_weaker
CHECK (instant_confidence_threshold >= confidence_threshold);