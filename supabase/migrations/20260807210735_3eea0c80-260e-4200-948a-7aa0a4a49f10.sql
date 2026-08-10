ALTER TABLE public.ai_rules ADD COLUMN IF NOT EXISTS engine_key text;

UPDATE public.ai_rules
   SET engine_key = 'mobile_phone_detection'
 WHERE id = '22222222-2222-4222-8222-000000000001'
   AND engine_key IS DISTINCT FROM 'mobile_phone_detection';

COMMENT ON COLUMN public.ai_rules.engine_key IS 'Machine-readable identifier of the algorithm implementing this rule. NULL means no implementation exists yet.';