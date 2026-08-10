-- Data-correctness: no invented operational state may exist in the app.
DELETE FROM public.events WHERE camera_id IN (SELECT id FROM public.cameras WHERE is_demo = true) OR source_mode = 'demo';
DELETE FROM public.ai_rule_cameras WHERE camera_id IN (SELECT id FROM public.cameras WHERE is_demo = true);
DELETE FROM public.camera_credentials WHERE camera_id IN (SELECT id FROM public.cameras WHERE is_demo = true);
DELETE FROM public.cameras WHERE is_demo = true;
DELETE FROM public.service_health WHERE is_demo = true;
UPDATE public.system_settings SET operation_mode = 'live' WHERE operation_mode = 'demo';
ALTER TABLE public.system_settings ALTER COLUMN operation_mode SET DEFAULT 'live';
