-- Phase 2 correctness patch: mark legacy prototype seed rows as demo (idempotent)

UPDATE public.cameras
   SET is_demo = true
 WHERE id IN (
   '11111111-1111-4111-8111-000000000001',
   '11111111-1111-4111-8111-000000000002',
   '11111111-1111-4111-8111-000000000003',
   '11111111-1111-4111-8111-000000000004'
 );

UPDATE public.events
   SET source_mode = 'demo'
 WHERE camera_id IN (
   '11111111-1111-4111-8111-000000000001',
   '11111111-1111-4111-8111-000000000002',
   '11111111-1111-4111-8111-000000000003',
   '11111111-1111-4111-8111-000000000004'
 )
   AND source_mode <> 'demo';

-- Seeded service health rows are demonstration placeholders.
-- Real Python AI / NVR heartbeat writers MUST upsert is_demo = false explicitly
-- when reporting genuine hardware/service health.
UPDATE public.service_health
   SET is_demo = true
 WHERE service IN ('ai', 'nvr')
   AND is_demo IS DISTINCT FROM true;