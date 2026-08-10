ALTER TABLE public.cameras
  ADD COLUMN IF NOT EXISTS source_type text NOT NULL DEFAULT 'direct_camera',
  ADD COLUMN IF NOT EXISTS rtsp_port integer NOT NULL DEFAULT 554,
  ADD COLUMN IF NOT EXISTS stream_path text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS stream_profile text NOT NULL DEFAULT 'main',
  ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS updated_at timestamp with time zone NOT NULL DEFAULT now();

UPDATE public.cameras SET source_type = 'demo' WHERE is_demo = true;
UPDATE public.cameras SET source_type = 'direct_camera' WHERE is_demo = false AND source_type NOT IN ('direct_camera','nvr_channel','demo');

DO $$ BEGIN
  ALTER TABLE public.cameras ADD CONSTRAINT cameras_source_type_check CHECK (source_type IN ('direct_camera','nvr_channel','demo'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.cameras ADD CONSTRAINT cameras_rtsp_port_check CHECK (rtsp_port BETWEEN 1 AND 65535);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.cameras ADD CONSTRAINT cameras_stream_profile_check CHECK (stream_profile IN ('main','sub','custom'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.cameras ADD CONSTRAINT cameras_channel_check CHECK (channel >= 1);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.cameras ADD CONSTRAINT cameras_stream_path_no_credentials CHECK (stream_path !~ '@' AND stream_path !~* 'rtsp://');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = public
AS $$ BEGIN NEW.updated_at = now(); RETURN NEW; END; $$;

DROP TRIGGER IF EXISTS update_cameras_updated_at ON public.cameras;
CREATE TRIGGER update_cameras_updated_at
BEFORE UPDATE ON public.cameras
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- Camera credentials stay server/service-only: no browser role may read or write them.
ALTER TABLE public.camera_credentials ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE pol record;
BEGIN
  FOR pol IN SELECT policyname FROM pg_policies WHERE schemaname = 'public' AND tablename = 'camera_credentials'
  LOOP
    EXECUTE format('DROP POLICY %I ON public.camera_credentials', pol.policyname);
  END LOOP;
END $$;

REVOKE ALL ON public.camera_credentials FROM anon, authenticated;
GRANT ALL ON public.camera_credentials TO service_role;