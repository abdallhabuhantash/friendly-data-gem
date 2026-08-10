-- ============ ROLES ============
CREATE TYPE public.app_role AS ENUM ('administrator', 'operator');

CREATE TABLE public.profiles (
  id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name text NOT NULL DEFAULT '',
  email text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','suspended')),
  last_active_at timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.profiles TO authenticated;
GRANT ALL ON public.profiles TO service_role;
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE TABLE public.user_roles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role public.app_role NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, role)
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.user_roles TO authenticated;
GRANT ALL ON public.user_roles TO service_role;
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.has_role(_user_id uuid, _role public.app_role)
RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT EXISTS (SELECT 1 FROM public.user_roles WHERE user_id = _user_id AND role = _role);
$$;

CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT public.has_role(auth.uid(), 'administrator');
$$;

CREATE POLICY "profiles_select_all" ON public.profiles FOR SELECT TO authenticated USING (true);
CREATE POLICY "profiles_update_own" ON public.profiles FOR UPDATE TO authenticated USING (id = auth.uid()) WITH CHECK (id = auth.uid());
CREATE POLICY "profiles_admin_write" ON public.profiles FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE POLICY "user_roles_select" ON public.user_roles FOR SELECT TO authenticated USING (true);
CREATE POLICY "user_roles_admin_write" ON public.user_roles FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
  INSERT INTO public.profiles (id, full_name, email)
  VALUES (NEW.id, COALESCE(NEW.raw_user_meta_data->>'full_name', ''), COALESCE(NEW.email, ''))
  ON CONFLICT (id) DO NOTHING;

  INSERT INTO public.user_roles (user_id, role)
  VALUES (
    NEW.id,
    CASE WHEN NEW.raw_user_meta_data->>'role' = 'administrator' THEN 'administrator'::public.app_role
         ELSE 'operator'::public.app_role END
  )
  ON CONFLICT DO NOTHING;
  RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============ CAMERAS ============
CREATE TABLE public.cameras (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  location text NOT NULL DEFAULT '',
  host text NOT NULL DEFAULT '',
  channel integer NOT NULL DEFAULT 1,
  status text NOT NULL DEFAULT 'offline' CHECK (status IN ('online','offline','degraded')),
  ai_enabled boolean NOT NULL DEFAULT false,
  recording boolean NOT NULL DEFAULT false,
  resolution text NOT NULL DEFAULT '1920x1080',
  fps integer NOT NULL DEFAULT 0,
  is_demo boolean NOT NULL DEFAULT true,
  last_heartbeat_at timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.cameras TO authenticated;
GRANT ALL ON public.cameras TO service_role;
ALTER TABLE public.cameras ENABLE ROW LEVEL SECURITY;
CREATE POLICY "cameras_select" ON public.cameras FOR SELECT TO authenticated USING (true);
CREATE POLICY "cameras_admin_write" ON public.cameras FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

-- Service-role only. No grants for anon/authenticated: RTSP details never reach the browser.
CREATE TABLE public.camera_credentials (
  camera_id uuid PRIMARY KEY REFERENCES public.cameras(id) ON DELETE CASCADE,
  rtsp_url text NOT NULL,
  username text,
  password text,
  updated_at timestamptz NOT NULL DEFAULT now()
);
GRANT ALL ON public.camera_credentials TO service_role;
ALTER TABLE public.camera_credentials ENABLE ROW LEVEL SECURITY;

-- ============ AI RULES ============
CREATE TABLE public.ai_rules (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  description text NOT NULL DEFAULT '',
  available boolean NOT NULL DEFAULT false,
  enabled boolean NOT NULL DEFAULT false,
  confidence_threshold numeric NOT NULL DEFAULT 0.7,
  min_duration_seconds integer NOT NULL DEFAULT 3,
  cooldown_seconds integer NOT NULL DEFAULT 60,
  severity text NOT NULL DEFAULT 'warning' CHECK (severity IN ('critical','warning','info')),
  save_snapshot boolean NOT NULL DEFAULT true,
  sound_notification boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.ai_rules TO authenticated;
GRANT ALL ON public.ai_rules TO service_role;
ALTER TABLE public.ai_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY "ai_rules_select" ON public.ai_rules FOR SELECT TO authenticated USING (true);
CREATE POLICY "ai_rules_admin_write" ON public.ai_rules FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE TABLE public.ai_rule_cameras (
  rule_id uuid NOT NULL REFERENCES public.ai_rules(id) ON DELETE CASCADE,
  camera_id uuid NOT NULL REFERENCES public.cameras(id) ON DELETE CASCADE,
  PRIMARY KEY (rule_id, camera_id)
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.ai_rule_cameras TO authenticated;
GRANT ALL ON public.ai_rule_cameras TO service_role;
ALTER TABLE public.ai_rule_cameras ENABLE ROW LEVEL SECURITY;
CREATE POLICY "ai_rule_cameras_select" ON public.ai_rule_cameras FOR SELECT TO authenticated USING (true);
CREATE POLICY "ai_rule_cameras_admin_write" ON public.ai_rule_cameras FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

-- ============ EVENTS ============
CREATE TABLE public.events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  type text NOT NULL CHECK (type IN ('suspicious_cheating_activity','possible_cheating_activity','mobile_phone_detected')),
  severity text NOT NULL DEFAULT 'warning' CHECK (severity IN ('critical','warning','info')),
  status text NOT NULL DEFAULT 'new' CHECK (status IN ('new','under_review','confirmed','rejected')),
  camera_id uuid REFERENCES public.cameras(id) ON DELETE SET NULL,
  camera_name text NOT NULL DEFAULT '',
  rule_id uuid REFERENCES public.ai_rules(id) ON DELETE SET NULL,
  confidence numeric NOT NULL DEFAULT 0,
  duration_seconds integer NOT NULL DEFAULT 0,
  snapshot_path text,
  detected_at timestamptz NOT NULL DEFAULT now(),
  reviewed_by text,
  reviewed_at timestamptz,
  note text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX events_detected_at_idx ON public.events (detected_at DESC);
GRANT SELECT, UPDATE ON public.events TO authenticated;
GRANT ALL ON public.events TO service_role;
ALTER TABLE public.events ENABLE ROW LEVEL SECURITY;
CREATE POLICY "events_select" ON public.events FOR SELECT TO authenticated USING (true);
CREATE POLICY "events_review" ON public.events FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

ALTER PUBLICATION supabase_realtime ADD TABLE public.events;
ALTER TABLE public.events REPLICA IDENTITY FULL;

-- ============ SERVICE HEALTH ============
CREATE TABLE public.service_health (
  service text PRIMARY KEY CHECK (service IN ('ai','nvr')),
  online boolean NOT NULL DEFAULT false,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  updated_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT ON public.service_health TO authenticated;
GRANT ALL ON public.service_health TO service_role;
ALTER TABLE public.service_health ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_health_select" ON public.service_health FOR SELECT TO authenticated USING (true);

-- ============ SETTINGS ============
CREATE TABLE public.system_settings (
  id boolean PRIMARY KEY DEFAULT true CHECK (id),
  ai_service_url text NOT NULL DEFAULT '',
  websocket_url text NOT NULL DEFAULT '',
  retention_days integer NOT NULL DEFAULT 30,
  snapshot_storage text NOT NULL DEFAULT 'cloud' CHECK (snapshot_storage IN ('local','cloud')),
  sound_alerts boolean NOT NULL DEFAULT true,
  auto_acknowledge_minutes integer NOT NULL DEFAULT 30,
  timezone text NOT NULL DEFAULT 'Asia/Amman',
  updated_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE ON public.system_settings TO authenticated;
GRANT ALL ON public.system_settings TO service_role;
ALTER TABLE public.system_settings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "system_settings_select" ON public.system_settings FOR SELECT TO authenticated USING (true);
CREATE POLICY "system_settings_admin_write" ON public.system_settings FOR ALL TO authenticated USING (public.is_admin()) WITH CHECK (public.is_admin());

-- ============ SEED ============
INSERT INTO public.cameras (id, name, location, host, channel, status, ai_enabled, recording, resolution, fps, is_demo, last_heartbeat_at) VALUES
 ('11111111-1111-4111-8111-000000000001','Exam Hall A — Front','Building C / Hall A','192.168.1.64',1,'online',true,true,'1920x1080',25,false, now()),
 ('11111111-1111-4111-8111-000000000002','Exam Hall A — Rear [DEMO]','Building C / Hall A','192.168.1.65',2,'online',true,true,'1920x1080',20,true, now() - interval '1 minute'),
 ('11111111-1111-4111-8111-000000000003','Exam Hall B [DEMO]','Building C / Hall B','192.168.1.66',3,'degraded',true,false,'1280x720',12,true, now() - interval '4 minutes'),
 ('11111111-1111-4111-8111-000000000004','Corridor West [DEMO]','Building C / Level 1','192.168.1.67',4,'offline',false,false,'1280x720',0,true, now() - interval '96 minutes');

INSERT INTO public.ai_rules (id, name, description, available, enabled, confidence_threshold, min_duration_seconds, cooldown_seconds, severity, save_snapshot, sound_notification) VALUES
 ('22222222-2222-4222-8222-000000000001','Mobile Phone Cheating Detection','Flags a person holding or using a mobile phone for longer than the configured duration and raises a suspicious cheating activity event.',true,true,0.65,3,60,'critical',true,true),
 ('22222222-2222-4222-8222-000000000002','Repeated Head Turning','Planned behavioural rule. Not implemented in this prototype.',false,false,0.70,5,120,'warning',false,false),
 ('22222222-2222-4222-8222-000000000003','Unauthorized Person In Hall','Planned rule. Not implemented in this prototype.',false,false,0.75,4,90,'critical',false,false),
 ('22222222-2222-4222-8222-000000000004','Object Passing Between Seats','Planned rule. Not implemented in this prototype.',false,false,0.70,2,60,'warning',false,false);

INSERT INTO public.ai_rule_cameras (rule_id, camera_id) VALUES
 ('22222222-2222-4222-8222-000000000001','11111111-1111-4111-8111-000000000001'),
 ('22222222-2222-4222-8222-000000000001','11111111-1111-4111-8111-000000000002'),
 ('22222222-2222-4222-8222-000000000001','11111111-1111-4111-8111-000000000003');

INSERT INTO public.events (type, severity, status, camera_id, camera_name, rule_id, confidence, duration_seconds, detected_at, reviewed_by, note) VALUES
 ('suspicious_cheating_activity','critical','new','11111111-1111-4111-8111-000000000001','Exam Hall A — Front','22222222-2222-4222-8222-000000000001',0.91,6, now() - interval '3 minutes', NULL, NULL),
 ('mobile_phone_detected','warning','new','11111111-1111-4111-8111-000000000002','Exam Hall A — Rear [DEMO]','22222222-2222-4222-8222-000000000001',0.73,3, now() - interval '11 minutes', NULL, NULL),
 ('possible_cheating_activity','warning','under_review','11111111-1111-4111-8111-000000000001','Exam Hall A — Front','22222222-2222-4222-8222-000000000001',0.68,4, now() - interval '27 minutes','Operator on duty', NULL),
 ('suspicious_cheating_activity','critical','confirmed','11111111-1111-4111-8111-000000000003','Exam Hall B [DEMO]','22222222-2222-4222-8222-000000000001',0.88,9, now() - interval '64 minutes','System Administrator','Escalated to invigilator.'),
 ('mobile_phone_detected','info','rejected','11111111-1111-4111-8111-000000000002','Exam Hall A — Rear [DEMO]','22222222-2222-4222-8222-000000000001',0.55,2, now() - interval '122 minutes','Operator on duty','Object was a calculator.'),
 ('possible_cheating_activity','warning','confirmed','11111111-1111-4111-8111-000000000001','Exam Hall A — Front','22222222-2222-4222-8222-000000000001',0.79,5, now() - interval '190 minutes','System Administrator', NULL);

INSERT INTO public.service_health (service, online, payload) VALUES
 ('ai', true, '{"version":"0.9.3-prototype","model":"YOLOv8n (person, cell phone)","device":"CUDA:0","inference_fps":22.4,"queue_depth":1,"gpu_load_percent":47,"uptime_seconds":32400}'::jsonb),
 ('nvr', true, '{"model":"Generic NVR / 8CH","channels_used":4,"channels_total":8,"storage_used_percent":62,"retention_days":30}'::jsonb);

INSERT INTO public.system_settings (id, ai_service_url, websocket_url, retention_days, snapshot_storage, sound_alerts, auto_acknowledge_minutes, timezone)
VALUES (true, 'http://ai-server.local:8000', 'ws://ai-server.local:8000/ws/events', 30, 'cloud', true, 30, 'Asia/Amman');