# Sentinel — AI Camera Monitoring Prototype: Implementation Plan

## 1. Overall architecture

```text
IP cameras / NVR ──RTSP──> Python AI service (Windows Server)
                              │  YOLO detect, persistence, cooldown, snapshots
                              │  MJPEG/HLS annotated stream  ──> browser <img>/<video>
                              └─REST + service key──> Supabase (events, snapshots, heartbeats)
Browser (Lovable app) ── Supabase Auth/DB/Storage/Realtime ──> live UI
Browser ── signed proxy route (/api/stream/$cameraId) ──> Python stream (RTSP never exposed)
```

Three owners, no overlap:
- **Lovable**: auth, all UI, camera/rule CRUD, event review, reports, roles, stream proxy.
- **Python service**: RTSP ingest, YOLO inference, detection persistence + cooldown, snapshot capture and upload, event insert, heartbeat, annotated stream endpoint.
- **NVR**: camera recording, retention, channel management, RTSP endpoints. The app only *reads* status the Python service reports.

## 2. Frontend routes and layouts

Keep the current TanStack layout shell.
- Public: `/login`.
- `_authenticated/` gate (existing) wraps: `/dashboard`, `/monitoring`, `/events`, `/cameras`, `/ai-rules`, `/reports`, `/users`, `/settings`, `/profile`.
- Admin-only subtree gate for `/users`, `/settings`, `/cameras` write actions, `/ai-rules` write actions. Operators get read + event review only.
- Layout stays the SOC console from the screenshots: left context rail, dense center stage, right live event feed.

## 3. Reusable components

Existing: `Panel`, `StatTile`, `StatusDot`, `EventBadges`, `CameraTile`, `TopBar`, `AppSidebar`, `PageContainer`.
To add: `LiveStreamPlayer` (proxy URL + reconnect + offline placeholder), `EventFeed` / `EventRow`, `EventDetailDrawer` (snapshot, confidence, review actions), `AlertToastStack` (realtime overlay on monitoring), `ConfidenceMeter`, `RuleEditorForm`, `RoleGate`, `CameraFormDialog`, `HeartbeatBadge`.

## 4. Supabase tables

- `profiles` (id → auth.users, full_name, email, status, last_active_at) — trigger-created on signup.
- `user_roles` (user_id, role: `administrator` | `operator`) — separate table, enum, `has_role()` security-definer fn.
- `cameras` (id, name, location, host, channel, status, ai_enabled, recording, resolution, fps, is_demo, last_heartbeat_at). **Credentials/RTSP path are NOT here** — see §6.
- `camera_credentials` (camera_id, rtsp_url, username, password) — no anon/authenticated grants at all; service role only.
- `ai_rules` (id, name, description, available, enabled, confidence_threshold, min_duration_seconds, cooldown_seconds, severity, save_snapshot, sound_notification).
- `ai_rule_cameras` (rule_id, camera_id) — many-to-many.
- `events` (id, type, severity, status, camera_id, rule_id, confidence, duration_seconds, snapshot_path, detected_at, reviewed_by, reviewed_at, note).
- `service_health` (service, online, payload jsonb, updated_at) — AI service + NVR status.
- Storage bucket `snapshots` (private, signed URLs only).

Every `CREATE TABLE` ships with explicit GRANTs, then RLS enable, then policies.

## 5. Auth and roles

Supabase email/password only. No registration page, no social login: administrators create accounts via an admin-only server function using the Auth Admin API. Roles read from `user_roles` through `has_role()`; UI gates are cosmetic, RLS is the real boundary.

## 6. RLS requirements

- `profiles`: read own + admins read all; update own.
- `user_roles`: select for authenticated; insert/update/delete admin-only.
- `cameras`, `ai_rules`: select for authenticated; write admin-only.
- `events`: select for authenticated; update (review fields) for authenticated; insert restricted to service role (the Python service).
- `camera_credentials`: **no policies for anon/authenticated** — service role only. The browser never sees an RTSP URL or password.
- `snapshots` bucket: private; access via short-lived signed URLs issued by a server function.

## 7. Python AI service API contract

Auth: static `X-Service-Key` header both directions (secret stored in Lovable Cloud secrets and on the Windows box).

- `GET  /health` → `{ online, version, model, device, inference_fps, queue_depth, gpu_load_percent, uptime_seconds }`
- `GET  /cameras` → runtime state per camera id
- `POST /cameras/{id}/start` · `POST /cameras/{id}/stop`
- `POST /rules/sync` ← Lovable pushes rule config on save
- `GET  /stream/{camera_id}` → annotated MJPEG (`multipart/x-mixed-replace`), HLS optional later
- `GET  /snapshot/{camera_id}` → single annotated JPEG
- Outbound: service inserts into Supabase `events` + uploads snapshot to storage using the service role key.

## 8. Real-time events

Supabase Realtime on `events` (postgres_changes INSERT/UPDATE). One subscription in a shared hook; drives the right-hand feed, dashboard counters, and the toast stack on `/monitoring`. Fallback polling every 15s if the channel drops.

## 9. Live video integration

Browser never touches RTSP. A TanStack server route `/api/stream/$cameraId` verifies the session and role, resolves the camera, and proxies the Python MJPEG stream server-side with the service key attached. `LiveStreamPlayer` renders it and shows an offline/demo placeholder when the camera is not reachable.

## 10. Mobile phone detection logic (Python side)

Per frame: detect `person` and `cell phone`. If a phone box overlaps/associates with a person box above the rule's `confidence_threshold` continuously for `min_duration_seconds`, emit one event of type `suspicious_cheating_activity`, then suppress further events for that camera+rule for `cooldown_seconds`. Snapshot saved when `save_snapshot` is on. Wording is fixed: Suspicious Cheating Activity / Possible Cheating Activity / Mobile Phone Detected — never "cheating confirmed". Other detection types appear only as `available: false`, visibly marked not implemented.

## 11. Mock-data strategy

Current mock service layer stays as the seam. Each service module gets a real Supabase implementation behind the same function signatures; a `VITE_USE_MOCKS` flag chooses. Demo cameras (`is_demo`) render a looping placeholder tile so the dashboard is never empty during the defense.

## 12. Build order

1. Enable Cloud; migrations for all tables + grants + RLS + seed demo rows.
2. Replace the local auth service with Supabase auth; admin-created users; role gates.
3. Swap cameras/rules/events services to Supabase; keep UI untouched.
4. Realtime feed + toast stack.
5. Stream proxy route + `LiveStreamPlayer` (placeholder until Python is up).
6. Python service: health, stream, one camera, phone rule, event insert.
7. Reports and NVR/service status from `service_health`.
8. Hardening: signed snapshot URLs, credential isolation audit, security scan.

## 13. Testing criteria per phase

- Auth: operator cannot open `/users`; direct SQL as operator cannot write `cameras`.
- Data: RLS verified by signed-in reads/writes, not just UI.
- Realtime: event inserted via SQL appears in the feed under 2s with no refresh.
- Stream: proxy returns frames; RTSP URL absent from every network response and bundle.
- Detection: phone held ≥ threshold produces exactly one event; repeat within cooldown produces none.
- Reports: counts reconcile with `events` rows.

## 14. Risks (one-month window)

- YOLO throughput on the available GPU/CPU — mitigate with frame skipping and reduced resolution.
- RTSP/NVR credentials and network access at the exam site arriving late.
- MJPEG latency/bandwidth over the campus network.
- Windows Server firewall/HTTPS mixed-content between the published app and the local service (may need a tunnel).
- Snapshot storage cost/retention.

## 15. Reduced MVP (guaranteed demo)

Login, dashboard, monitoring with **one** camera, live annotated stream, phone-detection events arriving in realtime with snapshot, event review (confirm/reject), cameras list, one editable AI rule, basic reports, admin/operator split. Everything else stays visible but read-only or marked not implemented.

## Open questions (only if placeholders won't do)

1. Will the Python service be reachable over HTTPS from the published app, or only on the local network during the demo?
2. Should administrators create operator accounts in-app, or will you seed them manually?
