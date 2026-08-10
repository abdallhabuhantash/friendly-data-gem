# Service Health Contract

Heartbeat writers for the `service_health` table. Producers are not implemented
yet; this documents what they must write.

## AI service

```json
{
  "version": "1.0.0",
  "model": "YOLO",
  "device": "cuda:0",
  "inference_fps": 22.4,
  "queue_depth": 0,
  "gpu_load_percent": 47,
  "uptime_seconds": 1800
}
```

The AI writer must set:

- `service = "ai"`
- `online = true/false`
- `is_demo = false`
- `updated_at = current heartbeat time`
- `payload = ` the object above

## NVR

```json
{
  "model": "...",
  "channels_used": 1,
  "channels_total": 4,
  "storage_used_percent": 30,
  "retention_days": 7,
  "recording_active": true
}
```

The NVR writer must set:

- `service = "nvr"`
- `online = true/false`
- `is_demo = false`
- `updated_at = current heartbeat time`
- `payload = ` the object above

### recording_active

- `true` — recording explicitly reported active; the UI may show `REC`.
- `false` — explicitly reported inactive.
- missing/null — unknown; the UI must not claim `REC`.

## Freshness

A stored `online = true` is never trusted indefinitely. The console treats a
heartbeat as stale past its threshold (AI 30s, NVR 120s, cameras 60s) and shows
`Stale` or `Offline` regardless of the stored flag.

## Camera heartbeats

`cameras.status` and `cameras.last_heartbeat_at` must be updated only from
actual observed runtime connectivity to that stream — never optimistically on
configuration save and never from the browser.
### Limitation: recording_active is global-only

`recording_active` describes the NVR as a whole. The contract carries no
per-channel/per-camera recording signal, so the global flag can never prove that
a specific camera is being recorded. The web UI therefore derives per-camera
recording through one shared helper (`effectiveRecordingState` in
`src/lib/health.ts`) which returns:

- `active` — fresh camera report with `recording = true` AND a fresh NVR
  heartbeat explicitly reporting `recording_active = true`.
- `stopped` — NVR explicitly reports `recording_active = false`, or the camera
  runtime reports it is not recording.
- `unknown` — camera heartbeat stale/offline, or the NVR never reported / is
  stale / `recording_active` is null.

Per-camera `REC` badges are only rendered for `active`; `unknown` is shown as
text rather than claimed as recording. A real per-camera signal (per-channel
recording state in the NVR heartbeat) is required before the UI can attribute
recording to an individual channel with certainty.

## Capability independence

AI detection, camera/source connectivity and recording are three INDEPENDENT
capabilities. The console derives them separately (`systemCapabilities` in
`src/lib/health.ts`):

- **Detection** — `operational` when the AI heartbeat is fresh and at least one
  configured source is online, `idle` when the service runs but no source is
  online, `unavailable` when the AI service never reported or is stale.
- **Recording** — `not_configured` when no NVR (or other recording service) ever
  reported, otherwise derived from the NVR heartbeat only.

A missing NVR never makes the AI heartbeat false and never blocks detection. The
overall `System` indicator stays reserved for "every supported service online",
so it reads `Degraded` when an optional capability is absent; the accompanying
capability subtext states truthfully which capability is missing (for example
"Detection operational · no recording service configured").
