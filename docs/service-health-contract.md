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