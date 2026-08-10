# Camera Integration Contract

Configuration the future Python AI service reads from the `cameras` table.
Credentials are never part of this payload.

```json
{
  "id": "uuid",
  "name": "Hall A Front",
  "location": "Hall A",
  "source_type": "direct_camera",
  "host": "10.77.10.100",
  "rtsp_port": 554,
  "channel": 1,
  "stream_path": "/stream2",
  "stream_profile": "sub",
  "ai_enabled": true,
  "active": true
}
```

## Source types

- `direct_camera` — Python connects directly to the camera's RTSP endpoint.
- `nvr_channel` — Python connects to the NVR's channel RTSP endpoint.
- `demo` — test/demo source only; never real hardware.

`is_demo` is always derived from `source_type`: `demo` sets `is_demo = true`,
every other source type sets `is_demo = false`. Live mode shows only
`is_demo = false` rows.

## Credentials

Camera username/password are never exposed to the browser and never returned by
the Data API. They come from secure local/server configuration on the machine
running the AI service (or the service-role-only `camera_credentials` table).

## Scope

- `active = false` means archived: excluded from monitoring, history retained.
- `ai_enabled = false` means the service must not run inference on that stream.
- Runtime fields (`status`, `last_heartbeat_at`, `recording`, `fps`) are written
  by the service, never by the console.