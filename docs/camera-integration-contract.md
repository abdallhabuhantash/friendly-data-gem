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
## Source-generic, multi-camera architecture

Vigilant Eye is a generic multi-camera platform. The software supports zero, one
or many configured cameras; nothing in the schema, the web UI or the AI service
is aware of a camera count, a channel label or a fixed IP address.

- **Direct RTSP is supported.** A `direct_camera` row plus locally held
  credentials is enough for the AI service to capture and run inference.
- **NVR is optional for AI detection.** No NVR object, heartbeat or API is
  required to process a direct RTSP source. NVR health is a separate capability.
- **An NVR may be used** for centralised recording and/or as an RTSP stream
  source (`nvr_channel`). Both resolve through the same source builder, so
  inference only cares about the resolved stream, not the vendor hardware type.
- **Network switch/PoE infrastructure is a deployment concern**, not a software
  requirement. Supported topologies include: a single IP camera direct to the AI
  host; several IP cameras behind a switch; NVR-channel RTSP into the AI service;
  and cameras recorded by an NVR while the AI service consumes suitable streams.
- **The graduation demonstration hardware count does not define software
  limits.** The initial physical validation (one IP camera, one Windows laptop,
  direct RTSP, no NVR) is only one example validation configuration — there is no
  "single-camera mode" and no code path special-cases it.
- **Credentials never reach the browser.** RTSP URLs are built server-side or in
  the AI service from the local secret file / service-role credential provider;
  browser-visible camera rows carry non-secret metadata only.
