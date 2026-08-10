# AI Event Contract (Python service → Lovable)

The Python AI service is the only producer of events. It writes rows to
`public.events` with the service role key. The web app never infers AI data
from free text — the reviewer `note` is for humans only.

## Event row

| Column | Type | Meaning |
| --- | --- | --- |
| `type` | text | Generic identifier, e.g. `suspicious_cheating_activity`, `possible_cheating_activity`, `mobile_phone_detected`. Unknown values render humanised. |
| `severity` | text | `critical` \| `warning` \| `info` |
| `status` | text | Human review lifecycle: `new` → `under_review` → `confirmed` \| `rejected`. Never set by AI beyond `new`. |
| `camera_id` / `camera_name` | uuid / text | Source camera. |
| `rule_id` | uuid | Rule that produced the event. |
| `person_tracking_id` | text | Temporary tracker ID. Not an identity. |
| `trigger_object_class` | text | e.g. `cell_phone`. Generic, not phone-specific in schema. |
| `trigger_confidence` | numeric | 0–1 confidence of the trigger object. |
| `association_status` | text | `associated` \| `uncertain` \| `unassociated` \| `not_applicable`. Independent of review status. |
| `association_confidence` | numeric | 0–1, null when not applicable. |
| `detection_duration_seconds` | numeric | Fractional seconds of persistence. |
| `detection_frame_count` | int | Matching frames observed. |
| `evidence` | jsonb array | See below. |
| `source_mode` | text | `live` \| `demo`. Demo rows are hidden in live mode. |
| `snapshot_path` | text | Object path inside the private `snapshots` bucket. |

## Evidence item

```json
{
  "object_id": "phone-17",
  "tracking_id": null,
  "class_name": "cell_phone",
  "confidence": 0.88,
  "bbox": { "x": 0.505, "y": 0.68, "width": 0.055, "height": 0.1 },
  "role": "trigger_object",
  "associated_person_tracking_id": "03",
  "association_confidence": 0.91
}
```

`bbox` values are normalized 0–1 relative to the analysed frame, never pixels,
so overlays scale with any viewport size. `role` is `person`,
`trigger_object`, or any future role.

## Two evidence levels

The `mobile_phone_detection` engine produces two distinct, independent kinds of
evidence. Both are ordinary rows in `public.events`; no extra column is needed.

### 1. Instant visible-phone evidence

A phone that is visible for only a very short time — possibly a single analysed
frame — is preserved instead of being discarded:

- `type` = `mobile_phone_detected`, `severity` = `warning`, `status` = `new`
- `detection_frame_count` = `1` for a true one-frame trigger
- `detection_duration_seconds` = `0.0` (or the truthful measured value)
- `association_status` / `association_confidence` = the real result for that
  frame; `person_tracking_id` is present only when the association engine is
  definitive
- `evidence` and `snapshot_path` come from the exact triggering frame

It requires `instant_confidence_threshold`, a high-confidence single-frame
threshold (default `0.85`) that is never weaker than `confidence_threshold` —
equality is allowed and the invariant is enforced in the UI, in the database
(`ai_rules_instant_threshold_not_weaker`) and in `RuleConfig` — because a
single frame has no temporal corroboration. It means only "a mobile phone was
visibly detected" — never confirmed cheating, never `critical`.

Instant events are deduplicated per (camera, rule, subject) using the rule's
`cooldown_seconds`, in state fully separate from temporal confirmation. The
subject is the definitively associated person track when available, otherwise
the phone track, otherwise a deterministic short-lived spatial signature of the
phone box. Detection array position is never used as an identity.

### 2. Temporally confirmed activity

Unchanged: `min_duration_seconds` **and** `min_matching_frames` **and**
association decide whether `possible_cheating_activity` or
`suspicious_cheating_activity` is justified. An earlier instant warning never
resets, consumes or suppresses this escalation, so the same phone may yield one
instant warning and, later, one stronger temporal event.

Each analysed frame counted in `detection_frame_count` is a **distinct captured
frame**; a frozen stream cannot satisfy `min_matching_frames` by being
reprocessed.

## Alert semantics

- `associated` + duration ≥ rule threshold → `suspicious_cheating_activity`.
- `uncertain` → at most `possible_cheating_activity`; the UI presents it as
  "Mobile Phone Detected · Uncertain Person Association".
- `unassociated` → `mobile_phone_detected` only.
- No payload may ever assert confirmed cheating.

## Rule configuration read by the service

`ai_rules`: `confidence_threshold` (trigger object),
`person_confidence_threshold`, `association_confidence_threshold`,
`min_duration_seconds` (numeric), `min_matching_frames`, `cooldown_seconds`,
`require_person_association`, `save_snapshot`, `sound_notification`,
`instant_detection_enabled`, `instant_confidence_threshold`.
Camera scope comes from `ai_rule_cameras`.

## Service health heartbeats (`service_health`)

Rows for `ai` and `nvr` seeded during the prototype are demonstration
placeholders and are flagged `is_demo = true`. A real Python AI service or NVR
heartbeat writer MUST upsert its row with `is_demo = false` explicitly:

```json
{ "service": "ai", "online": true, "is_demo": false, "payload": { "...": "..." } }
```

Rows left with `is_demo = true` are never treated as live hardware and are
hidden while the system runs in Live mode.

## Event source mode

Every event written by the real service MUST set `source_mode = 'live'`.
Only prototype/demonstration rows use `source_mode = 'demo'`; the UI queries
events strictly by the active operation mode, so a mislabelled row is invisible
in the other mode.

## NVR / service heartbeat: recording state

The console never claims recording is happening unless the heartbeat says so.

`service_health.payload` for the `nvr` service may include:

```json
{ "model": "Generic NVR", "recording_active": true }
```

- `recording_active: true` → the UI shows `REC` and `NVR: Recording`.
- `recording_active: false` → `NVR: Online`, and `No REC`.
- field omitted, or heartbeat stale/never reported → recording state is unknown;
  the UI shows `No REC` and never displays a recording indicator.

## Camera configuration fields

Cameras are configured in the console (`source_type`, `host`, `rtsp_port`,
`channel`, `stream_path`, `stream_profile`, `active`). Credentials are NOT
stored in these fields — the AI service resolves the username/password from its
own local configuration or from `camera_credentials` using the service role.
Archived cameras (`active = false`) must be ignored by the AI service.
