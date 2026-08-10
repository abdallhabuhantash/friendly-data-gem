# Notification Contract (Future Phase)

No notification provider is implemented. This document only fixes the payload
and the safety rules any future notifier must follow.

## Payload

```json
{
  "event_title": "Suspicious Cheating Activity",
  "camera": "Hall A Front",
  "effective_severity": "warning",
  "person_tracking_id": "P-14",
  "phone_confidence": 0.91,
  "association_confidence": 0.78,
  "detection_duration_seconds": 4.2,
  "review_required": true,
  "snapshot": "signed-url-or-null"
}
```

| Field | Notes |
| --- | --- |
| `event_title` | Advisory wording only (Suspicious / Possible / Mobile Phone Detected). |
| `camera` | Camera display name. Never a host, RTSP URL or credential. |
| `effective_severity` | Result of the uncertain-association safety rule, not the raw stored severity. |
| `person_tracking_id` | Included **only** when `association_status = "associated"`. Omit otherwise. |
| `phone_confidence` | Trigger-object confidence, 0–1. |
| `association_confidence` | Person↔object association confidence, 0–1, or null. |
| `detection_duration_seconds` | Fractional seconds. |
| `review_required` | Always true until a human reviews the event. |
| `snapshot` | Optional, short-lived signed URL. Never a stored URL, never a public link. |

## Rules

- An uncertain association must not identify a definitive person: drop
  `person_tracking_id` and never phrase the message as an accusation.
- An uncertain association must never be presented as critical; it is
  downgraded to `warning` by the effective-severity rule.
- No camera credentials, RTSP URLs or hosts in any message.
- No API or provider secrets in any message or in the browser.
- Provider tokens live in server-side secrets only.

## Future channels

Telegram, WhatsApp, SMS. None are implemented in this phase; the Settings page
reports each as `Not Configured` and collects no tokens.