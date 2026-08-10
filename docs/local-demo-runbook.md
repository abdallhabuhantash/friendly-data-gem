# Local Demo Runbook

This guide explains how to run the Vigilant Eye AI service locally using a demo
MP4 video file instead of a physical RTSP camera, and how the data flows from
raw video through to the web UI. It also documents an example local-network
deployment for the graduation demonstration.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- A virtual environment in `ai-service/.venv`
- Dependencies installed: `pip install -r requirements.txt`
- A `.env` file in `ai-service/` copied from `.env.example` with `SUPABASE_URL`,
  `SUPABASE_SERVICE_ROLE_KEY`, and `AI_SERVICE_KEY` filled in
- The web application running (Lovable preview or `npm run dev`)
- Supabase project provisioned with all migrations applied

## Demo Video Setup

1. Place one or more MP4 files on the machine running the AI service.
2. In `ai-service/.env`, set:
   ```
   DEMO_VIDEO_PATH=/absolute/path/to/demo.mp4
   ```
   For multiple cameras, use per-camera mapping:
   ```
   DEMO_VIDEO_PATHS_JSON={"11111111-1111-4111-8111-000000000001":"/path/cam1.mp4"}
   ```
3. Set `DEMO_VIDEO_LOOP=true` so the clip replays continuously.
4. In the web console (Settings page), set **Operation Mode** to **Demo**.
5. Ensure at least one camera row has `source_type = 'demo'` and `ai_enabled = true`.

## End-to-End Data Flow

```
Demo MP4 / RTSP camera
        |
        v
Python + OpenCV (capture_worker.py)
    reads the newest frame
        |
        v
YOLO inference (detector.py)
    detects persons + cell phones
        |
        v
Per-camera tracking (tracker.py)
    ByteTrack IDs isolated per camera
        |
        v
Person-phone association (association.py)
    IoU + containment + distance scoring
        |
        v
Temporal confirmation (temporal_state.py)
    min duration + min frames + cooldown
        |
        v
Event + annotated snapshot (event_publisher.py, snapshot_service.py)
    UUID assigned before any I/O
        |
        v
Supabase insert + storage upload
    events table + snapshots bucket
        |
        v
Vigilant Eye web UI (React + TanStack)
    realtime events, monitoring viewport, review
        |
        v
Optional Telegram notification
    photo + caption, fallback to text
```

### Key safety properties

- Each camera has isolated tracking state. Camera A's track IDs never
  influence Camera B.
- An uncertain association is never reported as "confirmed cheating."
- If the snapshot upload fails but the event insert succeeds, the event is
  preserved with `snapshot_path = null` and a bounded retry updates only that
  column later (never overwriting human review fields).
- If Supabase is unreachable, the event is queued in a local SQLite store and
  retried with exponential backoff.

## Starting the Services

### AI service (Windows)

```powershell
cd ai-service
.\.venv\Scripts\Activate.ps1
python run.py
```

Or use the helper script:

```powershell
.\run_windows.ps1
```

### AI service (Linux/macOS)

```sh
cd ai-service
source .venv/bin/activate
python run.py
```

### Web application

The Lovable preview runs automatically. For local development:

```sh
npm install
npm run dev
```

## Validating the pipeline with no cloud and no hardware

Before wiring anything to the backend, run a local MP4 through the *real*
detection code (same detector, association, temporal confirmation and
annotation the service uses). Nothing is faked and nothing is written to the
cloud or Telegram:

```sh
cd ai-service
python -m app.tools.local_video_check ./samples/demo.mp4
python -m app.tools.local_video_check ./samples/demo.mp4 --save-annotated ./out
```

Useful calibration flags: `--phone-conf`, `--person-conf`, `--assoc-conf`,
`--min-duration`, `--min-frames`, `--every`, `--max-frames`. The tool prints
every confirmed event with its severity, association status and duration, plus
the achieved pipeline FPS. To publish to the backend, run the normal service
(`python run.py`) instead — the tool stays offline on purpose.

## Verifying It Works



1. Open the web application and sign in.
2. Go to **Monitoring**. The demo camera should show an annotated live stream.
3. Go to **Events**. New detection events should appear in real time.
4. Go to **Settings** and confirm the AI health badge is online.
5. If Telegram is configured, check the configured chat for photo messages.

## Example Local-Network Deployment

These are example values for the graduation demonstration where the laptop
running the AI service is connected by Ethernet to the camera/NVR switch while
Wi-Fi stays on normal DHCP for internet access.

| Device            | IP            | Notes                          |
|-------------------|---------------|--------------------------------|
| IP camera          | 10.77.10.100  | Static, same subnet as laptop  |
| NVR               | 10.77.10.110  | Static, same subnet as laptop  |
| Laptop (Ethernet) | 10.77.10.120  | Static, no gateway             |
| Subnet mask       | 255.255.255.0 | /24                            |
| Ethernet gateway  | (blank)       | No default route on Ethernet    |

### Why a static Ethernet with no gateway

The laptop's Wi-Fi stays connected to the normal DHCP network for internet
access (Supabase, Lovable, Telegram). The Ethernet interface is given a static
IP on the camera subnet with **no gateway** so that camera/NVR traffic stays on
the local switch and the default route for internet traffic remains on Wi-Fi.

### These are example values

Replace the IPs with the actual addresses of your hardware. The camera and NVR
must be reachable from the laptop over the Ethernet interface (test with
`ping 10.77.10.100`).

## What Requires the Windows Laptop / Hardware

- Physical RTSP camera or NVR stream validation
- GPU (CUDA) inference performance testing
- Telegram delivery over a real network
- Multi-camera physical isolation test with two real cameras
- Long-running stability test with real hardware

The demo MP4 path validates the full software pipeline (detection, tracking,
association, temporal confirmation, events, snapshots, UI) without any
hardware.
