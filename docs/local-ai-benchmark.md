# Local AI runtime benchmark (Windows)

Measures the real Task 1 phone-detection runtime by itself, then Task 1 with the
asynchronous Pose runtime enabled, over the **same** local video file. It is
measurement tooling only: no behaviour, no regions, no events, and no
pass/fail threshold is applied. A physical camera is not required.

## 1. Verify the machine

```powershell
python --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

The GPU model is read at runtime (`torch.cuda.get_device_name`) and never
hard-coded. If CUDA is unavailable the benchmark still runs and reports CPU
execution truthfully.

## 2. Install the service requirements

```powershell
cd ai-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# For GPU inference install a CUDA build of torch, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 3. Select a benchmark video

Place an MP4 in `ai-service/samples/` (videos are not committed). Both modes use
the same file, the same detector model/imgsz and the same Task 1 thresholds.

## 4. Baseline: Task 1 only

```powershell
python -m scripts.benchmark_runtime --video .\samples\demo.mp4 --mode baseline `
  --warmup-frames 30 --max-measured-frames 300 `
  --output benchmark-results\pose-runtime-benchmark.json
```

## 5. Task 1 + asynchronous Pose

Pose settings are explicit: there is no calibrated default for a model, device,
input size, confidence floor or cadence.

```powershell
python -m scripts.benchmark_runtime --video .\samples\demo.mp4 --mode both `
  --warmup-frames 30 --max-measured-frames 300 `
  --pose-model yolo11n-pose.pt --pose-device cuda:0 --pose-imgsz 640 `
  --pose-confidence 0.30 --pose-max-fps 2 `
  --output benchmark-results\pose-runtime-benchmark.json
```

`--mode both` runs the baseline first, then the Pose-enabled run, and prints the
comparison.

## 6. Inspect the result

The console prints a summary; the machine-readable report is written to the
`--output` path (default `benchmark-results/pose-runtime-benchmark.json`), which
is git-ignored and never committed.

The JSON report contains:

- `timestamp`, `benchmark_version`
- `configuration` (file names only, never absolute user paths)
- `hardware` (platform, torch version, CUDA availability/version, GPU name)
- `baseline` and `with_pose`: analysed frames, measured elapsed seconds,
  measured Task 1 FPS, detector latency (count/mean/median/p95/max)
- `with_pose.pose`: submitted, processed, replaced pending, stale discards,
  provider failures, association degraded, cadence skipped, pose inference
  latency and measured pose FPS
- `baseline_cuda_memory` / `with_pose_cuda_memory`: peak allocated/reserved
  CUDA memory when measurable, otherwise `null`
- `comparison`: Task 1 FPS with/without Pose, absolute difference, percentage
  change, and detector mean/median/p95 changes

Unavailable measurements are reported as `null` — zero is never used as a
stand-in for a missing measurement.

No credentials, RTSP URLs, service keys, tokens or absolute secret paths are
included in the report or in the console output.
