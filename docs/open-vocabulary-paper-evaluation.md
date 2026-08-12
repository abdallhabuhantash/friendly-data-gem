# Open-Vocabulary Paper Evidence — Evaluation Guide (Task 3E-B)

Status: **experimental, dormant, NOT production ready.**

This subsystem asks an open-vocabulary detector (YOLO-World) whether it sees
loose-paper concepts in a frame. It is not wired into the live runtime, emits no
events, performs no temporal fusion, and adds zero GPU cost to Task 1 phone
detection. It exists so paper evidence can be measured honestly before anyone
decides whether it is trustworthy.

## Truthfulness rules

* Paper is claimed only when a paper-specific prompt fires on a paper-specific
  model. There is no stock/COCO fallback and **`book` is never mapped to paper**.
* Every detection stores the exact prompt that fired (`raw_prompt`) alongside the
  single canonical semantic class `paper`.
* Zero detections means "this model detected no paper evidence" — never "there is
  no paper in the scene".
* Any malformed model output degrades the **whole frame** to a status; partial
  evidence is never kept.
* Reports contain basenames only; absolute paths, credentials and stream URLs are
  never written to reports or logs.

## Prompt semantics

Candidate prompts (`PAPER_PROMPT_CANDIDATES`):

```
paper, sheet of paper, exam paper, paper slip, small paper slip, folded paper
```

`PaperPromptConfig` rejects blank entries, duplicates (case/whitespace
insensitive), non-paper concepts (`document`, `card`, `white rectangle`), and
explicitly prohibited fallbacks (`book`, `notebook`, `folder`, `phone`, `pen`,
`hand`, `desk`, `object`, `thing`).

Prompt wording changes what the model looks for, so treat each wording as a
separate configuration and measure it separately. Repeat `--prompts` to compare
configurations in a single pass over the same footage.

## Frame statuses

| Status | Meaning |
| --- | --- |
| `ok` | Model ran; detections (possibly zero) are trustworthy as model output. |
| `model_unavailable` | Weights missing or failed to load. |
| `prompt_configuration_invalid` | Checkpoint is closed-vocabulary, or prompts could not be applied. |
| `inference_failed` | The model raised during inference. |
| `malformed_result` | Output shape/values were not parseable; frame discarded. |
| `model_schema_mismatch` | Custom checkpoint lacks the `paper` class. |

## Running an offline evaluation

Everything is explicit — the script chooses no thresholds for you.

```bash
cd ai-service
python scripts/evaluate_paper_open_vocab.py \
  --source samples/exam_clip.mp4 \
  --weights models/yolov8s-worldv2.pt \
  --prompts "paper" "sheet of paper" "exam paper" \
  --device cpu --imgsz 960 --confidence 0.15 --frame-stride 5 \
  --json-out paper-eval-results/run1.json \
  --annotated-out paper-eval-results/run1.mp4
```

Optional offline crop experiment (person-region cropping is an experiment only,
never a runtime behaviour):

```bash
  --crop 0.25 0.30 0.75 0.95
```

Outputs:

* console summary — descriptive counts, confidence distribution, latency, FPS;
* JSON report — per-frame statuses and every detection with its `raw_prompt` and
  an empty `review` slot for a human verdict;
* optional annotated video — each box labelled with its firing prompt and score.

## What is measured (and what is not)

Reported: frames sampled, frames with detections, total detections, detections
per prompt, frame statuses, confidence mean/median/p95/max, latency mean/p95,
processing FPS.

Deliberately **not** reported: precision, recall, accuracy, mAP or F1. No
labelled ground-truth dataset exists, so any such number would be fiction. Once
labelled data exists, add a separate labelled evaluator instead of inventing
numbers here.

## Manual review workflow

1. Run the evaluator over your clip with the annotated output enabled.
2. Watch the annotated video and open the JSON report.
3. For each detection, set `review` to `true_paper`, `false_positive` or
   `uncertain`. This is a human judgement; the model never fills it in.
4. Keep the reviewed JSON next to the clip so decisions stay auditable.

## Scenes you must test before trusting anything

Negative (must stay quiet): empty desk; hands only; handshake / hands near each
other; phone; pen or pencil; notebook or book; clothing; white desk surface;
monitor or tablet screen; printed signs or background posters.

Positive (must be measured): normal exam sheet; small slip; folded sheet;
partially occluded sheet; one hand holding paper; two hands holding paper; paper
moving between two people; paper on desk; paper angled to camera; distant or
small paper; motion blur.

Expect open-vocabulary weakness on small slips, folded paper, strong angles,
distance, motion blur, and bright white desks. Record what you observe rather
than tuning until the numbers look pleasing.

## Boundaries

* No temporal fusion, no handoff/exchange claim, no event emission, no alerting.
* No Task 1, Task 2 or Task 3D file is modified or imported by this subsystem.
* Nothing here runs on the live pipeline; enabling it is a future, separate,
  explicit decision.
