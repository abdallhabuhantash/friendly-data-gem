# Custom Paper / Exam-Sheet Detector — Training Guide (Task 3E)

This document defines what must exist before Vigilant Eye may claim that a
**paper** was seen. Task 3E delivers only the *foundation*: an immutable
evidence contract, a strict provider, a crop transform and a manual training
utility. There is currently **no trained paper model**, therefore the system
currently reports `MODEL_UNAVAILABLE` and never claims paper.

Paper evidence is **object evidence only**. It never means transfer, handoff,
exchange, cheating or suspicion. Fusion with the Task 3D temporal handoff layer
is deliberately not implemented.

---

## 1. Why stock COCO is insufficient

The stock YOLO/COCO class list contains no loose-paper class. It has `book`,
`laptop`, `cell phone`, `remote` and similar objects, none of which describe a
loose exam sheet. Using any of them as a stand-in would produce detections that
are literally false: a system that reports "paper" when the model actually
recognised "book" is lying about its evidence.

## 2. Why book is not paper

A book/notebook is a bound, multi-page, thick object with a spine. A loose sheet
is thin, flexible, often bright, frequently folded and usually handled very
differently. Their appearance, aspect ratio, deformation and handling all
differ. Mapping `book` → `paper` would:

* invent evidence that the model never produced,
* make every downstream exchange decision untrustworthy,
* be impossible to audit after the fact.

**A COCO `book` → `paper` mapping is forbidden and does not exist in the code.**
If a bound book/notebook ever needs to be recognised, it becomes its own
separate class with its own labelled data.

## 3. What the custom `paper` class means

One single generic semantic class: `paper`. It covers a *loose sheet a person
could hold, read or pass*:

* full exam sheet / answer sheet
* A4 / Letter-like loose sheet
* small paper slip / cheat sheet
* folded loose paper
* partially visible loose paper (visible extent only)

It explicitly excludes: `book`, `notebook`, `phone`, `pen`/`pencil`, `hand`,
`desk`, folders, tablets and screens. No aliases exist.

## 4. Dataset collection requirements

Positive examples (each must be genuinely represented, not a token handful):

* full exam sheet
* A4/Letter-like sheet
* small paper slip
* folded paper
* partially occluded paper
* paper held by one hand
* paper held by two hands
* paper moving between two people
* white paper on a light desk (low contrast)
* coloured paper
* angled / foreshortened paper
* far / small paper
* motion blur
* low light
* different camera heights and viewing angles

Negative / hard-negative examples (no paper label at all):

* hands with no paper
* phones (including a bright white phone/tablet screen)
* pens / pencils
* books / notebooks
* folders (if outside the class)
* desk edges and light desk surfaces
* clothing (white shirts, sleeves, cuffs)
* answer booklets, if intentionally excluded
* handshakes
* two students reaching toward each other
* a teacher walking between students

Capture with the **real deployment cameras, heights and lighting** wherever
possible. A dataset recorded only from a phone at desk level does not predict
ceiling-camera performance.

## 5. Annotation rules

* Tight bounding box around the **visible paper pixels** only.
* Partially visible paper: annotate the visible extent only.
* Heavily occluded paper: label only when enough paper is genuinely visible.
* Never guess or fabricate hidden extent behind hands, bodies or desks.
* No annotation at all when the paper is completely invisible.
* Folded paper: treat consistently across the whole dataset (annotate the
  visible folded shape as one `paper` instance).
* Minimum visible-size policy must be **established from real footage** and then
  documented here; do not invent a pixel threshold up front.
* Ambiguous frames should be discarded rather than guessed.

## 6. Train / validation / test split discipline

Three disjoint sets are mandatory:

1. **training recordings**
2. **validation recordings** (threshold and epoch selection)
3. **final held-out test recordings** (used once, for reporting only)

Rules:

* Split **by recording / session**, never by random frame sampling.
* The final held-out test set must **never** be used to tune thresholds,
  augmentation, confidence or epochs.
* Diversify students, clothing, backgrounds, camera views and lighting across
  splits where possible.

## 7. Leakage avoidance

Adjacent frames of one short clip are nearly identical images. If they are
scattered across train and validation/test, the model is effectively evaluated
on its own training data and metrics become meaningless.

* Frames from the SAME video sequence must stay entirely inside ONE split.
* Prefer different sessions / days / rooms per split.
* **No evaluation claim is valid if train/test leakage exists.**

## 8. Small-object reality

In surveillance footage a paper slip may occupy very few pixels. Model
feasibility depends on the *actual apparent pixel size* of the paper, not on
wishful thinking. Detection at arbitrary distance is not claimed.

Calibration must measure, from real footage:

* bbox pixel width/height distribution of paper instances,
* detection recall versus apparent size,
* which camera distances/angles are suitable at all.

If paper occupies too few pixels, the situation is **unsupported by the current
camera placement** — not an "AI failure". Document the supported range instead
of over-promising.

## 9. Evaluation metrics

Every future model review must report, from the held-out test split:

* precision
* recall
* mAP50
* mAP50-95
* false positives on the hard-negative set (per hard-negative category)
* recall for small paper (by apparent-size bucket)
* recall under partial occlusion

There is **no "100% accuracy" target**, and acceptance thresholds are
deliberately NOT hard-coded anywhere in the code. They will be chosen only after
real dataset results exist.

## 10. Dataset YAML contract

The dataset YAML must declare `train`, `val` and `names`, must contain a real
non-empty validation split, and `names` must contain the canonical `paper`
class. A `test` entry is required for the final held-out evaluation.

```yaml
train: images/train
val: images/val
test: images/test
names:
  0: paper
```

## 11. Training and evaluation commands

Training (manual only — nothing trains during tests, import or startup):

```bash
cd ai-service
python scripts/train_paper_detector.py train \
  --data datasets/paper/paper.yaml \
  --base-weights yolov8s.pt \
  --device 0 \
  --imgsz 960 \
  --epochs 100 \
  --batch 16 \
  --project training-runs/paper \
  --name paper_v1
```

Validation-split evaluation (threshold selection allowed):

```bash
python scripts/train_paper_detector.py validate \
  --data datasets/paper/paper.yaml \
  --weights training-runs/paper/paper_v1/weights/best.pt \
  --device 0 --imgsz 960 --split val
```

Final held-out test evaluation (report only, run once):

```bash
python scripts/train_paper_detector.py validate \
  --data datasets/paper/paper.yaml \
  --weights training-runs/paper/paper_v1/weights/best.pt \
  --device 0 --imgsz 960 --split test
```

Training runs and datasets are gitignored; no credentials, secrets, or absolute
private paths may appear in committed reports.

## 12. How to report real results back for review

Report, in plain text:

* dataset size per split and per positive/negative category,
* how splits were separated (which recordings/sessions),
* explicit confirmation that no clip spans two splits,
* the exact training command used,
* the metrics from section 9,
* apparent paper pixel-size distribution,
* failure examples (false positives on hard negatives, missed small paper).

## 13. Production readiness

The paper detector is **NOT production-ready** until:

1. a real labelled dataset satisfying sections 4–7 exists,
2. training completed with the documented command,
3. held-out test metrics are reported and reviewed,
4. acceptance thresholds are chosen from those real results.

Until then the provider returns `MODEL_UNAVAILABLE` / `MODEL_SCHEMA_MISMATCH`
and the system truthfully claims nothing about paper.
