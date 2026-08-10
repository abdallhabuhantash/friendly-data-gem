import type { DetectionEvidence, DetectionOverlay } from "@/types";

/**
 * Converts real, persisted normalized evidence (0–1 bounding boxes produced by
 * the Python AI service) into percentage-based viewport overlays.
 *
 * This module contains no sample or invented detections: every overlay it
 * returns is derived from evidence rows that actually exist.
 */
export function overlaysFromEvidence(
  evidence: DetectionEvidence[],
  alert: boolean,
): DetectionOverlay[] {
  return evidence.map((item) => ({
    objectId: item.objectId,
    trackingId: item.trackingId,
    className: item.className === "person" ? "person" : "cell_phone",
    confidence: item.confidence,
    x: item.bbox.x * 100,
    y: item.bbox.y * 100,
    width: item.bbox.width * 100,
    height: item.bbox.height * 100,
    associatedPersonId:
      item.associatedPersonTrackingId === null
        ? null
        : (evidence.find((candidate) => candidate.trackingId === item.associatedPersonTrackingId)
            ?.objectId ?? null),
    associationConfidence: item.associationConfidence,
    alertState:
      item.associationConfidence !== null && item.associationConfidence < 0.65
        ? "uncertain"
        : alert && item.role === "trigger_object"
          ? "alert"
          : item.className === "person" && alert
            ? "alert"
            : "normal",
  }));
}
