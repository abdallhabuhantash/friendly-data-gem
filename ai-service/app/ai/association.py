"""Person <-> trigger-object association engine.

Pure, reusable and hardware independent. Given the tracked persons and one
detected phone it decides whether the phone can be attributed to a person,
and how confident that attribution is.

The engine is deliberately conservative: ambiguity yields `uncertain`, never a
definitive person. The web application then downgrades severity accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from ..domain.geometry import (
    BBox,
    containment_ratio,
    contains_point,
    expand,
    normalized_distance,
)
from ..domain.models import AssociationResult, AssociationStatus, Detection

#: Best candidate must beat the runner-up by at least this margin.
DEFAULT_ASSOCIATION_MARGIN = 0.12
#: Minimum score for a candidate to be considered plausible at all.
PLAUSIBLE_SCORE = 0.30
#: How much temporal continuity may boost a candidate.
CONTINUITY_BONUS = 0.08


@dataclass(frozen=True)
class Candidate:
    person_tracking_id: str
    score: float


def _distance_score(distance: float) -> float:
    """Maps a size-normalized distance into 0..1 (1 == on the torso)."""
    if distance <= 0.15:
        return 1.0
    if distance >= 1.2:
        return 0.0
    return max(0.0, 1.0 - (distance - 0.15) / (1.2 - 0.15))


def score_candidate(
    phone: BBox,
    person: BBox,
    *,
    previous_confidence: float = 0.0,
) -> float:
    """Combines several visual heuristics into a single 0..1 score."""
    center = phone.center
    inside = 1.0 if contains_point(person, center) else 0.0
    inside_expanded = 1.0 if contains_point(expand(person, 0.25), center) else 0.0
    overlap = containment_ratio(phone, person)
    proximity = _distance_score(normalized_distance(center, person))

    score = (
        0.34 * inside
        + 0.14 * inside_expanded
        + 0.22 * min(1.0, overlap)
        + 0.30 * proximity
    )
    # Temporal continuity nudges a previously seen pairing but can never
    # promote a candidate that the current frame does not support.
    if previous_confidence > 0 and score > 0:
        score = min(1.0, score + CONTINUITY_BONUS * previous_confidence)
    return round(min(1.0, max(0.0, score)), 6)


def rank_candidates(
    phone: Detection,
    persons: Iterable[Detection],
    *,
    previous: Optional[Mapping[str, float]] = None,
) -> list[Candidate]:
    """Scores every tracked person against one phone, best first."""
    history = previous or {}
    candidates: list[Candidate] = []
    for person in persons:
        if not person.tracking_id:
            continue
        candidates.append(
            Candidate(
                person_tracking_id=person.tracking_id,
                score=score_candidate(
                    phone.bbox,
                    person.bbox,
                    previous_confidence=float(history.get(person.tracking_id, 0.0)),
                ),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def associate(
    phone: Detection,
    persons: Iterable[Detection],
    *,
    association_threshold: float,
    margin: float = DEFAULT_ASSOCIATION_MARGIN,
    previous: Optional[Mapping[str, float]] = None,
) -> AssociationResult:
    """Resolves the association status for a single detected phone."""
    candidates = rank_candidates(phone, persons, previous=previous)
    plausible = [item for item in candidates if item.score >= PLAUSIBLE_SCORE]

    if not plausible:
        return AssociationResult(AssociationStatus.UNASSOCIATED, None, None, None)

    best = plausible[0]
    runner_up = plausible[1].score if len(plausible) > 1 else 0.0

    if best.score >= association_threshold and (best.score - runner_up) >= margin:
        return AssociationResult(
            AssociationStatus.ASSOCIATED,
            best.person_tracking_id,
            best.score,
            runner_up or None,
        )

    # Plausible but ambiguous: never name a definitive person.
    return AssociationResult(
        AssociationStatus.UNCERTAIN,
        None,
        best.score,
        runner_up or None,
    )