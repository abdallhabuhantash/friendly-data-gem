# Exam Session / Anonymous Subject Identity Contract

This document is the non-negotiable contract for the Exam Session workflow in
Vigilant Eye. It describes what identity means in this system, and — more
importantly — what the system must never claim. Later tasks (AI session-subject
runtime, event identity resolution, Start Exam Session) are built on top of
this contract and must not contradict it.

## 1. Students are not application users

Application `Users` are system operators: administrators and operators. They
sign in, review events and configure the platform.

Students are **not** application users. They have no account, no role, no
session and no authentication. They exist only as roster records attached to an
exam session (`exam_roster_students`).

## 2. The roster stores real student records

An exam roster row stores exactly two domain facts:

- `full_name` — the real student name
- `university_id` — the university / student identifier

Both are required. `university_id` is unique **within one exam session**; the
same student ID may legitimately appear in many different exam sessions through
separate roster records.

## 3. A future AI Exam Subject (e.g. `S017`) is not an identity

A session subject label such as `S001`, `S002`, `S017` is a stable *anonymous*
handle for "a person the AI has been tracking inside this exam session". It is
explicitly **not**:

- a university ID
- a student name
- a raw YOLO / tracker ID

## 3b. A subject number is immortal, owned and mobile

Once `S017` has been assigned to a physical person inside an exam session:

- it is **never renumbered**, never transferred to another person and never
  reused — not after a tracker id change, not after a stream restart, not after
  the person is lost entirely;
- it belongs to **exactly one** person, and that person has exactly one number,
  so a raw tracking id is owned by at most one subject at a time;
- it is **not** tied to a seat, a desk, a place or a bounding box. Standing up,
  walking across the hall and sitting somewhere else are normal and cost the
  subject nothing.

Existence and binding are two separate facts, and are stored separately:

```text
lifecycle    ACTIVE -> TEMPORARILY_LOST -> LOST -> ENDED   (existence)
association  CONFIRMED / PROVISIONAL / UNRESOLVED / CONFLICT (raw-track binding)
```

A subject is **never** ended by a timeout. `LOST` means "not observed now, number
still reserved"; only the end of the exam session produces `ENDED`. Numbers are
allocated by the database (`allocate_session_subject_number`), so numbering stays
atomic, monotonic and unique per session across cameras and service instances,
and database triggers reject any attempt to renumber, re-parent or delete a
subject while its session exists.

Ambiguity is preserved instead of resolved: an impossible jump or a duplicated
raw id becomes `CONFLICT`, two equally plausible owners stay `UNRESOLVED`, and a
raw track that plausibly continues a lost subject is held as `UNRESOLVED` rather
than being given a second number. Recovery evidence is geometry and motion only
(predicted position, overlap, plausible speed) — never appearance, clothing
colour, face or any biometric signature.

## 4. Raw tracker identity is unstable

Raw tracking IDs are re-assigned, lost and recreated during an exam whenever
tracking breaks. Session-subject identity must therefore sit **above** raw
tracking as a separate layer:

```text
Raw AI tracking ID
        ↓
Stable anonymous Exam Session Subject   (S001, S002, S003, …)
        ↓
Optional real student identity          (resolved only when needed)
```

## 5. No facial recognition, no biometrics

The platform performs no face recognition, face embedding, gait, or any other
biometric identity matching — for students or for staff.

## 6. No mandatory physical seat registration

There are no seat maps, seat calibration, seat assignments, or seating order
requirements. Students may sit anywhere in the hall. The database intentionally
contains no halls/seats dependency; `location_label` on an exam session is
optional free text only.

## 7. Identity must never be guessed from visual proximity

A subject must never be resolved to a roster student because of where the
person sits, who they sit near, or any visual similarity heuristic.

## 8. Resolution is manual, on demand, and only when needed

Future event review will allow a human to manually resolve an anonymous
subject to a roster student. Real student identity normally stays unresolved.
Nothing in the system requires resolution for monitoring to work.

## 9. Uncertainty is preserved truthfully

When identity is unknown or uncertain, the system stores and displays
`UNKNOWN` / `UNRESOLVED`. It never substitutes a best guess, a placeholder
name, or demo data.

## 10. Paper Exchange stays advisory

Paper-exchange findings are advisory evidence. User-facing wording remains
"Possible Paper Exchange" (or equivalent hedged phrasing). The system never
states "Confirmed Cheating"; only a human reviewer changes an event's review
status.

## 11. Initial paper distribution is outside armed monitoring

Handing out exam papers at the start of an exam looks exactly like a paper
exchange. Paper-exchange monitoring must therefore be **unarmed** during
distribution. A future explicit "Start Exam Session" action is responsible for
arming monitoring from a clean state, after distribution is complete.

## 12. Live functionality is independent of identity

Live view, detection, events and review all function fully with every subject
unresolved. Identity resolution is an optional enrichment layer, never a
prerequisite.

## Status

Implemented: exam sessions, optional session→camera links, invigilator session
metadata, roster records, manual roster entry, spreadsheet roster import, the
Exam Sessions UI, and the immutable anonymous subject registry (creation,
exclusive raw-track ownership, mobility-aware short-gap recovery, lifecycle vs
association reporting, atomic per-session numbering, and anonymous labels on the
annotated stream, where an unowned raw track is drawn as `UNRESOLVED`).

Deliberately **not** implemented here: event identity resolution,
`event_subjects`, subject thumbnails, Locate Subject, recording/clips, paper
detector runtime integration, seats, QR check-in, facial recognition, and the
Start Exam Session runtime action.
