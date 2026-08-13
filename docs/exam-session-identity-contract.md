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

## Status of this foundation task

Implemented: exam sessions, optional session→camera links, invigilator session
metadata, roster records, manual roster entry, spreadsheet roster import, and
the Exam Sessions UI.

Deliberately **not** implemented here: session-subject runtime (`S001`/`S002`
creation), tracking→subject reassociation, event identity resolution,
`event_subjects`, subject thumbnails, Locate Subject, recording/clips, paper
detector runtime integration, seats, QR check-in, facial recognition, and the
Start Exam Session runtime action.
