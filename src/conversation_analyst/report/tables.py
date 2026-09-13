"""Tidy output tables.

The primary artifact is long format -- one row per session, person and
measure -- because that is what mixed-effects models want, and dyadic data
must be modeled with a random effect for the pair. A wide pivot is written
alongside for inspection, but the long table is the one to analyze.

Unavailable measures are written as rows with a null value and a stated
reason, never dropped and never zero-filled. Deleting them would make a
missing camera look like an absence of behavior, and in a table of
per-dyad values that difference is the whole analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from conversation_analyst.context import AnalysisContext
from conversation_analyst.measures.base import MeasureValue, registry


def measures_long(
    session_id: str,
    values: Sequence[MeasureValue],
    metadata: dict | None = None,
) -> pd.DataFrame:
    """One row per (session, person, measure)."""
    metadata = metadata or {}
    rows = []
    for value in values:
        spec = registry.spec(value.id) if value.id in registry else None
        rows.append(
            {
                "session_id": session_id,
                "person": value.person or "dyad",
                "level": value.level,
                "family": spec.family if spec else "",
                "measure": value.id,
                "value": value.value,
                "unit": spec.unit if spec else "",
                "n": value.n,
                "available": value.available,
                "unavailable_reason": value.unavailable_reason or "",
                **{f"meta_{k}": v for k, v in metadata.items()},
            }
        )
    return pd.DataFrame(rows)


def measures_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per (session, person)."""
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot_table(
        index=["session_id", "person"], columns="measure", values="value", dropna=False
    ).reset_index()
    wide.columns.name = None
    return wide


def turns_table(session_id: str, context: AnalysisContext) -> pd.DataFrame:
    """One row per floor-holding turn, with its text and timing."""
    if context.turn_set is None:
        return pd.DataFrame()
    rows = []
    for turn in context.turn_set.turns:
        rows.append(
            {
                "session_id": session_id,
                "turn_index": turn.index,
                "person": turn.person,
                "start_s": round(turn.start, 3),
                "end_s": round(turn.end, 3),
                "duration_s": round(turn.duration, 3),
                "speech_s": round(turn.speech_duration, 3),
                "n_ipus": len(turn.ipus),
                "n_words": turn.n_words,
                "fto_s": None if turn.fto is None else round(turn.fto, 3),
                "prev_person": turn.prev_person,
                "overlap_onset": turn.is_overlap_onset,
                "text": turn.text,
            }
        )
    return pd.DataFrame(rows)


def events_table(session_id: str, context: AnalysisContext) -> pd.DataFrame:
    """One row per discrete event, with its provenance."""
    rows: list[dict] = []

    def add(kind: str, person: str | None, start: float, end: float, detail: str = "",
            source: str = "") -> None:
        rows.append(
            {
                "session_id": session_id,
                "event": kind,
                "person": person or "dyad",
                "start_s": round(float(start), 3),
                "end_s": round(float(end), 3),
                "duration_s": round(float(end - start), 3),
                "detail": detail,
                "source": source,
            }
        )

    if context.turn_set is not None:
        for unit in context.turn_set.backchannels:
            add("backchannel", unit.person, unit.start, unit.end, unit.text, "audio+asr")
        for event in context.turn_set.interruptions:
            add(
                event.kind, event.interrupter, event.time,
                event.time + event.overlap_duration,
                f"{'successful' if event.successful else 'unsuccessful'}; "
                f"interrupted {event.interrupted}",
                "attribution",
            )

    if context.face:
        for person, signals in context.face.items():
            # Nods carry their length, magnitude and role into the events
            # table, so a study can model nods individually -- length by
            # role, magnitude by partner behaviour -- rather than only the
            # per-person aggregates in measures.csv.
            for event in getattr(signals, "nod_track", None) or ():
                add(
                    "nod", person, event.start, event.end,
                    f"{event.cycles} cycle{'s' if event.cycles != 1 else ''} "
                    f"({event.length_label}); {event.magnitude_deg:.1f} deg; "
                    f"{event.frequency_hz:.1f} Hz; while {event.role}",
                    "face",
                )
            for event in getattr(signals, "shake_track", None) or ():
                add(
                    "head_shake", person, event.start, event.end,
                    f"{event.cycles} cycle{'s' if event.cycles != 1 else ''}; "
                    f"{event.magnitude_deg:.1f} deg; while {event.role}",
                    "face",
                )
            for start, end in signals.smiles:
                add("smile", person, start, end, source="face")

    if context.laughter:
        for person, segments in context.laughter.items():
            for start, end in segments:
                add("laughter", person, start, end, source="yamnet")

    if context.semantics is not None:
        for callback in context.semantics.callbacks:
            add(
                "callback", callback.person, callback.time, callback.time,
                f"turn {callback.callback_turn} <- {callback.source_turn} "
                f"(lag {callback.lag}); anchors: {', '.join(callback.anchors)}",
                "semantics",
            )
        for topic in context.semantics.topics:
            add("topic", topic.initiator, topic.start, topic.end,
                f"turns {topic.start_turn}-{topic.end_turn}", "semantics")

    frame = pd.DataFrame(rows)
    return frame.sort_values("start_s").reset_index(drop=True) if not frame.empty else frame


def nods_table(session_id: str, context: AnalysisContext) -> pd.DataFrame:
    """One row per nod, with its length, size and role as columns.

    The events table has nods too, but with their properties packed into a
    text description. This is the same data in a shape a model can consume:
    predicting nod length from role, or magnitude from what the partner was
    doing, needs `cycles` and `role` as columns rather than as prose.
    """
    rows: list[dict] = []
    for person, signals in (context.face or {}).items():
        for kind, attribute in (("nod", "nod_track"), ("head_shake", "shake_track")):
            for event in getattr(signals, attribute, None) or ():
                rows.append(
                    {
                        "session_id": session_id,
                        "person": person,
                        "kind": kind,
                        "start_s": round(float(event.start), 3),
                        "end_s": round(float(event.end), 3),
                        "duration_s": round(float(event.duration), 3),
                        "cycles": int(event.cycles),
                        "length_class": event.length_label,
                        "half_cycles": int(event.half_cycles),
                        "magnitude_deg": round(float(event.magnitude_deg), 3),
                        "frequency_hz": round(float(event.frequency_hz), 3),
                        "role": event.role,
                        "face_coverage": round(float(signals.coverage), 3),
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.sort_values("start_s").reset_index(drop=True) if not frame.empty else frame


def timeline_table(session_id: str, context: AnalysisContext) -> pd.DataFrame:
    """Frame-level signals, for re-analysis outside this package.

    This is the substrate an analyst needs to compute a measure nobody
    thought of yet, without re-running the expensive stages.
    """
    n = context.n_frames
    data: dict[str, np.ndarray] = {
        "t_s": context.frame_times(),
    }

    if context.attribution is not None:
        state = context.attribution.state
        data["speaker_state"] = _fit(state, n)
        data["attribution_confidence"] = _fit(context.attribution.confidence, n)

    for person in context.persons:
        speech = context.speech(person)
        if len(speech):
            data[f"speech_{person}"] = speech.to_mask(n, context.frame_hz).astype(np.int8)
        if context.prosody and person in context.prosody:
            track = context.prosody[person]
            data[f"f0_{person}"] = _fit(track.f0_hz, n)
            data[f"intensity_{person}"] = _fit(track.intensity_db, n)
        if context.face and person in context.face:
            signals = context.face[person]
            data[f"smile_{person}"] = _fit(signals.smile, n)
            data[f"head_pitch_{person}"] = _fit(signals.head_pitch, n)
            data[f"head_yaw_{person}"] = _fit(signals.head_yaw, n)
            data[f"gaze_on_partner_{person}"] = _fit(
                signals.on_partner.astype(np.int8), n
            )
            data[f"face_tracked_{person}"] = _fit(signals.tracked.astype(np.int8), n)
            data[f"expressivity_{person}"] = _fit(signals.expressivity, n)
        if context.body and person in context.body:
            data[f"wrist_speed_{person}"] = _fit(context.body[person].wrist_speed, n)

    frame = pd.DataFrame(data)
    frame.insert(0, "session_id", session_id)
    return frame


def _fit(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values)
    if values.size == n:
        return values
    if values.size > n:
        return values[:n]
    pad = np.full(n - values.size, np.nan, dtype=float)
    return np.concatenate([values.astype(float), pad])


def write_session_tables(
    workspace,
    session_id: str,
    context: AnalysisContext,
    values: Sequence[MeasureValue],
) -> dict[str, Path]:
    """Write every per-session table; returns the paths written."""
    written: dict[str, Path] = {}

    long = measures_long(session_id, values, context.metadata)
    path = workspace.table("measures.csv")
    long.to_csv(path, index=False)
    written["measures"] = path

    wide = measures_wide(long)
    if not wide.empty:
        path = workspace.table("measures_wide.csv")
        wide.to_csv(path, index=False)
        written["measures_wide"] = path

    for name, frame in (
        ("turns", turns_table(session_id, context)),
        ("events", events_table(session_id, context)),
        ("nods", nods_table(session_id, context)),
    ):
        if not frame.empty:
            path = workspace.table(f"{name}.csv")
            frame.to_csv(path, index=False)
            written[name] = path

    timeline = timeline_table(session_id, context)
    if not timeline.empty:
        path = workspace.file("timeline.parquet")
        timeline.to_parquet(path, index=False)
        written["timeline"] = path

    # The transcript as plain text as well as inside the report, so it can be
    # read, searched and marked up without a browser.
    if context.turn_set is not None and context.turn_set.turns:
        from conversation_analyst.report.transcript import transcript_text

        path = workspace.file("transcript.txt")
        path.write_text(transcript_text(context), encoding="utf-8")
        written["transcript"] = path

    # Raw counts per person and the transcript as rows. Written here rather
    # than by each caller so that the CLI and the desktop app can never
    # produce different sets of files from the same analysis.
    from conversation_analyst.report.exports import write_session_exports

    written.update(write_session_exports(workspace, session_id, context, long))

    return written
