"""Analysis-ready exports: one file per question, not one file for all of them.

``measures_all.csv`` is long format and holds all 203 measures at once. That
is the right shape for a mixed-effects model and the wrong shape for most of
what actually gets asked of this data. Someone comparing two groups of
participants -- how often each nodded, how much each talked -- opens that
file, finds tens of thousands of rows covering everything from pitch jitter
to topic coherence, and has to reshape it before the first t-test. Someone
who wants to read what was said finds no text in it at all.

So the same numbers are written again in the shapes people arrive with:

**counts.csv** -- one row per participant, one column per raw count. How many
nods, words, laughs, questions, backchannels, smiles, interruptions. This is
the file to join a grouping variable onto: add a ``gender`` or ``condition``
column keyed on ``participant_id`` and every group comparison is one line of
R away. Denominators travel with it (conversation minutes, speaking seconds)
because a raw count is only interpretable next to how long there was to
accumulate it -- thirty nods in six minutes and thirty in sixteen are not the
same finding.

**by-measure-family/** -- the catalogue split into one file per family, long
and wide, with the same columns as the combined table. A question about head
movement opens the head file rather than filtering the whole catalogue.

**transcript_all.csv / transcript_words_all.csv** -- what was actually said,
as rows. Utterance level for reading and coding, word level with per-word
timing and recognizer confidence for anything that needs to count or locate
words directly.

Two conventions carry over from the long table and matter more in the wide
ones, where there is no column to state them. An empty cell is a measure that
could not be computed, never a zero -- a session whose camera failed must not
read as a participant who never nodded. And dyad-level counts, which belong
to the pair rather than to either person, are written to their own file
instead of being copied onto both participants' rows, because a value
duplicated across two rows enters a model as two observations when it is one.

One consequence is worth stating where someone will read it. The event tables
next door -- ``nods.csv``, ``events.csv`` -- list every detection, including
detections from a view whose movement measures were withheld as unreliable.
Counting rows there and counting from ``counts.csv`` therefore disagree
exactly where the recording could not support the measure, and it is
``counts.csv`` that is right to count from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from conversation_analyst.context import AnalysisContext
from conversation_analyst.measures.base import registry

COUNT_UNIT = "count"
"""The unit that marks a measure as a raw tally rather than a rate or index.

Selecting on the unit rather than on a hand-kept list means a count added to
any family lands in ``counts.csv`` the day it is registered, and a rate can
never be mistaken for a count because someone named it ``*_count``."""

DENOMINATORS: tuple[tuple[str, str], ...] = (
    ("speaking_time", "speaking_time_s"),
    ("listening_time", "listening_time_s"),
)
"""Person-level measures carried into ``counts.csv`` as exposure columns.

Not counts themselves, and not there to be analyzed. They are there so that a
count can be turned into a rate without opening a second file, which is the
difference between "she nodded more" and "she nodded more per minute of
listening" -- and those two claims disagree often enough to matter."""


def count_measure_ids(level: str | None = None) -> list[str]:
    """Ids of every registered measure whose unit is a raw count."""
    return sorted(
        spec.id
        for spec in registry.specs
        if spec.unit == COUNT_UNIT and (level is None or spec.level == level)
    )


def _participant_id(session_id: object, person: object) -> str:
    return f"{session_id}:{person}"


def _pivot(
    long: pd.DataFrame, measures: Sequence[str], index: Sequence[str]
) -> pd.DataFrame:
    """Pivot selected measures to columns, keeping every one of them.

    ``pivot_table`` drops a measure that is missing everywhere, which would
    silently change the columns of the export from one corpus to the next and
    break any script that reads them by position. Measures that produced no
    value anywhere are added back as empty columns instead.
    """
    subset = long[long["measure"].isin(list(measures))]
    if subset.empty:
        frame = pd.DataFrame(columns=list(index))
    else:
        # pivot_table averages duplicates without saying so, which would turn
        # a corpus assembled twice into a table of means that still looks like
        # a table of counts. There should never be two rows for the same
        # measure on the same person, so refuse rather than average.
        duplicated = subset.duplicated(list(index) + ["measure"])
        if duplicated.any():
            offenders = subset[duplicated][list(index) + ["measure"]].head(5)
            raise ValueError(
                "the long table has more than one value for the same measure "
                f"on the same person; refusing to average them:\n{offenders}"
            )
        frame = subset.pivot_table(
            index=list(index), columns="measure", values="value", dropna=False
        ).reset_index()
        frame.columns.name = None
    for measure in measures:
        if measure not in frame.columns:
            frame[measure] = np.nan
    return frame


# ----------------------------------------------------------------------
# Counts
# ----------------------------------------------------------------------


def counts_long(long: pd.DataFrame) -> pd.DataFrame:
    """The long table filtered to raw counts, reasons for absence intact."""
    if long.empty:
        return long.copy()
    return long[long["measure"].isin(count_measure_ids())].reset_index(drop=True)


def _minutes(frame: pd.DataFrame, sessions: pd.DataFrame | None) -> pd.Series:
    if sessions is None or sessions.empty or "duration_s" not in sessions:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    lookup = dict(zip(sessions["session_id"], sessions["duration_s"]))
    return (frame["session_id"].map(lookup).astype(float) / 60.0).round(3)


def counts_wide(
    long: pd.DataFrame, sessions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per participant, one column per person-level count.

    ``sessions`` supplies conversation length: a frame with ``session_id`` and
    ``duration_s`` columns. It is optional because the table is still usable
    without it, but a count column with no minutes beside it invites exactly
    the comparison that length confounds, so every caller in this package
    supplies it.
    """
    if long.empty:
        return pd.DataFrame()

    people = long[long["person"] != "dyad"]
    if people.empty:
        return pd.DataFrame()

    measures = count_measure_ids(level="person")
    frame = _pivot(people, measures, ("session_id", "person"))
    exposure = _pivot(
        people, [m for m, _ in DENOMINATORS], ("session_id", "person")
    ).rename(columns=dict(DENOMINATORS))

    frame = frame.merge(exposure, on=["session_id", "person"], how="left")
    frame.insert(
        0,
        "participant_id",
        [_participant_id(s, p) for s, p in zip(frame["session_id"], frame["person"])],
    )
    frame.insert(3, "duration_min", _minutes(frame, sessions))

    lead = ["participant_id", "session_id", "person", "duration_min"]
    lead += [name for _, name in DENOMINATORS]
    frame = frame[lead + [c for c in measures if c in frame.columns]]
    return frame.sort_values(["session_id", "person"]).reset_index(drop=True)


def counts_dyad(
    long: pd.DataFrame, sessions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per conversation, one column per dyad-level count.

    Shared laughter, mutual gaze episodes and the pair's total turns belong to
    the pair. Copied onto both participants they would be counted twice by any
    model that treats a row as an observation, so they live here instead.
    """
    if long.empty:
        return pd.DataFrame()

    dyad = long[long["person"] == "dyad"]
    measures = count_measure_ids(level="dyad")
    frame = _pivot(dyad, measures, ("session_id",))
    if frame.empty:
        return pd.DataFrame()
    frame.insert(1, "duration_min", _minutes(frame, sessions))
    frame = frame[
        ["session_id", "duration_min"] + [c for c in measures if c in frame.columns]
    ]
    return frame.sort_values("session_id").reset_index(drop=True)


# ----------------------------------------------------------------------
# One family per file
# ----------------------------------------------------------------------


def family_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot one family's long rows to one row per unit of analysis.

    Dyad-level measures keep ``dyad`` in the person column rather than being
    spread across the pair, so a row is always exactly one unit of analysis
    and the column says which.
    """
    if long.empty:
        return pd.DataFrame()
    measures = sorted(long["measure"].unique())
    frame = _pivot(long, measures, ("session_id", "person"))
    frame.insert(
        0,
        "participant_id",
        [_participant_id(s, p) for s, p in zip(frame["session_id"], frame["person"])],
    )
    return frame.sort_values(["session_id", "person"]).reset_index(drop=True)


def _slug(family: str) -> str:
    return family.replace("_", "-").replace(" ", "-").lower()


FAMILY_KEY = "family:"
"""Prefix for the family entries in the returned path maps.

Family names are not unique against the other export names -- ``counts`` is
both a measure family and the top-level per-participant file -- so merging
the two maps unprefixed silently replaced the path to ``counts.csv`` with the
path to the family file of the same name, and the caller printed the wrong
one. The prefix makes the two namespaces impossible to confuse."""


def write_family_tables(directory: str | Path, long: pd.DataFrame) -> dict[str, Path]:
    """Write one long and one wide file per measure family, plus an index."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if long.empty:
        return written

    labels = {spec.id: spec.label for spec in registry.specs}
    index_rows: list[dict] = []

    for family, frame in long.groupby("family", sort=True):
        if not str(family):
            continue
        slug = _slug(str(family))
        frame = frame.reset_index(drop=True)

        path = directory / f"{slug}.csv"
        frame.to_csv(path, index=False)
        written[f"{FAMILY_KEY}{family}"] = path

        wide_path = directory / f"{slug}_wide.csv"
        family_wide(frame).to_csv(wide_path, index=False)
        written[f"{FAMILY_KEY}{family}_wide"] = wide_path

        index_rows.append(
            {
                "family": family,
                "file_long": path.name,
                "file_wide": wide_path.name,
                "n_measures": int(frame["measure"].nunique()),
                "n_rows": int(len(frame)),
                "n_values_available": (
                    int(frame["available"].sum()) if "available" in frame else 0
                ),
                "measures": "; ".join(
                    labels.get(m, m) for m in sorted(frame["measure"].unique())
                ),
            }
        )

    index_path = directory / "INDEX.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    written[f"{FAMILY_KEY}INDEX"] = index_path
    return written


# ----------------------------------------------------------------------
# Transcript
# ----------------------------------------------------------------------


def _mmss(seconds: float) -> str:
    """A clock reading, for finding the moment in the recording by hand."""
    if seconds is None or not np.isfinite(seconds):
        return ""
    minutes, rest = divmod(float(seconds), 60.0)
    return f"{int(minutes):02d}:{rest:05.2f}"


def _utterances(session_id: str, context: AnalysisContext) -> list[dict]:
    """Every utterance in the order it was said, with the spans it occupies.

    The single source of truth behind both transcript tables, so the row a
    word is attributed to and the row printed in the utterance file cannot
    disagree. ``_spans`` is the utterance's actual speech, which for a turn is
    its inter-pausal units rather than its whole extent: a turn spans its own
    internal pauses, and a short acknowledgment by the same person can sit
    inside one of those gaps without belonging to the turn. Matching a word on
    the extent rather than on the units would file that acknowledgment's words
    under the surrounding turn.
    """
    turn_set = context.turn_set
    if turn_set is None or not (turn_set.turns or turn_set.backchannels):
        return []

    rows: list[dict] = []
    for turn in turn_set.turns:
        rows.append(
            {
                "session_id": session_id,
                "kind": "turn",
                "person": turn.person,
                "start_s": round(float(turn.start), 3),
                "end_s": round(float(turn.end), 3),
                "duration_s": round(float(turn.duration), 3),
                "speech_s": round(float(turn.speech_duration), 3),
                "n_words": int(turn.n_words),
                "text": turn.text or "",
                "turn_index": int(turn.index),
                "fto_s": None if turn.fto is None else round(float(turn.fto), 3),
                "prev_person": turn.prev_person or "",
                "overlap_onset": bool(turn.is_overlap_onset),
                "_spans": [(float(u.start), float(u.end)) for u in turn.ipus]
                or [(float(turn.start), float(turn.end))],
            }
        )
    for kind, units in (
        ("backchannel", turn_set.backchannels),
        ("aborted_attempt", turn_set.non_floor),
    ):
        for unit in units:
            rows.append(
                {
                    "session_id": session_id,
                    "kind": kind,
                    "person": unit.person,
                    "start_s": round(float(unit.start), 3),
                    "end_s": round(float(unit.end), 3),
                    "duration_s": round(float(unit.duration), 3),
                    "speech_s": round(float(unit.duration), 3),
                    "n_words": int(unit.n_words),
                    "text": unit.text or "",
                    "turn_index": None,
                    "fto_s": None,
                    "prev_person": "",
                    "overlap_onset": False,
                    "_spans": [(float(unit.start), float(unit.end))],
                }
            )

    rows.sort(key=lambda r: (r["start_s"], r["person"]))
    for index, row in enumerate(rows):
        row["utterance_index"] = index
    return rows


def _mmss(seconds: float) -> str:
    """A clock reading, for finding the moment in the recording by hand."""
    if seconds is None or not np.isfinite(seconds):
        return ""
    minutes, rest = divmod(float(seconds), 60.0)
    return f"{int(minutes):02d}:{rest:05.2f}"


TRANSCRIPT_COLUMNS = (
    "session_id", "utterance_index", "kind", "participant_id", "person",
    "start_s", "start_mmss", "end_s", "duration_s", "speech_s", "n_words",
    "text", "turn_index", "fto_s", "prev_person", "overlap_onset",
)


def transcript_table(session_id: str, context: AnalysisContext) -> pd.DataFrame:
    """Everything said, one row per utterance, in the order it was said.

    ``turns.csv`` holds floor-holding turns only. A conversation is not only
    its turns: the "mm-hm" that kept the speaker going and the attempt that
    got talked over are speech, and a transcript that omits them is not the
    conversation that happened. All three kinds are here, labelled, and no two
    rows describe the same speech, so they can be counted without counting
    anything twice.
    """
    rows = _utterances(session_id, context)
    if not rows:
        return pd.DataFrame()

    for row in rows:
        row.pop("_spans", None)
        row["participant_id"] = _participant_id(session_id, row["person"])
        row["start_mmss"] = _mmss(row["start_s"])

    frame = pd.DataFrame(rows)[list(TRANSCRIPT_COLUMNS)]
    frame["turn_index"] = frame["turn_index"].astype("Int64")
    return frame


def transcript_words_table(session_id: str, context: AnalysisContext) -> pd.DataFrame:
    """One row per recognized word, with its timing, confidence and utterance.

    The utterance table is for reading; this is for counting. Word counts,
    vocabulary, the exact moment someone said a particular thing, and how sure
    the recognizer was about it all come from here, and the confidence column
    is what lets a badly recognized stretch be excluded rather than trusted.

    ``utterance_index`` joins each word to its row in ``transcript.csv``. A
    word matches an utterance only when it falls inside that utterance's own
    speech, so a word is never filed under a turn that merely surrounds it.
    Words that match nothing -- speech the turn builder did not keep -- are
    left unjoined rather than attached to the nearest thing.
    """
    transcript = context.transcript
    if transcript is None or not transcript.words:
        return pd.DataFrame()

    by_person: dict[str, list[tuple[float, float, int, str, object]]] = {}
    for row in _utterances(session_id, context):
        for start, end in row["_spans"]:
            by_person.setdefault(row["person"], []).append(
                (start, end, row["utterance_index"], row["kind"], row["turn_index"])
            )
    for spans in by_person.values():
        spans.sort()

    def locate(person: str, start: float, end: float):
        mid = 0.5 * (start + end)
        for span_start, span_end, index, kind, turn_index in by_person.get(person, ()):
            if span_start <= mid < span_end:
                return index, kind, turn_index
        return None, "", None

    rows = []
    for word in sorted(transcript.words, key=lambda w: (w.start, w.person)):
        index, kind, turn_index = locate(word.person, word.start, word.end)
        rows.append(
            {
                "session_id": session_id,
                "participant_id": _participant_id(session_id, word.person),
                "person": word.person,
                "start_s": round(float(word.start), 3),
                "end_s": round(float(word.end), 3),
                "duration_s": round(float(word.duration), 3),
                "word": word.text,
                "confidence": round(float(word.probability), 4),
                "utterance_index": index,
                "utterance_kind": kind,
                "turn_index": turn_index,
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("utterance_index", "turn_index"):
        frame[column] = frame[column].astype("Int64")
    frame.insert(1, "word_index", range(len(frame)))
    return frame


# Writers
# ----------------------------------------------------------------------


def write_session_exports(
    workspace, session_id: str, context: AnalysisContext, long: pd.DataFrame
) -> dict[str, Path]:
    """Per-session counts and transcript tables. Returns the paths written."""
    written: dict[str, Path] = {}
    sessions = pd.DataFrame(
        [{"session_id": session_id, "duration_s": float(context.duration)}]
    )

    for name, frame in (
        ("counts", counts_wide(long, sessions)),
        ("counts_dyad", counts_dyad(long, sessions)),
        ("counts_long", counts_long(long)),
        ("transcript", transcript_table(session_id, context)),
        ("transcript_words", transcript_words_table(session_id, context)),
    ):
        if frame is not None and not frame.empty:
            path = workspace.table(f"{name}.csv")
            frame.to_csv(path, index=False)
            written[name] = path
    return written


def write_corpus_exports(
    output: str | Path,
    long: pd.DataFrame,
    sessions: pd.DataFrame | Sequence[dict] | None = None,
    transcripts: Iterable[pd.DataFrame] = (),
    words: Iterable[pd.DataFrame] = (),
) -> dict[str, Path]:
    """Write every corpus-level export. Returns the paths written.

    This is the point of the module for anyone analyzing more than one
    conversation: the per-session files answer questions about a session, and
    a comparison between groups of people is never a question about one.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if sessions is not None and not isinstance(sessions, pd.DataFrame):
        sessions = pd.DataFrame(list(sessions))

    for name, frame in (
        ("counts.csv", counts_wide(long, sessions)),
        ("counts_dyad.csv", counts_dyad(long, sessions)),
        ("counts_long.csv", counts_long(long)),
    ):
        if frame is not None and not frame.empty:
            path = output / name
            frame.to_csv(path, index=False)
            written[Path(name).stem] = path

    written.update(write_family_tables(output / "by-measure-family", long))

    for name, frames in (
        ("transcript_all", transcripts),
        ("transcript_words_all", words),
    ):
        parts = [f for f in frames if f is not None and not f.empty]
        if parts:
            path = output / f"{name}.csv"
            pd.concat(parts, ignore_index=True).to_csv(path, index=False)
            written[name] = path

    return written
