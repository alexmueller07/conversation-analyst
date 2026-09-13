"""Counts, per-family splits and the transcript as rows.

These files exist so that a group comparison and a read-through of the
conversation do not require reshaping ``measures_all.csv`` first. What is
tested here is therefore mostly shape and honesty: that a row is one unit of
analysis, that a measure which could not be computed stays empty rather than
becoming a zero, and that nothing said is counted twice or dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conversation_analyst.measures.base import MeasureValue, registry
from conversation_analyst.report import exports
from conversation_analyst.report.tables import measures_long
from conversation_analyst.speech.asr import Transcript, Word
from conversation_analyst.timeline import Segments
from conversation_analyst.turns import IPU, TurnSet

from conftest import make_turn


def _values(overrides: dict | None = None) -> list[MeasureValue]:
    """A measure value for every registered measure, mostly available."""
    overrides = overrides or {}
    out: list[MeasureValue] = []
    for spec in registry.specs:
        people = (None,) if spec.level == "dyad" else ("A", "B")
        for i, person in enumerate(people):
            key = (spec.id, person)
            if key in overrides:
                value, reason = overrides[key]
            else:
                value, reason = float(10 + i), None
            out.append(
                MeasureValue(spec.id, spec.level, person, value,
                             unavailable_reason=reason)
            )
    return out


@pytest.fixture
def long() -> pd.DataFrame:
    frames = [
        measures_long("s1", _values()),
        measures_long("s2", _values({("nod_count", "A"): (None, "no face tracked")})),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sessions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"session_id": "s1", "duration_s": 600.0},
            {"session_id": "s2", "duration_s": 300.0},
        ]
    )


class TestCountSelection:
    def test_counts_are_selected_by_unit_not_by_name(self):
        ids = exports.count_measure_ids()
        assert "nod_count" in ids
        assert "word_count" in ids
        assert "laughter_count" in ids
        assert "nod_rate" not in ids, "a rate is not a count"

    def test_the_counts_randy_asks_for_are_person_level(self):
        person = exports.count_measure_ids(level="person")
        for measure in ("nod_count", "word_count", "smile_count", "question_count",
                        "backchannel_count", "laughter_count", "turn_count"):
            assert measure in person

    def test_dyad_counts_are_separated_from_person_counts(self):
        person = set(exports.count_measure_ids(level="person"))
        dyad = set(exports.count_measure_ids(level="dyad"))
        assert not person & dyad
        assert "shared_laughter_count" in dyad


class TestCountsWide:
    def test_one_row_per_participant(self, long, sessions):
        frame = exports.counts_wide(long, sessions)
        assert len(frame) == 4
        assert list(frame.participant_id) == ["s1:A", "s1:B", "s2:A", "s2:B"]

    def test_every_registered_count_is_a_column(self, long, sessions):
        frame = exports.counts_wide(long, sessions)
        for measure in exports.count_measure_ids(level="person"):
            assert measure in frame.columns

    def test_no_dyad_rows(self, long, sessions):
        frame = exports.counts_wide(long, sessions)
        assert "dyad" not in set(frame.person)

    def test_exposure_columns_travel_with_the_counts(self, long, sessions):
        frame = exports.counts_wide(long, sessions)
        assert frame.loc[frame.session_id == "s1", "duration_min"].iloc[0] == 10.0
        assert frame.loc[frame.session_id == "s2", "duration_min"].iloc[0] == 5.0
        assert "speaking_time_s" in frame.columns
        assert "listening_time_s" in frame.columns

    def test_an_unmeasurable_count_is_empty_not_zero(self, long, sessions):
        frame = exports.counts_wide(long, sessions)
        row = frame[frame.participant_id == "s2:A"].iloc[0]
        assert pd.isna(row.nod_count), "a failed camera must not read as zero nods"
        assert frame[frame.participant_id == "s1:A"].iloc[0].nod_count == 10.0

    def test_usable_without_session_durations(self, long):
        frame = exports.counts_wide(long, None)
        assert len(frame) == 4
        assert frame.duration_min.isna().all()

    def test_empty_input_gives_an_empty_frame(self):
        assert exports.counts_wide(pd.DataFrame(), None).empty


class TestCountsDyad:
    def test_one_row_per_conversation(self, long, sessions):
        frame = exports.counts_dyad(long, sessions)
        assert list(frame.session_id) == ["s1", "s2"]
        assert "shared_laughter_count" in frame.columns

    def test_a_pair_level_count_is_never_copied_onto_both_people(self, long, sessions):
        """The reason this file exists: one value must not enter a model twice."""
        person_table = exports.counts_wide(long, sessions)
        for measure in exports.count_measure_ids(level="dyad"):
            assert measure not in person_table.columns


class TestCountsLong:
    def test_keeps_the_reason_a_value_is_missing(self, long):
        frame = exports.counts_long(long)
        row = frame[(frame.session_id == "s2") & (frame.measure == "nod_count")
                    & (frame.person == "A")].iloc[0]
        assert row.available == False  # noqa: E712 - pandas truthiness
        assert row.unavailable_reason == "no face tracked"

    def test_holds_only_counts(self, long):
        frame = exports.counts_long(long)
        assert set(frame.measure) <= set(exports.count_measure_ids())


class TestFamilyTables:
    def test_one_long_and_one_wide_file_per_family(self, long, tmp_path):
        written = exports.write_family_tables(tmp_path, long)
        for family in sorted(long.family.unique()):
            assert f"{exports.FAMILY_KEY}{family}" in written
            assert f"{exports.FAMILY_KEY}{family}_wide" in written
        assert (tmp_path / "INDEX.csv").exists()

    def test_family_keys_cannot_shadow_the_top_level_files(self, long, sessions,
                                                           tmp_path):
        """`counts` is both a family and a file; merging them must not lose one."""
        written = exports.write_corpus_exports(tmp_path, long, sessions=sessions)
        assert written["counts"] == tmp_path / "counts.csv"
        assert written[f"{exports.FAMILY_KEY}counts"] == (
            tmp_path / "by-measure-family" / "counts.csv"
        )
        assert written["counts"] != written[f"{exports.FAMILY_KEY}counts"]

    def test_family_files_keep_the_columns_of_the_combined_table(self, long, tmp_path):
        exports.write_family_tables(tmp_path, long)
        head = pd.read_csv(tmp_path / "head.csv")
        assert list(head.columns) == list(long.columns)
        assert set(head.family) == {"head"}

    def test_every_row_lands_in_exactly_one_family(self, long, tmp_path):
        exports.write_family_tables(tmp_path, long)
        total = sum(
            len(pd.read_csv(p))
            for p in tmp_path.glob("*.csv")
            if not p.name.endswith("_wide.csv") and p.name != "INDEX.csv"
        )
        assert total == len(long)

    def test_index_names_the_files_and_their_contents(self, long, tmp_path):
        exports.write_family_tables(tmp_path, long)
        index = pd.read_csv(tmp_path / "INDEX.csv")
        assert set(index.family) == set(long.family.unique())
        for _, row in index.iterrows():
            assert (tmp_path / row.file_long).exists()
            assert (tmp_path / row.file_wide).exists()

    def test_wide_family_rows_are_one_unit_of_analysis_each(self, long, tmp_path):
        exports.write_family_tables(tmp_path, long)
        wide = pd.read_csv(tmp_path / "counts_wide.csv")
        assert not wide.duplicated(["session_id", "person"]).any()


# ----------------------------------------------------------------------
# Transcript
# ----------------------------------------------------------------------


@pytest.fixture
def talking_context(context):
    """Turns, a backchannel and an attempt that got talked over."""
    turns = [
        make_turn(0, "A", 0.0, 4.0, text="so how was the exam"),
        make_turn(1, "B", 4.2, 8.2, text="honestly it was brutal", fto=0.2, prev="A"),
    ]
    backchannel = IPU(person="A", start=5.0, end=5.4, is_backchannel=True,
                      text="mm hm", n_words=2)
    aborted = IPU(person="A", start=6.0, end=6.3, text="wait", n_words=1)
    context.turn_set = TurnSet(
        turns=turns,
        ipus=[u for t in turns for u in t.ipus] + [backchannel, aborted],
        backchannels=[backchannel],
        non_floor=[aborted],
        duration=45.0,
        speech={
            "A": Segments.from_pairs([(0.0, 4.0), (5.0, 5.4), (6.0, 6.3)]),
            "B": Segments.from_pairs([(4.2, 8.2)]),
        },
    )
    context.transcript = Transcript(
        words=[
            Word("A", 0.1, 0.4, "so", 0.99),
            Word("A", 0.5, 0.9, "how", 0.97),
            Word("A", 1.0, 1.4, "was", 0.95),
            Word("A", 1.5, 2.1, "the", 0.91),
            Word("A", 2.2, 3.0, "exam", 0.42),
            Word("B", 4.4, 5.0, "honestly", 0.88),
            Word("A", 5.1, 5.3, "mm", 0.60),
        ],
        model="test",
    )
    return context


class TestTranscriptTable:
    def test_holds_everything_that_was_said(self, talking_context):
        frame = exports.transcript_table("s1", talking_context)
        assert len(frame) == 4
        assert set(frame.kind) == {"turn", "backchannel", "aborted_attempt"}

    def test_rows_are_in_the_order_it_was_said(self, talking_context):
        frame = exports.transcript_table("s1", talking_context)
        assert list(frame.start_s) == sorted(frame.start_s)
        assert list(frame.utterance_index) == list(range(len(frame)))

    def test_carries_the_text_and_a_clock_reading(self, talking_context):
        frame = exports.transcript_table("s1", talking_context)
        first = frame.iloc[0]
        assert first.text == "so how was the exam"
        assert first.start_mmss == "00:00.00"
        assert frame[frame.kind == "backchannel"].iloc[0].text == "mm hm"

    def test_participant_id_matches_the_counts_table(self, talking_context, long,
                                                     sessions):
        frame = exports.transcript_table("s1", talking_context)
        counts = exports.counts_wide(long, sessions)
        assert set(frame.participant_id) <= set(counts.participant_id)

    def test_backchannels_are_not_also_inside_turns(self, talking_context):
        """No double counting: a row is one utterance, once."""
        frame = exports.transcript_table("s1", talking_context)
        turns = frame[frame.kind == "turn"]
        others = frame[frame.kind != "turn"]
        for _, other in others.iterrows():
            same_person = turns[turns.person == other.person]
            inside = (
                (same_person.start_s <= other.start_s)
                & (other.end_s <= same_person.end_s)
            )
            assert not inside.any()

    def test_no_turns_gives_an_empty_frame(self, context):
        context.turn_set = None
        assert exports.transcript_table("s1", context).empty


class TestTranscriptWordsTable:
    def test_one_row_per_word_with_confidence(self, talking_context):
        frame = exports.transcript_words_table("s1", talking_context)
        assert len(frame) == 7
        assert frame.iloc[0].word == "so"
        assert frame.iloc[0].confidence == pytest.approx(0.99)

    def test_words_carry_the_turn_they_fell_in(self, talking_context):
        frame = exports.transcript_words_table("s1", talking_context)
        assert frame[frame.word == "exam"].iloc[0].turn_index == 0
        assert frame[frame.word == "honestly"].iloc[0].turn_index == 1
        assert pd.isna(frame[frame.word == "mm"].iloc[0].turn_index), (
            "a backchannel word belongs to no turn"
        )

    def test_word_counts_per_person_can_be_recovered(self, talking_context):
        frame = exports.transcript_words_table("s1", talking_context)
        assert (frame.person == "A").sum() == 6
        assert (frame.person == "B").sum() == 1

    def test_no_transcript_gives_an_empty_frame(self, context):
        context.transcript = None
        assert exports.transcript_words_table("s1", context).empty


class TestWriters:
    def test_corpus_exports_write_every_promised_file(self, long, sessions,
                                                      talking_context, tmp_path):
        written = exports.write_corpus_exports(
            tmp_path, long, sessions=sessions,
            transcripts=[exports.transcript_table("s1", talking_context)],
            words=[exports.transcript_words_table("s1", talking_context)],
        )
        for name in ("counts", "counts_dyad", "counts_long", "transcript_all",
                     "transcript_words_all", f"{exports.FAMILY_KEY}INDEX"):
            assert name in written, name
            assert written[name].exists()
        assert (tmp_path / "by-measure-family").is_dir()

    def test_session_exports_land_in_the_tables_folder(self, long, talking_context,
                                                       tmp_path):
        from conversation_analyst.workspace import Workspace

        workspace = Workspace(tmp_path, "s1")
        written = exports.write_session_exports(
            workspace, "s1", talking_context, long[long.session_id == "s1"]
        )
        assert written["counts"].parent.name == "tables"
        assert written["transcript"].exists()
        assert written["transcript_words"].exists()

    def test_corpus_exports_survive_an_empty_corpus(self, tmp_path):
        assert exports.write_corpus_exports(tmp_path, pd.DataFrame()) == {}


class TestCorpusReportFileSection:
    """The results page names the files, and only the ones that exist."""

    def _report(self, written):
        from conversation_analyst.report.corpus import (
            SessionEntry, render_corpus_report,
        )

        return render_corpus_report(
            [SessionEntry("s1", "pass", 600.0, 100)], "run", written
        )

    def test_lists_the_files_that_were_written(self, long, sessions, tmp_path):
        written = exports.write_corpus_exports(tmp_path, long, sessions=sessions)
        written["measures_all"] = tmp_path / "measures_all.csv"
        html = self._report(written)
        for name in ("counts.csv", "counts_dyad.csv", "by-measure-family/",
                     "measures_all.csv"):
            assert name in html, name

    def test_does_not_name_a_file_that_was_not_written(self, long, sessions,
                                                       tmp_path):
        written = exports.write_corpus_exports(tmp_path, long, sessions=sessions)
        html = self._report(written)
        assert "transcript_all.csv" not in html, (
            "no transcripts were written, so the page must not send anyone "
            "looking for one"
        )

    def test_says_nothing_when_it_has_no_paths(self):
        html = self._report(None)
        assert "What is in this folder" in html


class TestDuplicateGuard:
    """A duplicated row must not be silently averaged into a count."""

    def test_duplicated_measure_rows_are_refused(self, long, sessions):
        doubled = pd.concat([long, long], ignore_index=True)
        with pytest.raises(ValueError, match="more than one value"):
            exports.counts_wide(doubled, sessions)

    def test_the_same_measure_for_two_people_is_not_a_duplicate(self, long,
                                                                sessions):
        frame = exports.counts_wide(long, sessions)
        assert len(frame) == 4


class TestWordsJoinTheRightUtterance:
    """A turn spans its own pauses; what sits in a pause is not the turn."""

    @pytest.fixture
    def pause_context(self, context):
        """One turn in two units, with the speaker's own "mm-hm" in the gap.

        The gap is real: the turn runs 0-10 s but is only speaking 0-3 and
        7-10. A short acknowledgment at 4.5 s sits inside the turn's extent
        and outside its speech, which is exactly the case that files words
        under the wrong row if the extent is what gets matched.
        """
        units = [
            IPU(person="A", start=0.0, end=3.0, text="so anyway", n_words=2),
            IPU(person="A", start=7.0, end=10.0, text="that's the thing", n_words=3),
        ]
        turn = make_turn(0, "A", 0.0, 10.0, text="so anyway that's the thing",
                         ipus=units)
        inner = IPU(person="A", start=4.4, end=4.9, is_backchannel=True,
                    text="mm hm", n_words=2)
        context.turn_set = TurnSet(
            turns=[turn], ipus=units + [inner], backchannels=[inner],
            duration=20.0,
            speech={"A": Segments.from_pairs([(0.0, 3.0), (4.4, 4.9), (7.0, 10.0)]),
                    "B": Segments.empty()},
        )
        context.transcript = Transcript(
            words=[
                Word("A", 0.2, 0.9, "so", 0.98),
                Word("A", 1.0, 2.4, "anyway", 0.97),
                Word("A", 4.5, 4.6, "mm", 0.71),
                Word("A", 4.65, 4.85, "hm", 0.69),
                Word("A", 7.2, 7.6, "that's", 0.95),
                Word("A", 12.0, 12.4, "stray", 0.80),
            ],
            model="test",
        )
        return context

    def test_a_word_in_the_pause_is_not_filed_under_the_turn(self, pause_context):
        frame = exports.transcript_words_table("s1", pause_context)
        mm = frame[frame.word == "mm"].iloc[0]
        assert pd.isna(mm.turn_index), (
            "the acknowledgment sits in the turn's pause, not in the turn"
        )
        assert mm.utterance_kind == "backchannel"

    def test_a_word_inside_the_turn_is_filed_under_it(self, pause_context):
        frame = exports.transcript_words_table("s1", pause_context)
        assert frame[frame.word == "so"].iloc[0].turn_index == 0
        assert frame[frame.word == "that's"].iloc[0].turn_index == 0

    def test_a_word_matching_nothing_stays_unjoined(self, pause_context):
        frame = exports.transcript_words_table("s1", pause_context)
        stray = frame[frame.word == "stray"].iloc[0]
        assert pd.isna(stray.utterance_index)
        assert stray.utterance_kind == ""

    def test_utterance_index_joins_the_two_files(self, pause_context):
        words = exports.transcript_words_table("s1", pause_context)
        utterances = exports.transcript_table("s1", pause_context)
        joined = words.dropna(subset=["utterance_index"]).merge(
            utterances, on=["session_id", "utterance_index"],
            suffixes=("_w", "_u"),
        )
        assert len(joined) == int(words.utterance_index.notna().sum())
        assert (joined.person_w == joined.person_u).all()
        assert (joined.utterance_kind == joined.kind).all()

    def test_word_time_falls_inside_the_utterance_it_joined(self, pause_context):
        words = exports.transcript_words_table("s1", pause_context)
        utterances = exports.transcript_table("s1", pause_context).set_index(
            "utterance_index"
        )
        for _, w in words.dropna(subset=["utterance_index"]).iterrows():
            u = utterances.loc[w.utterance_index]
            assert u.start_s <= 0.5 * (w.start_s + w.end_s) < u.end_s
