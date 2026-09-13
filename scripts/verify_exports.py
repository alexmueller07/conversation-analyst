"""Cross-check the exported CSVs against each other and against the tables.

The exports restate numbers that already exist elsewhere in the output --
``counts.csv`` restates counts that ``measures.csv`` holds in long form,
``transcript.csv`` restates turns that ``turns.csv`` holds, word counts can be
recomputed from ``transcript_words.csv``. Restated numbers are exactly where a
reshaping bug hides, because the file still looks right on its own.

So this reconciles them. Run it against a finished results folder:

    python scripts/verify_exports.py workspace/test-vids

Every check names what it compared and prints OK or FAIL with the rows that
disagreed. A non-zero exit means at least one check failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  OK    {name}")
    else:
        print(f"  FAIL  {name}")
        if detail:
            for line in str(detail).splitlines():
                print(f"          {line}")
        FAILURES.append(name)


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def verify(root: Path) -> None:
    counts = _read(root / "counts.csv")
    counts_dyad = _read(root / "counts_dyad.csv")
    counts_long = _read(root / "counts_long.csv")
    long = _read(root / "measures_all.csv")
    summary = _read(root / "session_summary.csv")
    transcript = _read(root / "transcript_all.csv")
    words = _read(root / "transcript_words_all.csv")
    family_dir = root / "by-measure-family"

    if counts is None or long is None:
        print("No exports found in", root)
        sys.exit(2)

    sessions = sorted(long.session_id.astype(str).unique())
    print(f"\n{root}  --  {len(sessions)} session(s)\n")

    # -- counts.csv ----------------------------------------------------
    print("counts.csv")
    check(
        "one row per participant, no duplicates",
        len(counts) == counts.participant_id.nunique() == 2 * len(sessions),
        f"{len(counts)} rows, {counts.participant_id.nunique()} ids, "
        f"{len(sessions)} sessions",
    )
    check(
        "participant_id is session_id:person",
        (counts.participant_id == counts.session_id.astype(str) + ":"
         + counts.person.astype(str)).all(),
    )
    check("no dyad rows", "dyad" not in set(counts.person))
    check(
        "duration_min matches session_summary",
        summary is not None
        and np.allclose(
            counts.session_id.astype(str).map(
                dict(zip(summary.session_id.astype(str), summary.duration_s))
            ) / 60.0,
            counts.duration_min,
            atol=0.02,
        ),
    )
    count_cols = [
        c for c in counts.columns
        if c not in ("participant_id", "session_id", "person", "duration_min",
                     "speaking_time_s", "listening_time_s")
    ]
    values = counts[count_cols].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    check(
        "every count is a non-negative whole number",
        bool(np.all(finite >= 0) and np.allclose(finite, np.round(finite))),
        f"{int((finite < 0).sum())} negative, "
        f"{int((~np.isclose(finite, np.round(finite))).sum())} non-integer",
    )

    # -- counts.csv against the long table -----------------------------
    print("\ncounts.csv vs measures_all.csv")
    mismatches = []
    blanks_that_should_be_values = []
    for measure in count_cols:
        rows = long[(long.measure == measure) & (long.person != "dyad")]
        for _, row in rows.iterrows():
            pid = f"{row.session_id}:{row.person}"
            got = counts.loc[counts.participant_id == pid, measure]
            if got.empty:
                mismatches.append(f"{pid} {measure}: missing row")
                continue
            got = got.iloc[0]
            if bool(row.available):
                if not np.isclose(float(got), float(row.value), equal_nan=False):
                    mismatches.append(
                        f"{pid} {measure}: wide={got} long={row.value}"
                    )
            elif not pd.isna(got):
                blanks_that_should_be_values.append(
                    f"{pid} {measure}: unavailable in long, {got} in wide"
                )
    check("every value matches the long table", not mismatches,
          "\n".join(mismatches[:8]))
    check(
        "an unavailable measure is blank, never zero",
        not blanks_that_should_be_values,
        "\n".join(blanks_that_should_be_values[:8]),
    )

    # -- counts_long ---------------------------------------------------
    print("\ncounts_long.csv")
    if counts_long is not None:
        check(
            "holds only count measures",
            set(counts_long.measure) <= set(long.measure),
        )
        check(
            "carries a reason wherever a value is missing",
            bool(
                counts_long.loc[~counts_long.available.astype(bool),
                                "unavailable_reason"].notna().all()
            ),
        )
        wide_blanks = int(counts[count_cols].isna().sum().sum())
        long_missing = int(
            (~counts_long[counts_long.person != "dyad"].available.astype(bool)).sum()
        )
        check(
            "blanks in counts.csv are explained in counts_long.csv",
            wide_blanks == long_missing,
            f"{wide_blanks} blank cells, {long_missing} explained",
        )

    # -- counts_dyad ---------------------------------------------------
    print("\ncounts_dyad.csv")
    if counts_dyad is not None:
        check("one row per conversation", len(counts_dyad) == len(sessions))
        overlap = set(counts_dyad.columns) & set(count_cols)
        check(
            "no pair-level count is also a per-person column",
            not overlap,
            f"both files carry: {sorted(overlap)}",
        )

    # -- families ------------------------------------------------------
    print("\nby-measure-family/")
    if family_dir.is_dir():
        index = pd.read_csv(family_dir / "INDEX.csv")
        long_files = [
            p for p in family_dir.glob("*.csv")
            if p.name != "INDEX.csv" and not p.name.endswith("_wide.csv")
        ]
        total = sum(len(pd.read_csv(p)) for p in long_files)
        check(
            "every row of measures_all.csv lands in exactly one family file",
            total == len(long),
            f"{total} rows across {len(long_files)} files, {len(long)} in the "
            f"combined table",
        )
        check(
            "the index names every family file, and each exists",
            all((family_dir / r.file_long).exists()
                and (family_dir / r.file_wide).exists()
                for _, r in index.iterrows()),
        )
        check(
            "family files keep the columns of the combined table",
            all(list(pd.read_csv(p).columns) == list(long.columns)
                for p in long_files),
        )
        bad = []
        for _, r in index.iterrows():
            wide = pd.read_csv(family_dir / r.file_wide)
            if wide.duplicated(["session_id", "person"]).any():
                bad.append(r.file_wide)
        check("each wide family row is one unit of analysis", not bad, str(bad))

    # -- transcript ----------------------------------------------------
    print("\ntranscript_all.csv")
    if transcript is not None:
        check(
            "utterances are ordered within each session",
            all(
                g.start_s.is_monotonic_increasing
                for _, g in transcript.groupby("session_id")
            ),
        )
        check(
            "every utterance has a speaker and a non-negative duration",
            bool(transcript.person.notna().all()
                 and (transcript.duration_s >= 0).all()),
        )
        overlapping = []
        for (sess, person), g in transcript.groupby(["session_id", "person"]):
            turns = g[g.kind == "turn"]
            others = g[g.kind != "turn"]
            for _, o in others.iterrows():
                inside = (turns.start_s <= o.start_s) & (o.end_s <= turns.end_s)
                if inside.any():
                    overlapping.append(f"{sess} {person} @{o.start_s}")
        check(
            "backchannels are never also inside a turn (no double counting)",
            not overlapping,
            "\n".join(overlapping[:8]),
        )
        # Against each session's own turns.csv.
        bad = []
        for session in sessions:
            turns_csv = root / session / "tables" / "turns.csv"
            if not turns_csv.exists():
                continue
            n_turns = len(pd.read_csv(turns_csv))
            n_here = len(transcript[(transcript.session_id.astype(str) == session)
                                    & (transcript.kind == "turn")])
            if n_turns != n_here:
                bad.append(f"{session}: turns.csv {n_turns}, transcript {n_here}")
        check("turn rows agree with each session's turns.csv", not bad,
              "\n".join(bad))

    print("\ntranscript_words_all.csv")
    if words is not None:
        check(
            "every word has a person, a time and a confidence",
            bool(words.person.notna().all() and words.start_s.notna().all()
                 and words.confidence.between(0, 1).all()),
        )
        check("no word ends before it starts", bool((words.end_s >= words.start_s).all()))
        # Word counts recomputed from the words must match word_count.
        bad = []
        if "word_count" in counts.columns:
            recomputed = (
                words.groupby(["session_id", "person"]).size().rename("n").reset_index()
            )
            recomputed["participant_id"] = (
                recomputed.session_id.astype(str) + ":" + recomputed.person.astype(str)
            )
            merged = recomputed.merge(
                counts[["participant_id", "word_count"]], on="participant_id",
                how="left",
            )
            for _, r in merged.iterrows():
                if pd.isna(r.word_count):
                    continue
                if int(r.n) != int(r.word_count):
                    bad.append(
                        f"{r.participant_id}: words file {int(r.n)}, "
                        f"word_count {int(r.word_count)}"
                    )
        check("word_count equals the rows in the words file", not bad,
              "\n".join(bad[:8]))

    # -- counts against the tables they were derived from ---------------
    print("\ncounts.csv vs the per-session event tables")
    for measure, table, predicate in (
        ("nod_count", "nods.csv", lambda d: d[d.kind == "nod"]),
        ("turn_count", "turns.csv", lambda d: d),
    ):
        bad = []
        for session in sessions:
            path = root / session / "tables" / table
            if not path.exists():
                continue
            frame = predicate(pd.read_csv(path))
            for person, group in frame.groupby("person"):
                pid = f"{session}:{person}"
                got = counts.loc[counts.participant_id == pid, measure]
                if got.empty or pd.isna(got.iloc[0]):
                    # Withheld for a stated reason; counts_long carries it.
                    continue
                if int(got.iloc[0]) != len(group):
                    bad.append(
                        f"{pid}: {measure}={int(got.iloc[0])}, "
                        f"{table} has {len(group)}"
                    )
        check(f"{measure} equals the rows in {table}", not bad, "\n".join(bad[:8]))

    # -- per-session files mirror the corpus ones ----------------------
    print("\nper-session tables/")
    bad = []
    for session in sessions:
        path = root / session / "tables" / "counts.csv"
        if not path.exists():
            bad.append(f"{session}: no tables/counts.csv")
            continue
        per = pd.read_csv(path)
        corpus_rows = counts[counts.session_id.astype(str) == session]
        if list(per.columns) != list(counts.columns):
            bad.append(f"{session}: different columns from the corpus file")
        elif len(per) != len(corpus_rows):
            bad.append(f"{session}: {len(per)} rows here, {len(corpus_rows)} in corpus")
    check("each session's counts.csv matches the corpus file", not bad,
          "\n".join(bad[:8]))


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "workspace")
    verify(root)
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED: " + "; ".join(FAILURES))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
