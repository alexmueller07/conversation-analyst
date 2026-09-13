"""One page for a whole run, rather than one page per conversation.

A per-session dashboard answers "what happened in this conversation". Nobody
analyzing a corpus asks that first. They ask which sessions are usable, how
the measures are distributed, whether anything is an outlier, and where to
look when something is. Answering those by opening eight dashboards in turn
is how a real problem in session five gets missed.

Three things are shown, in the order they are needed.

**Which sessions can be used.** Verdict, duration, turns, and the specific
check that failed -- so a corpus can be filtered on evidence before anyone
looks at a value.

**How every measure is distributed across the corpus.** Each person-level
measure gets a strip of its per-session values with the median marked, on a
per-measure scale. A single session sitting far from the rest is visible
immediately, and that is nearly always a detector failure rather than a
remarkable participant.

**What was not measurable, and why.** Withheld measures are grouped by
reason. On a corpus recorded with one shared audio feed, nine measures are
withheld from every session for the same reason, and seeing that once at the
top is more useful than meeting it nine times per dashboard.

Self-contained HTML, like the session dashboards: no CDN, no build step,
openable from a network share years later.
"""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from conversation_analyst.measures.base import registry
from conversation_analyst.report.qc import VERDICT_LABELS as _VERDICT_LABEL

_VERDICT_ORDER = {"fail": 0, "review": 1, "pass_limits": 2, "pass": 3}
"""Sort order for the session table: worst first.

Deliberately not alphabetical and not chronological. The sessions that need
a decision are the ones that failed, and a table which lists them last is one
where a broken session reaches a results table because nobody scrolled."""


@dataclass
class SessionEntry:
    """One session's line in the corpus report."""

    session_id: str
    verdict: str
    duration_s: float = 0.0
    n_turns: int = 0
    values_available: int = 0
    values_total: int = 0
    seconds: float = 0.0
    dashboard: str = ""
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    """(check name, severity, message) for each check that did not pass."""
    values: dict[tuple[str, str | None], float] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    """measure id -> the reason it could not be computed."""
    error: str = ""

    @property
    def minutes(self) -> float:
        return self.duration_s / 60.0


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


_CSS = """
:root{--bg:#0d1117;--fg:#e6e8ec;--muted:#8b93a1;--line:#232b38;--card:#161c26;
--a:#2dd4bf;--b:#fbbf24;--ok:#4ade80;--warn:#fbbf24;--fail:#f87171;
--accent:#2dd4bf;}
@media (prefers-color-scheme:light){:root{--bg:#fff;--fg:#0f172a;
--muted:#64748b;--line:#e2e8f0;--card:#f8fafc;--a:#0f766e;--b:#b45309;
--ok:#15803d;--warn:#b45309;--fail:#b91c1c;--accent:#4f46e5;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:34px 22px 80px}
h1{font-size:28px;margin:0 0 6px}
h2{font-size:18px;margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:13px;margin:22px 0 8px;color:var(--muted);text-transform:uppercase;
letter-spacing:.06em}
.sub{color:var(--muted);margin:0 0 22px;font-size:14px}
.desc{color:var(--muted);font-size:12.5px;max-width:68ch;margin:0 0 12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.tile .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .k{font-size:12px;color:var(--muted);margin-top:2px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:8px 11px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;
letter-spacing:.04em;background:var(--card);position:sticky;top:0}
td.num,th.num{text-align:right}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;
font-weight:650;letter-spacing:.03em}
.badge.pass{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
.badge.pass_limits{background:color-mix(in srgb,var(--ok) 12%,transparent);
color:var(--ok);border:1px dashed var(--ok)}
.badge.review{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.badge.fail{background:color-mix(in srgb,var(--fail) 18%,transparent);color:var(--fail)}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.na{color:var(--muted);font-style:italic}
.note{background:var(--card);border-left:3px solid var(--warn);padding:10px 13px;
border-radius:0 8px 8px 0;margin:8px 0;font-size:13.5px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
details{margin:10px 0}
summary{cursor:pointer;color:var(--muted);font-size:13.5px}
svg{display:block}
.strip{display:grid;grid-template-columns:minmax(190px,26%) 1fr minmax(120px,15%);
gap:10px;align-items:center;padding:5px 0;border-bottom:1px dashed var(--line)}
.strip:last-child{border-bottom:none}
.strip .name{font-size:13px}
.strip .stat{font-size:12px;color:var(--muted);text-align:right;
font-variant-numeric:tabular-nums;white-space:nowrap}
footer{margin-top:52px;color:var(--muted);font-size:12.5px;
border-top:1px solid var(--line);padding-top:16px}
"""


def _distribution_strip(
    values: Sequence[float], labels: Sequence[str], width: int = 460, height: int = 26
) -> str:
    """Per-session values on a shared axis, with the median marked.

    A strip rather than a histogram: with eight sessions a histogram is
    mostly empty bins, while individual dots stay individually identifiable
    -- hovering names the session, which is what makes an outlier actionable
    rather than merely visible.
    """
    finite = [(v, l) for v, l in zip(values, labels) if np.isfinite(v)]
    if len(finite) < 2:
        return '<span class="na">too few sessions</span>'

    vals = np.array([v for v, _ in finite], dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def x(v: float) -> float:
        return 6 + (width - 12) * (v - lo) / (hi - lo)

    mid = height / 2
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<line x1="6" y1="{mid}" x2="{width - 6}" y2="{mid}" '
        f'stroke="var(--line)" stroke-width="2"/>',
    ]
    median = float(np.median(vals))
    parts.append(
        f'<line x1="{x(median):.1f}" y1="4" x2="{x(median):.1f}" y2="{height - 4}" '
        f'stroke="var(--fg)" stroke-width="1.5" opacity=".55"/>'
    )
    for value, label in finite:
        parts.append(
            f'<circle cx="{x(value):.1f}" cy="{mid}" r="4.5" fill="var(--accent)" '
            f'opacity=".75"><title>{_esc(label)}: {value:.4g}</title></circle>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _overview(entries: Sequence[SessionEntry]) -> str:
    counts = Counter(e.verdict for e in entries)
    minutes = sum(e.minutes for e in entries if np.isfinite(e.minutes))
    turns = sum(e.n_turns for e in entries)
    usable = counts["pass"] + counts["review"]
    tiles = [
        (f"{len(entries)}", "Conversations"),
        (f"{usable}", "Usable (pass or review)"),
        (f"{counts['pass']}", "Passed every check"),
        (f"{counts['fail']}", "Failed"),
        (f"{minutes:.0f} min", "Total recorded"),
        (f"{turns:,}", "Turns measured"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{_esc(k)}</div></div>'
        for v, k in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _session_table(entries: Sequence[SessionEntry]) -> str:
    rows = []
    for entry in sorted(
        entries, key=lambda e: (_VERDICT_ORDER.get(e.verdict, 3), e.session_id)
    ):
        link = (
            f'<a href="{_esc(entry.dashboard)}">{_esc(entry.session_id)}</a>'
            if entry.dashboard else _esc(entry.session_id)
        )
        if entry.error:
            detail = f'<span class="mono">{_esc(entry.error)}</span>'
        else:
            fatal = [f for f in entry.failures if f[1] == "fatal"]
            shown = fatal or entry.failures
            detail = (
                "<br>".join(_esc(m) for _, _, m in shown[:2])
                if shown else '<span class="na">nothing flagged</span>'
            )
        coverage = (
            f"{entry.values_available}/{entry.values_total}"
            if entry.values_total else "&mdash;"
        )
        rows.append(
            f"<tr><td><strong>{link}</strong></td>"
            f'<td><span class="badge {_esc(entry.verdict)}">'
            f"{_esc(_VERDICT_LABEL.get(entry.verdict, entry.verdict.upper()))}</span></td>"
            f'<td class="num">{entry.minutes:.1f}</td>'
            f'<td class="num">{entry.n_turns}</td>'
            f'<td class="num">{entry.n_turns / entry.minutes:.1f}</td>'
            f'<td class="num">{coverage}</td>'
            f"<td>{detail}</td></tr>"
            if entry.minutes > 0 else
            f"<tr><td><strong>{link}</strong></td>"
            f'<td><span class="badge {_esc(entry.verdict)}">'
            f"{_esc(_VERDICT_LABEL.get(entry.verdict, entry.verdict.upper()))}</span></td>"
            f'<td class="num">&mdash;</td><td class="num">&mdash;</td>'
            f'<td class="num">&mdash;</td><td class="num">{coverage}</td>'
            f"<td>{detail}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>Session</th><th>Verdict</th>'
        '<th class="num">Minutes</th><th class="num">Turns</th>'
        '<th class="num">Turns/min</th><th class="num">Values</th>'
        "<th>What was flagged</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _distributions(entries: Sequence[SessionEntry]) -> str:
    """Every measure's spread across the corpus, grouped by family."""
    by_family: dict[str, list[str]] = defaultdict(list)
    for spec in registry.specs:
        by_family[spec.family].append(spec.id)

    blocks = []
    for family in sorted(by_family):
        strips = []
        for measure_id in sorted(by_family[family]):
            spec = registry.spec(measure_id)
            people = ["A", "B"] if spec.level == "person" else [None]
            series, labels = [], []
            for entry in entries:
                for person in people:
                    value = entry.values.get((measure_id, person))
                    if value is not None and np.isfinite(value):
                        series.append(float(value))
                        labels.append(
                            f"{entry.session_id}" + (f" ({person})" if person else "")
                        )
            if len(series) < 2:
                continue
            values = np.array(series)
            strips.append(
                f'<div class="strip"><div class="name">{_esc(spec.label)}</div>'
                f"<div>{_distribution_strip(series, labels)}</div>"
                f'<div class="stat">med {np.median(values):.3g}<br>'
                f"n={values.size}</div></div>"
            )
        if strips:
            blocks.append(
                f"<details><summary>{_esc(family.replace('_', ' '))} "
                f"({len(strips)})</summary>{''.join(strips)}</details>"
            )
    return "".join(blocks) or '<p class="na">No measures were computed.</p>'


def _withheld(entries: Sequence[SessionEntry]) -> str:
    """Measures that could not be computed, grouped by why."""
    reasons: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        for measure_id, reason in entry.unavailable.items():
            reasons[reason].add(measure_id)
    if not reasons:
        return '<p class="desc">Every measure was computed for every session.</p>'

    rows = []
    for reason, ids in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        labels = sorted(
            registry.spec(m).label if m in registry else m for m in ids
        )
        shown = ", ".join(labels[:8]) + (f" and {len(labels) - 8} more" if len(labels) > 8 else "")
        rows.append(
            f'<tr><td class="num">{len(ids)}</td>'
            f"<td>{_esc(shown)}</td><td>{_esc(reason)}</td></tr>"
        )
    return (
        '<p class="desc">A measure that could not be computed is recorded as '
        "missing with a reason, never as zero. When the same reason covers many "
        "measures across every session it is usually a property of how the "
        "conversations were recorded rather than of the conversations.</p>"
        '<div class="scroll"><table><thead><tr><th class="num">Measures</th>'
        "<th>Which</th><th>Why</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _check_family(name: str) -> str:
    """Collapse per-view and per-person check names to the underlying check.

    ``video_continuity_close_a`` and ``video_continuity_close_b`` are the same
    finding about two files, and ``face_coverage_A`` about two people. Left
    apart they fill the summary with near-duplicates and bury whatever else
    went wrong.
    """
    for suffix in ("_close_a", "_close_b", "_wide", "_A", "_B"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _common_warnings(entries: Sequence[SessionEntry]) -> str:
    """What went wrong across the corpus, one line per distinct problem.

    Counted by *session* rather than by occurrence: a check that fires on
    both views of one recording is one affected session, and reporting it as
    two overstates how widespread it is.
    """
    sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    example: dict[tuple[str, str], str] = {}
    for entry in entries:
        for name, severity, message in entry.failures:
            key = (_check_family(name), severity)
            sessions[key].add(entry.session_id)
            example.setdefault(key, message)
    if not sessions:
        return ""

    ordered = sorted(
        sessions.items(),
        key=lambda kv: (kv[0][1] != "fatal", -len(kv[1])),
    )
    items = []
    for (name, severity), affected in ordered[:8]:
        which = ", ".join(sorted(affected)[:6])
        if len(affected) > 6:
            which += f" and {len(affected) - 6} more"
        label = "must fix" if severity == "fatal" else "worth knowing"
        items.append(
            f'<div class="note"><strong>{len(affected)} of {len(entries)} '
            f"sessions</strong> &middot; {_esc(label)} &mdash; "
            f"{_esc(example[(name, severity)])}"
            f'<div class="desc" style="margin:4px 0 0">{_esc(which)}</div></div>'
        )
    return "".join(items)


_FILE_NOTES: tuple[tuple[str, str, str], ...] = (
    ("counts.csv", "One row per participant, one column per raw count",
     "Nods, words, smiles, laughs, questions, backchannels, turns. Join your "
     "own grouping variable on <code>participant_id</code> and compare groups "
     "directly. Conversation minutes and speaking seconds are in the same "
     "file, because a raw count compared across conversations of unequal "
     "length is a comparison of length."),
    ("counts_dyad.csv", "Counts that belong to the pair, not to either person",
     "Shared laughter, mutual gaze episodes, the pair's total turns. Kept "
     "apart so a single pair-level value does not enter a model twice."),
    ("counts_long.csv", "The same counts, one row each, with reasons",
     "Where a count is blank in the wide file, this says why it could not be "
     "measured. Worth reading before counting rows in a session's "
     "<code>nods.csv</code> or <code>events.csv</code> instead: those list "
     "every detection, including ones from a view whose movement measures "
     "were withheld as unreliable. <code>counts.csv</code> is the file to "
     "count from."),
    ("by-measure-family/", "The catalogue split one file per family",
     "<code>&lt;family&gt;.csv</code> is long, with the columns of "
     "<code>measures_all.csv</code>; <code>&lt;family&gt;_wide.csv</code> is "
     "one row per participant. <code>INDEX.csv</code> lists which measures "
     "are in which file."),
    ("transcript_all.csv", "What was said, one row per utterance",
     "Turns, backchannels and attempts that were talked over, each labelled, "
     "in the order they happened, with a mm:ss reading for finding the moment "
     "in the recording."),
    ("transcript_words_all.csv", "One row per word",
     "Start, end and the recognizer's confidence in each word, for counting "
     "words directly or excluding a badly recognized stretch."),
    ("measures_all.csv", "Every measure, long format",
     "One row per pair, person and measure. The shape a mixed-effects model "
     "wants; dyadic data needs a random intercept for the pair."),
    ("codebook.csv", "What every measure means",
     "Definition, unit, level, interpretation and citations, generated from "
     "the same registry that computed the values."),
)


def _files(written: Mapping[str, object] | None) -> str:
    """What is in this folder and which question each file answers.

    People arrive at a results folder, not at this page, and a directory of
    thirty CSVs says nothing about which one to open. Only files that were
    actually written are listed: naming a file that is not there sends
    somebody looking for it.
    """
    from conversation_analyst.report.exports import FAMILY_KEY

    # Matched on the keys the writers return rather than on file names: two
    # different files here are both called counts.csv, and matching on the
    # name would let one of them vouch for the other.
    present = {f"{key}.csv" for key in (written or {})}
    if any(str(key).startswith(FAMILY_KEY) for key in (written or {})):
        present.add("by-measure-family/")

    rows = []
    for name, what, detail in _FILE_NOTES:
        if written is not None and name not in present:
            continue
        rows.append(
            f'<tr><td class="mono"><strong>{_esc(name)}</strong></td>'
            f"<td>{_esc(what)}<div class=\"desc\" style=\"margin:3px 0 0\">"
            f"{detail}</div></td></tr>"
        )
    if not rows:
        return ""
    return (
        '<p class="desc">Each file answers a different question, so nothing '
        "has to be reshaped before the first test. An empty cell is always a "
        "measure that could not be computed &mdash; never a zero.</p>"
        '<div class="scroll"><table><thead><tr><th>File</th>'
        "<th>What it is for</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_corpus_report(
    entries: Sequence[SessionEntry],
    title: str = "",
    written: Mapping[str, object] | None = None,
) -> str:
    """Build the whole-run HTML document."""
    usable = sum(1 for e in entries if e.verdict in ("pass", "pass_limits", "review"))
    heading = title or "Conversation corpus"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(heading)} &mdash; convlab</title>
<style>{_CSS}</style></head><body><div class="wrap">

<h1>{_esc(heading)}</h1>
<p class="sub">{len(entries)} conversation(s) analyzed &middot; {usable} usable
&middot; {len(registry)} measures in the catalogue</p>

{_overview(entries)}

<h2>Sessions</h2>
<p class="desc">Sorted worst first. Click a session to open its full report,
including the synchronized video review. A <strong>fail</strong> means a check
on the <em>inputs</em> did not pass &mdash; nothing here is filtered on whether
the results looked plausible, because screening on that is how a real effect
gets discarded.</p>
{_session_table(entries)}
{_common_warnings(entries)}

<h2>What is in this folder</h2>
{_files(written)}

<h2>What could not be measured</h2>
{_withheld(entries)}

<h2>Measure distributions</h2>
<p class="desc">Every measure that has values from at least two sessions, one
dot per session (per person, where the measure is per person), on its own
scale with the median marked. Hover a dot to name the session. A point sitting
far from the rest is worth opening before it is interpreted &mdash; in a
corpus this size it is more often a detector failure than a remarkable
participant.</p>
{_distributions(entries)}

<footer>
Generated by conversation_analyst. Every value traces to a documented measure in
<span class="mono">codebook.csv</span> and to the parameters recorded in each
session's <span class="mono">manifest.json</span>. Measures shown as missing
were not computable and are reported as such rather than as zero.
</footer>
</div></body></html>"""


def write_corpus_report(
    path: str | Path,
    entries: Sequence[SessionEntry],
    title: str = "",
    written: Mapping[str, object] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_corpus_report(entries, title, written), encoding="utf-8")
    return path
