"""The documentation page: every decision, its reasoning, its source.

A measurement tool for research has a duty an ordinary app does not: every
number it emits will end up defended in front of a reviewer, so every
decision behind the number has to be inspectable -- what was decided, why,
and on whose published evidence. This page is that record, generated from
a structured registry so it cannot drift from the code, and shipped inside
the app so the person defending a number does not need the repository.

The measure catalogue section is built from the live measure registry --
the same objects that compute the numbers -- so a measure cannot exist
without appearing here, and its citations are the ones its spec carries.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Decision:
    area: str
    decision: str
    why: str
    sources: tuple[str, ...] = ()


DECISIONS: tuple[Decision, ...] = (
    # ---- speaker attribution -----------------------------------------
    Decision(
        "Speaker attribution",
        "Who is speaking is decided from the level difference between the "
        "two close-up microphones, with lip motion and a learned per-session "
        "voice model as supporting cues, decoded jointly by an HMM.",
        "Each camera's microphone sits nearer its own participant, so the "
        "same voice arrives 15-25 dB apart -- a cue that does not depend on "
        "the voices being distinguishable. When both files carry one shared "
        "feed (every Zoom-era export), the difference is zero everywhere; "
        "the pipeline detects this and shifts weight to lip-audio coherence "
        "and a voice model validated on held-out frames before use.",
    ),
    Decision(
        "Speaker attribution",
        "Simultaneous speech is measured only when the recording can carry "
        "evidence for it; on shared-audio recordings every overlap measure "
        "is withheld and response latencies are flagged right-censored.",
        "Measured against known overlap, recall on a shared feed never "
        "exceeds 0.26 at any setting -- a single mixed channel simply does "
        "not say whether one voice or two is present. Reporting a number "
        "anyway would be reporting the prior.",
    ),
    # ---- turns ---------------------------------------------------------
    Decision(
        "Turns",
        "Speech separated by less than 180 ms of silence is one "
        "inter-pausal unit; turns are floor-holdings built from IPUs, and "
        "vocalisations under 1.2 s / 4 words inside the partner's floor are "
        "backchannels rather than turns.",
        "The 180 ms bound keeps stop closures from splitting units, and "
        "the backchannel definition follows the listener-response "
        "literature: 'mhm' is not a bid for the floor.",
        (
            "Heldner & Edlund (2010) J. Phonetics 38:555",
            "Yngve (1970) CLS 6:567",
            "Bavelas, Coates & Johnson (2000) JPSP 79:941",
        ),
    ),
    Decision(
        "Turns",
        "Response-latency distributions are centred near 200 ms with "
        "10-20% overlapping onsets, and the quality checks treat large "
        "departures as evidence of broken boundaries, not exotic behavior.",
        "Cross-linguistically, floor transfers cluster around 200 ms -- "
        "fast enough that listeners must project the end of the incoming "
        "turn. A session whose 'turns' alternate five times a second is a "
        "decoder artifact, and the checks say so.",
        (
            "Stivers et al. (2009) PNAS 106:10587",
            "Heldner & Edlund (2010) J. Phonetics 38:555",
        ),
    ),
    # ---- nods ----------------------------------------------------------
    Decision(
        "Nods",
        "A nod is one or more continuous vertical head movements; its "
        "length is its count of cycles (one up-and-down pair; a trailing "
        "half-cycle counts as a cycle). Single, double and triple nods are "
        "therefore exact categories, tabulated separately.",
        "Definitions adopted verbatim from the largest hand-annotated nod "
        "corpus available (9,223 nods). Counts and cycle structure are what "
        "human coders produce; a rate cannot express them.",
        ("Mori, Den & Jokinen (2025) PLoS ONE 20(5):e0323448",),
    ),
    Decision(
        "Nods",
        "Starting a nod requires a 2.0 deg pitch excursion; continuing one "
        "requires half that. The pair (2.0, 0.5) was calibrated against the "
        "published length distribution, not by eye: the full pipeline "
        "reproduces 42.1% single nods and 97.6% within five cycles on the "
        "lab's corpus, against the published 42% and 'more than 95%'.",
        "Nod magnitude declines across cycles -- real nods taper -- so one "
        "threshold amputates long nods and reports triples as singles. "
        "Distribution agreement is evidence the detector cuts nods at the "
        "right joints; nod-for-nod human agreement is measured separately "
        "with the built-in coding mode.",
        ("Mori, Den & Jokinen (2025) PLoS ONE 20(5):e0323448",),
    ),
    Decision(
        "Nods",
        "Every nod is labelled speaking / listening / neither from what its "
        "producer was doing at its midpoint, and the two roles are never "
        "summed.",
        "Speakers' head movements do different work -- emphasis, "
        "quotation, enumeration -- from listeners' acknowledgement. The nod "
        "typology literature is organised on exactly this split.",
        (
            "Poggi, D'Errico & Vincze (2010) LREC 2010:2570",
            "McClave (2000) J. Pragmatics 32:855",
            "Dittmann & Llewellyn (1968) JPSP 9:79",
        ),
    ),
    Decision(
        "Nods",
        "The admissible nod band is 0.8-5 Hz.",
        "Conversational head oscillation spans 0.2-7 Hz, split into slow, "
        "ordinary (1.9-3.6 Hz) and rapid classes; below 0.8 Hz the head is "
        "drifting, not nodding, and above 5 Hz a 25 fps camera cannot "
        "resolve the movement honestly. Observed nodding on the lab's "
        "corpus sits near 1.4-1.7 Hz.",
        (
            "Hadar, Steiner, Grant & Rose (1983) Human Movement Science 2:35",
            "Hadar, Steiner & Rose (1985) J. Nonverbal Behavior 9:214",
        ),
    ),
    # ---- faces ----------------------------------------------------------
    Decision(
        "Facial expression",
        "A smile counts as Duchenne when the muscles around the eyes are "
        "active at the smile's apex; the ratio is offered as a proxy, not a "
        "sincerity detector.",
        "Orbicularis oculi involvement is the standard marker separating "
        "enjoyment smiles from social smiles, and it is a matter of degree.",
        ("Ekman, Davidson & Friesen (1990) JPSP 58:342",),
    ),
    Decision(
        "Gaze",
        "The partner's direction is estimated from the mode of each "
        "person's own gaze distribution, and gaze is reported separately "
        "while speaking and while listening.",
        "Camera geometry is not recorded, so a fixed 'straight ahead' "
        "assumption would be wrong by an unknown amount every session. "
        "Speakers look away to plan; listeners look at speakers; a pooled "
        "average mostly measures how much the person listened.",
        (
            "Kendon (1967) Acta Psychologica 26:22",
            "Argyle & Dean (1965) Sociometry 28:289",
        ),
    ),
    # ---- speech recognition ---------------------------------------------
    Decision(
        "Transcription",
        "Each person is recognized from their own close-up track, cut to "
        "their own attributed speech, with conditioning on previous text "
        "disabled and known hallucination phrases dropped.",
        "Whisper cannot separate speakers after the fact, and conditioning "
        "propagates a hallucinated phrase through every following segment. "
        "Measured on this material, per-person compaction cut word error "
        "from 8.7% to 5.1% while running 11.7x faster.",
    ),
    Decision(
        "Transcription",
        "A lab vocabulary biases the decoder toward names it has never "
        "seen, and a conservative phonetic pass repairs what still comes "
        "out wrong; every repair is recorded, displayed, and "
        "timing-preserving.",
        "The recognizer writes 'Sunny Portland' for SUNY Cortland because "
        "the wrong reading is a thousand times more common in its training "
        "data -- a vocabulary gap no larger model fixes. Measured on eight "
        "such sentences: 5/8 names correct alone, 7/8 with the vocabulary; "
        "the remaining miss ('Oak layer' for Eau Claire) sits below the "
        "phonetic threshold that keeps ordinary words from being rewritten "
        "into place names, and staying conservative is the point.",
    ),
    Decision(
        "Hesitations",
        "'um' and 'uh' are found acoustically -- spectrally steady, "
        "pitch-flat held vowels -- not read from the transcript.",
        "The recognizer drops most of them (0 of 4 'uh' on scripted "
        "material), so a lexical count would be a count of recognizer "
        "habits.",
    ),
    # ---- verdicts --------------------------------------------------------
    Decision(
        "Quality verdicts",
        "A session's verdict answers 'can the numbers it reports be "
        "trusted?' -- PASS, PASS WITH LIMITS, REVIEW or FAIL. Structural "
        "limitations whose affected measures are already withheld are "
        "named limits, not review flags.",
        "The pipeline withholds what a recording cannot support (overlap "
        "on shared audio, movement measures from a frozen view, boundary "
        "timing from an uncorroborated track). Failing a session over a "
        "limitation it already handled punishes the honesty and teaches "
        "people to ignore the verdict column.",
    ),
    Decision(
        "Quality verdicts",
        "The speaker-track stability guard counts only short runs that "
        "nothing corroborates -- no recognized word, no laughter -- rather "
        "than all short runs.",
        "Real conversation is dense with genuine 200 ms vocalisations: on "
        "the lab's corpus 62-87% of short runs contain a word the "
        "recognizer found independently ('yeah', 'nice', 'oh cool'). The "
        "raw metric read listener responses as decoder noise and failed "
        "honest sessions; the uncorroborated fraction on the same sessions "
        "is 2.8-8.7%, inside the scripted ground-truth band, while genuine "
        "decoder soup still measures 50-60% raw and still fails.",
    ),
    # ---- the model-based families ----------------------------------------
    Decision(
        "Responsiveness (model-based)",
        "Listener responsiveness is fitted as a conditional-intensity "
        "model: while listening, the rate of producing nods and "
        "backchannels is a baseline plus an exponentially decaying "
        "excitation triggered by each completed partner utterance unit. "
        "Exact MLE; parameters reported as baseline rate, responses evoked "
        "per opportunity, and timescale.",
        "Two listeners with identical response rates can differ completely "
        "in whether their responding tracks the partner or an internal "
        "clock; no rate expresses that. Validated by parameter recovery on "
        "simulated listeners (8.4% median error on the evoked quantity), "
        "and an uncoupled simulated listener does not acquire a coupling "
        "(evoked share < 0.10 gate).",
        (
            "Hawkes (1971) Biometrika 58:83",
            "Ogata (1981) IEEE Trans. Inf. Theory 27:23",
            "Yngve (1970) CLS 6:567",
        ),
    ),
    Decision(
        "Tempo phases (model-based)",
        "Where the conversation changed gear is found by Bayesian online "
        "changepoint detection over turn onsets (Gamma-Poisson), with each "
        "candidate boundary required to win an exact Bayes-factor test "
        "(log BF > 3) and then refined to the evidence peak.",
        "Thirds-of-the-session trends impose an arbitrary grid; phases "
        "find the structure the conversation actually has. The pruning "
        "exists because measured failure modes demanded it: filtered "
        "change probabilities never concentrate, and raw run-length "
        "backtracking invents a boundary a minute on constant-tempo nulls. "
        "Validated: 8/10 planted double gear-changes recovered within "
        "60 s; 10/10 constant-tempo simulations reported as one phase.",
        (
            "Adams & MacKay (2007) arXiv:0710.3742",
            "Kass & Raftery (1995) JASA 90:773",
        ),
    ),
    Decision(
        "Synchrony",
        "Every synchrony value is reported as the excess over circularly "
        "shifted surrogates, never as a raw correlation.",
        "Two independent behavioral series produce sizeable correlations "
        "by autocorrelation alone; measured here, independent signals show "
        "raw r = 0.32 where the surrogate-corrected excess correctly reads "
        "chance.",
        (
            "Boker et al. (2002) Psychol. Methods 7:338",
            "Moulder et al. (2018) Psychol. Methods 23:757",
        ),
    ),
    # ---- output shape --------------------------------------------------
    Decision(
        "Output",
        "Every rate is written with its raw count beside it, and the counts "
        "are collected again in one file with a row per participant, an "
        "identifier to join a grouping variable onto, and the conversation "
        "length and speaking time they accumulated over.",
        "A rate is a count over a denominator, and dividing discards the "
        "count: 1.4 laughs per minute is eight laughs in a six-minute "
        "conversation and twenty-two in a sixteen-minute one, which are "
        "different observations. Group comparisons are made on counts, so "
        "the counts are given in the shape a group comparison needs rather "
        "than left to be reconstructed; the exposure columns travel with "
        "them because a raw count compared across conversations of unequal "
        "length is a comparison of length.",
    ),
    Decision(
        "Output",
        "Counts that belong to the pair -- shared laughter, mutual gaze "
        "episodes, the pair's total turns -- are written to their own file "
        "rather than copied onto both participants' rows.",
        "Dyadic data is already non-independent; a pair-level value repeated "
        "on two rows enters a model as two observations when it is one, "
        "which inflates the effective sample size without anything in the "
        "table saying so.",
        (
            "Kenny, Kashy & Cook (2006) Dyadic Data Analysis",
        ),
    ),
    Decision(
        "Output",
        "The transcript is written as rows -- one per utterance and one per "
        "word -- and includes backchannels and attempts that were talked "
        "over, each labelled, alongside the floor-holding turns.",
        "A turn table is not the conversation: the \"mm-hm\" that kept the "
        "speaker going and the attempt that got cut off are speech, and a "
        "transcript that omits them misrepresents what was said. Labelling "
        "the kinds rather than merging them keeps the three separable, and "
        "because a backchannel is never also part of a turn the rows can be "
        "counted without counting anything twice. Per-word recognizer "
        "confidence travels with the words so a badly recognized stretch can "
        "be excluded rather than trusted.",
    ),
    # ---- validation frame --------------------------------------------------
    Decision(
        "Validation",
        "Detectors are validated five independent ways: synthetic material "
        "with planted events; the lab's own hand-run Praat statistics on "
        "the same recordings (median pitch r = 0.997, 2.2 Hz median "
        "error); frame-level convergence with the lab's OpenFace runs -- an "
        "entirely independent tracking stack -- at median |r| 0.81 for head "
        "pitch, 0.93 for yaw and 0.81 for the smile channel against AU12, "
        "across 16 participants; the study's skill groups and partner "
        "reports; and a built-in blind coding mode that scores any RA's "
        "marks against the detectors (event F1 at +/-0.5 s, onset error, "
        "Cohen's kappa).",
        "No single line is sufficient: synthetic truth proves mechanics, "
        "cross-toolchain convergence proves the signals are right "
        "independently of the front-end, criterion links to instruments "
        "psychology already trusts, and human coding is the standard the "
        "field ultimately asks for.",
        (
            "Baltrusaitis, Zadeh, Lim & Morency (2018) IEEE FG -- OpenFace 2.0",
            "Cohen (1960) Educ. Psychol. Meas. 20:37",
            "Bakeman & Quera (2011) Sequential Analysis and Observational "
            "Methods for the Behavioral Sciences",
            "Miller, Berg & Archer (1983) JPSP 44:1234 -- the Opener scale",
        ),
    ),
)


_CSS = """
:root{--bg:#0d1117;--fg:#e6e8ec;--muted:#8b93a1;--card:#161c26;--line:#232b38;
--accent:#2dd4bf;--accent2:#fbbf24}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:40px 22px 90px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:18px;margin:40px 0 14px;padding-bottom:7px;
border-bottom:1px solid var(--line);color:var(--accent)}
.sub{color:var(--muted);font-size:14px;max-width:72ch}
.decision{background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:16px 18px;margin:12px 0}
.decision .what{font-weight:600;margin-bottom:6px}
.decision .why{color:var(--muted);font-size:13.5px}
.decision .src{margin-top:8px;font-size:12.5px;color:var(--accent2)}
.area{font-size:11px;text-transform:uppercase;letter-spacing:.08em;
color:var(--accent);margin-bottom:4px}
.measure{border-bottom:1px solid var(--line);padding:10px 0}
.measure b{font-size:14px}
.measure .meta{color:var(--muted);font-size:12px;margin:2px 0}
.measure .refs{color:var(--accent2);font-size:12px}
details{margin:6px 0} summary{cursor:pointer;color:var(--muted)}
.toc{display:flex;gap:14px;flex-wrap:wrap;font-size:13.5px;margin:14px 0}
.toc a{color:var(--accent);text-decoration:none}
footer{margin-top:52px;color:var(--muted);font-size:12.5px;
border-top:1px solid var(--line);padding-top:14px}
code{background:#232b38;border-radius:5px;padding:1px 6px;font-size:12.5px}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def build_documentation() -> str:
    """The complete documentation page, as a string of HTML."""
    from conversation_analyst import __version__
    from conversation_analyst.measures.base import registry

    # ---- decisions grouped by area, in first-appearance order ----------
    areas: dict[str, list[Decision]] = {}
    for decision in DECISIONS:
        areas.setdefault(decision.area, []).append(decision)

    decision_html = []
    for area, items in areas.items():
        for d in items:
            sources = (
                f'<div class="src">{_esc("; ".join(d.sources))}</div>'
                if d.sources else ""
            )
            decision_html.append(
                f'<div class="decision"><div class="area">{_esc(area)}</div>'
                f'<div class="what">{_esc(d.decision)}</div>'
                f'<div class="why">{_esc(d.why)}</div>{sources}</div>'
            )

    # ---- catalogue from the live registry ------------------------------
    by_family: dict[str, list] = {}
    for spec in registry.specs:
        by_family.setdefault(spec.family, []).append(spec)

    catalogue = []
    for family in sorted(by_family):
        rows = []
        for spec in by_family[family]:
            refs = (
                f'<div class="refs">{_esc("; ".join(spec.references))}</div>'
                if spec.references else ""
            )
            interp = (
                f'<div class="meta">{_esc(spec.interpretation)}</div>'
                if spec.interpretation else ""
            )
            rows.append(
                f'<div class="measure"><b>{_esc(spec.label)}</b> '
                f'<code>{_esc(spec.id)}</code>'
                f'<div class="meta">{_esc(spec.description)} '
                f"[{_esc(spec.unit)}; {_esc(spec.level)}-level]</div>"
                f"{interp}{refs}</div>"
            )
        catalogue.append(
            f"<details><summary><strong>{_esc(family.replace('_', ' '))}"
            f"</strong> — {len(by_family[family])} measures</summary>"
            + "".join(rows) + "</details>"
        )

    # ---- deduplicated reference list ------------------------------------
    seen: dict[str, None] = {}
    for d in DECISIONS:
        for s in d.sources:
            seen.setdefault(s, None)
    for spec in registry.specs:
        for s in spec.references:
            seen.setdefault(s, None)
    references = "".join(f"<li>{_esc(s)}</li>" for s in sorted(seen))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Conversation Analyst — Documentation</title>
<style>{_CSS}</style></head><body><div class="wrap">
<h1>Conversation Analyst</h1>
<p class="sub">Documentation of record, version {_esc(__version__)}. Every
measurement decision in the pipeline, the reasoning behind it, and the
published work it rests on. Generated from the same registry that computes
the numbers, so a measure cannot exist without appearing here.</p>
<div class="toc">
  <a href="#decisions">Decisions &amp; grounding</a>
  <a href="#catalogue">Measure catalogue ({len(registry)})</a>
  <a href="#references">References</a>
</div>

<h2 id="decisions">Decisions and their grounding</h2>
<p class="sub">What was decided, why, and on whose evidence. Numbers quoted
in the reasoning are measured on this pipeline — from its validation suite,
its benchmark, or the lab's own corpus — not aspirations.</p>
{''.join(decision_html)}

<h2 id="catalogue">The measure catalogue</h2>
<p class="sub">All {len(registry)} measures, from the live registry, with
each measure's own citations. A dash in a report is a withheld value with a
stated reason — never a zero.</p>
{''.join(catalogue)}

<h2 id="references">References</h2>
<ol style="font-size:13.5px;color:var(--muted)">{references}</ol>

<footer>
Conversation Analyst — built at the Niedenthal Emotions Lab, University of
Wisconsin–Madison. Engine: <code>conversation_analyst</code> (historically
<code>convlab</code>). This page ships with the app and regenerates with
<code>conversation-analyst docs</code>.
</footer>
</div></body></html>"""


def write_documentation(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_documentation(), encoding="utf-8")
    return path
