# Conversation Analyst

**Measure a conversation the way the literature says to.**

Website: **[conversation-analyst.vercel.app](https://conversation-analyst.vercel.app)**
· Engine name: `conversation_analyst` (historically `convlab`)

Point it at a folder of recorded conversations — **two videos per pair, one
per person**. It works out who
spoke when, how quickly each replied, what they looked at, when they nodded,
smiled and laughed, how pleasant they looked and how much that followed
their partner, how their speech and movement tracked one another, and how all
of that changed as the conversation went on — **203 measures**, each
defined and unit-labeled in a codebook, plus a visual report per pair.

![The Conversation Analyst desktop application](site/assets/app.png)

## How it is validated

Five independent lines, every number measured rather than aspirational:

| Line | Result |
|---|---|
| Synthetic ground truth | nod recall/precision/cycle-count 1.00; zero false positives from dips, shakes, drift |
| The lab's own Praat runs | median pitch **r = 0.997**, 2.2 Hz median error, 8 sessions |
| The lab's OpenFace runs (independent tracking stack) | per-frame head pitch **\|r\| = 0.81**, yaw **0.93**, smile vs AU12 **0.81**, 16 participants |
| Published nod distribution (Mori, Den & Jokinen 2025) | **42.1 %** single nods vs their 42 %; 97.6 % within five cycles vs their >95 % |
| Study criterion (preliminary, n=16) | smiling ↔ partner-reported enjoyment **ρ = 0.67**; 8/12 a-priori directional tests consistent |

Plus a built-in **blind human-coding mode**: any RA can code a session
against the video and get event F1, onset error and Cohen's κ versus the
detectors — the standard the field ultimately asks for, as a 20-minute task.
What is *not* claimed: nod-for-nod human agreement is unmeasured until
someone codes sessions; that is precisely why the coding mode exists.

---

# Install it

You need **Python 3.10 or newer**. Everything else the installer handles.

The first run downloads about **1.6 GB** of libraries and takes **15–30
minutes** on a normal connection. That happens once. After that the app opens
in seconds and runs entirely offline apart from a one-time 27 MB model
download. No GPU required.

## Windows

1. **Install Python** if you don't have it — [python.org/downloads](https://www.python.org/downloads/).
   On the first screen of the installer, **tick "Add python.exe to PATH"** before
   clicking Install. This is the step people miss.

2. **Download this project.** Either:
   - click the green **Code** button above → **Download ZIP** → right-click the
     downloaded file → **Extract All**, or
   - if you have Git: `git clone https://github.com/alexmueller07/conversation-analyst.git`

3. **Open the folder** you just extracted or cloned.

4. **Double-click `launch-conversation-analyst.bat`.**

   The first run installs everything — you'll see a black window with progress
   text for 15–30 minutes. Leave it alone until the app appears. Every run
   after that opens in a couple of seconds.

   > If Windows shows a blue "Windows protected your PC" box, click
   > **More info** → **Run anyway**. That appears for any unsigned script.

## macOS

1. **Install Python with Tk support** (the version Apple ships is missing it):
   ```bash
   brew install python python-tk
   ```
   No Homebrew? Get it at [brew.sh](https://brew.sh), or install Python from
   [python.org](https://www.python.org/downloads/) which includes Tk.

2. **Download the project:**
   ```bash
   git clone https://github.com/alexmueller07/conversation-analyst.git
   cd conversation-analyst
   ```

3. **Start it:**
   ```bash
   chmod +x launch-conversation-analyst.sh
   ./launch-conversation-analyst.sh
   ```
   First run takes 5–15 minutes.

## Linux

```bash
sudo apt install python3 python3-venv python3-tk git    # Debian/Ubuntu
git clone https://github.com/alexmueller07/conversation-analyst.git
cd conversation-analyst
chmod +x launch-conversation-analyst.sh
./launch-conversation-analyst.sh
```

---

# Use it

## Try it with no data at all

Click **Use demo data**. It builds a synthetic two-person conversation using
your computer's speech voices, writes three video files, and analyzes them.
Takes about four minutes start to finish. This is the fastest way to see what
the tool produces. *(Windows only — it needs the system speech engine.)*

## Analyze your own recordings

**1. Name your files.** Each conversation is **two videos** — one showing each
person's face. Give the pair a shared id and a person token:

```
recordings/
├── dyad012_close_a.mp4     person A's face
├── dyad012_close_b.mp4     person B's face
├── dyad013_close_a.mp4     next pair
├── dyad013_close_b.mp4
└── ...                     as many pairs as you like
```

Put every pair in the one folder — it processes all of them in a batch.

**Both files will contain both voices. That is expected.** Working out who is
speaking is the tool's job, and it handles the two setups differently:

- **In-person, one camera per person.** Each camera's microphone sits nearer
  its own participant, so the same voice arrives at the two microphones at
  different levels. That level difference is the primary cue and it is very
  strong (0.04 % speaker confusion).
- **Zoom, Teams, or any per-participant export.** These mix one shared audio
  feed into every participant's file, so the two recordings are *identical*
  in audio and the level cue does not exist at all. The tool detects this
  automatically and attributes speech from **which person's mouth is moving**
  instead, using the per-participant video.

You don't have to tell it which you have — it measures the level difference
and says which mode it used in the log and the quality report.

**Naming.** It recognizes `close_a` / `cam_a` / `person_a` / `p1` for person A
and the `b` / `p2` equivalents for person B. It also handles files with **no
A/B token at all**, such as `<participant>_<session>.mp4`:

```
1101_101.mp4  +  1102_101.mp4       →  session 101
AN101_AN101.mp4 + AN102_AN101.mp4   →  session AN101
```

It works out which field identifies the session, pairs on it, and records
which participant became person A in the results (`meta_participant_a`). If
your filenames follow no pattern at all, use a
[manifest](#for-developers).

A third **wide** view showing both people is supported but not needed —
nothing measured depends on it. Against scripted conversations, two cameras
score within 0.002 of three on speech detection and identically on turn
detection.

**2. Click Browse** next to *Recordings* and choose that folder. The app
immediately lists what it found, so you'll know at once if a filename didn't
parse — not forty minutes later.

**3. Choose where results go** (or accept the default).

**4. Untick anything you don't need, and set the Speed.** Face and body
tracking together are about 93 % of the runtime and cost roughly the same as
each other, so turning *Track body* off roughly halves a run. **Fastest**
runs all four tracking jobs at once and needs about 1.5 GB free — close other
applications first. **Automatic** picks from the memory available when the
run starts, which on a busy laptop is often one or two.

**5. Click Analyze.** Progress and a running log appear as it works. **Stop**
is safe at any point — it finishes the current step and leaves valid output.

**6. Click Open report** when it finishes.

> **First run downloads about 27 MB of model files.** It happens once,
> automatically, and needs an internet connection. After that the app runs
> entirely offline.

## How long it takes

**Vision is the whole story.** Across the lab's sixteen-file test corpus,
face and body landmarking were **93 % of total stage time** — 126 minutes out
of 135. Everything else together, including transcription, is noise by
comparison. So the only question that matters for runtime is how many
tracking jobs run at once.

A session has **four independent tracking jobs**: a face track and a body
track for each participant. They do not depend on each other or on anything
else in the pipeline, so all four start before the audio stages do, and each
is joined only where its result is first needed. Body tracking therefore
overlaps transcription, prosody and semantics rather than queueing behind
them, and on a machine with room it stops contributing to wall-clock at all.

Measured on the lab laptop (12 threads, 8 GB, no graphics card). A single
tracking child runs the face landmarker at about 60 frames a second and the
pose landmarker at about 35, using 1.2 of the twelve logical cores — so
running them one after another leaves nine tenths of the machine idle for
the twenty minutes that dominates a session. Four children at once return
**1.9× the frames per second in aggregate**, which is most of what a
15 W laptop chip has to give; the rest goes to its power limit rather than
to the work.

**Check the log line that says how many workers it chose.** The automatic
policy reads free memory, and on a laptop with a browser open it will often
pick one or two. Closing other applications before a run is the single
fastest thing available; failing that, set `tracking_workers: 4` in the
config to force it. A tracking child costs about 520 MB for the first and
only ~200 MB for each one after — MediaPipe's images are shared between
processes — so four fit in about 1.2 GB, not the 5 GB an earlier and wrong
estimate assumed.

Two smaller savings, both free: frames are decoded ahead of the tracker on a
background thread, and a frame byte-identical to the one before it reuses the
previous result instead of being landmarked again. The second matters on
conferencing recordings that freeze, where it can remove most of the work;
it changes nothing about the output, because an identical image produces an
identical answer.

Body tracking samples at 12.5 Hz by default (gesture and posture live below
3 Hz), which halves the pose landmarker's frame count with oversampling to
spare. Turning body tracking off entirely roughly halves what remains.

Re-running is much faster than the first pass, because every slow stage is
cached — face and body tracks, the transcript, voice activity, prosody and
laughter. Adding a measure and re-running a corpus costs seconds per
session, not minutes. `conversation-analyst benchmark` measures cold and warm runtime on
your own machine.

---

# What you get

```
results/
├── counts.csv              one row per participant, one column per raw count —
│                           nods, words, smiles, laughs, questions, turns
├── transcript_all.csv      what was said, one row per utterance
├── transcript_words_all.csv one row per word, with timing and confidence
├── by-measure-family/      the catalogue split one file per family, long and
│                           wide, plus INDEX.csv naming what is in each
├── measures_all.csv        every pair, every measure, long format
├── index.html              open this first: every session, what passed,
│                          what was withheld, and every distribution
├── codebook.csv            what all 203 measures mean
├── counts_dyad.csv         counts that belong to the pair, not to either person
├── session_summary.csv     pass / review / fail per pair
└── dyad012/
    ├── dashboard.html      the visual report
    ├── transcript.txt      the whole conversation, timestamped, for reading
    ├── tables/counts.csv   this pair's counts, same columns as the corpus file
    ├── tables/transcript.csv        and transcript_words.csv, for this pair
    ├── tables/turns.csv    every turn, with its text and timing
    ├── tables/events.csv   nods, smiles, laughs, interruptions, callbacks
    ├── tables/nods.csv     one row per nod: cycles, magnitude, speaking or
    │                       listening — for modelling nods individually
    ├── timeline.parquet    frame-level signals, for re-analysis
    ├── qc.json             every quality check and its result
    └── manifest.json       exact settings used, for reproducibility
```

![The generated report](docs/images/dashboard.png)

## Comparing groups of people

`counts.csv` is the file for "did this group do more of X than that group".
One row per participant, one column per raw count, and a `participant_id` to
join a grouping variable onto. Nothing needs reshaping first:

```r
d <- read.csv("results/counts.csv")
d <- merge(d, read.csv("my-participants.csv"))   # adds a `group` column
t.test(nod_count ~ group, data = d)
aggregate(cbind(nod_count, word_count, smile_count) ~ group, d, mean)
```

Conversation length rides along in the same file, so the comparison that
length confounds can be corrected rather than made by accident — and it is
worth making both ways, because they disagree more often than people expect:

```r
t.test(I(nod_count / duration_min) ~ group, data = d)
```

`by-measure-family/` holds the same catalogue split one file per family, so a
question about head movement opens `head_wide.csv` rather than filtering all
203 measures. Each family comes in both shapes: `<family>.csv` is long, with
the columns of `measures_all.csv`, and `<family>_wide.csv` is one row per
participant. `INDEX.csv` lists which measures are in which file.

## Modelling the pair

`measures_all.csv` is long format — one row per pair, person and measure — so
it goes straight into a mixed-effects model. Dyadic data is non-independent,
so it wants a random intercept for the pair:

```r
library(lme4)
d <- read.csv("results/measures_all.csv")
lat <- subset(d, measure == "response_latency_median" & available)
summary(lmer(value ~ meta_condition + (1 | session_id), data = lat))
```

## Reading what was said

`transcript_all.csv` is one row per utterance — turns, backchannels and
attempts that got talked over, each labelled, in the order they happened, with
a `mm:ss` clock reading for finding the moment in the recording.
`transcript_words_all.csv` is one row per word, with its start, end and the
recognizer's confidence in it, for counting words directly or excluding a
badly recognized stretch instead of trusting it.

**Three conventions to know before analyzing.** A measure that could not be
computed is a row with an empty value and a stated reason — never a zero, and
never a dropped row, because a failed camera and an absence of behavior must
not look the same. The wide tables have no column for the reason, so an empty
cell there means the same thing and `counts_long.csv` says why — worth reading
before counting rows in `nods.csv` or `events.csv` instead, because those list
every detection, including ones from a view whose movement measures were
withheld as unreliable. `counts.csv` is the file to count from. Counts that
belong to the pair rather than to either person — shared laughter, mutual
gaze, the pair's total turns — are in `counts_dyad.csv` rather than copied
onto both participants, because a value duplicated across two rows enters a
model as two observations when it is one. And every pair carries a quality verdict based on the
*inputs* (sync confidence, tracking coverage, attribution certainty), not on
whether the numbers look plausible — filtering on surprising values is how a
real effect gets thrown away.

---

# What it measures

| Family | n | Examples |
|---|---|---|
| Turn taking | 26 | response latency median/IQR, talk-time balance, silence rate, longest lapse |
| Lexical | 25 | question rate and openness, hedging, fillers, pronouns, politeness, style matching |
| **Head** | **20** | **nods by length — single, double, triple, longer — and split by whether the person was speaking or listening** |
| **Counts** | **20** | **the raw number behind every rate in the catalogue** |
| Semantic | 16 | response coherence, topic count and duration, **long-range callbacks** |
| Prosody | 13 | pitch variability in semitones, jitter, shimmer, entrainment |
| Affect | 11 | facial valence while speaking vs listening, and how it moved |
| Backchannel | 10 | rate per minute of *partner* speech, coverage, placement within turn |
| Dynamics | 8 | change from the first third to the last: latency, silence, gaze, smiling |
| Facial expression | 8 | smiling, **Duchenne ratio**, expressivity, brow raises, shared smiling |
| Gaze | 7 | gaze at partner while speaking vs listening, mutual gaze episodes |
| Interruption | 7 | interruption vs transition overlap, success rate, floor retention |
| Synchrony | 7 | smile / head / expressivity / loudness coordination, **above chance** |
| Laughter | 4 | laughter rate, **shared laughter**, reciprocity |
| Repair | 4 | self- and other-initiated repair, change-of-state tokens |
| Body | 3 | gesture rate, postural shifts, self-touch |
| Rhythm | 3 | tempo of exchange and how steady it is |
| Structure | 3 | how the conversation is shaped over its length |

Full definitions: [`docs/measures.md`](docs/measures.md).

Three deserve their own note.

**Nods, counted by cycle.** A nod is not one thing. One down-and-up is a
different signal from a run of four, and a nod produced while listening is
doing different work from one produced mid-sentence by the person talking.
The detector follows Mori, Den & Jokinen (2025), who annotated 9,223 nods and
define a *cycle* as one consecutive up-and-down movement; a nod's length is
its cycle count, so **single**, **double** and **triple** are exact. Every nod
is also labelled by whether its producer was speaking, listening, or neither
— a distinction Poggi, D'Errico & Vincze (2010) build their whole typology of
nods on, and which McClave (2000) shows matters because speakers use head
movement for intensification and quotation rather than for agreement.

Calibrated against the published distribution: on the lab's own sixteen
recordings the detector finds **42.1 % single nods and 97.6 % at five cycles
or fewer**, over 2,178 nods, against Mori et al.'s 42 % and "more than 95 %". That agreement is
evidence the detector is cutting nods at roughly the right joints; it is not
evidence it agrees with a human coder nod-for-nod, which nobody has measured
yet and which remains this pipeline's largest open gap.

**A count beside every rate.** A rate is a count divided by a denominator, and
dividing throws the count away. "1.4 laughs per minute" is eight laughs in a
six-minute conversation and twenty-two in a sixteen-minute one, and a reader
who sees only 1.4 cannot tell which — nor whether the number rests on eight
events or eight hundred. Every rate in the catalogue now has its count
registered alongside it, and a test fails if a new rate is added without one.

**Names the recognizer has never heard.** Whisper handles conversation well
and then writes "Sunny Portland" for SUNY Cortland — the acoustics are close
and the wrong reading is the one it has seen a thousand times more often.
That is a vocabulary gap, not a capacity one, so a bigger model does not
reliably fix it. `configs/vocabulary.txt` holds the names the lab expects to
hear; they are passed to the recognizer as hotwords while it decodes, and a
conservative phonetic pass repairs what still comes out wrong. Measured over
eight sentences whose proper nouns this recognizer gets wrong, names correct
went from **5/8 to 7/8**.

**When you notice a name coming out wrong, add a line to that file and
re-run.** That is the fix — no code change, no new model. Every correction it
makes is listed in the report with what was originally heard, so nothing is
changed silently.

**Long-range callbacks** — a turn that revives something dropped at least four
turns earlier. Embedding similarity alone is useless here: any two turns about
childhood look similar whether or not one *refers back* to the other. A
callback is counted only when three things hold together — the turns are far
apart, they share a rare content anchor, and **that anchor appears in no
intervening turn**, so the topic was genuinely dropped and then picked back
up. Scored against planted callbacks: **precision 0.97, recall 1.00**.

**Synchrony above chance** — two independent behavioral time series correlate
at around 0.3 simply because behavior is autocorrelated. Reporting that as
mimicry isn't a weak result, it's an invalid one: the same number arises
between two people who never met. Every synchrony measure here is the *excess
over a surrogate baseline*, with a z score saying whether it clears that
baseline at all.

---

# The problem this solves

Each pair is filmed with one camera per person, and **every microphone picks
up both people**:

| View | Picture | Audio |
|---|---|---|
| `close_a` | person A's face | both voices |
| `close_b` | person B's face | both voices |
| `wide` *(optional)* | both people | both voices |

Nothing in the audio says whose voice it is, the cameras are started by hand
so files are offset by seconds, and their clocks drift. Nearly every measure
worth having is a *time difference*, so both problems have to be solved before
anything means anything.

**Alignment.** Full-length energy envelopes give a coarse offset that tolerates
gaps of tens of seconds; GCC-PHAT on several excerpts refines it to the
sample. Excerpt scatter becomes a confidence score, their slope a clock-drift
estimate. Recovery error on known offsets: **0.0 ms**.

**Who is speaking.** Each close-up mic sits nearer one person, so the same
voice reaches the two mics at different levels — a robust first pass, but
blind to simultaneous speech, since two people talking at once looks exactly
like one person talking ambiguously. A second pass therefore *unmixes* the two
channels into each person's own source power, which makes silence, A, B and
both genuinely distinguishable. Lip motion from each close-up joins as
independent evidence, and the result is decoded with an HMM so it stays
coherent instead of flickering mid-word.

Measured: **0.04 % speaker identity confusion**, **5.8 ms median turn-onset
error**, overlap detection at **0.97 precision**.

---

# Validation

`conversation-analyst validate` builds material whose answer is known by construction, runs
the real detectors on it, and scores them. All 29 checks pass:

| Check | Result |
|---|---|
| Camera sync recovery | 0.0 ms max error, offsets 0–11 s |
| Speech detection | F1 0.940 per person |
| Speaker identity confusion | 0.12 % |
| Overlap detection | precision 0.969, recall 0.578 |
| Turn detection | precision 0.955, recall 1.000 |
| **Turn onset accuracy** | **5.7 ms** median error |
| **Response latency accuracy** | **31 ms** median error |
| Backchannel detection | precision 0.964, recall 0.742 |
| Hesitation detection | precision 1.00, recall 0.87 |
| Interjection is not a turn | a mid-turn incursion does not split the holder |
| **Same audio in both files** — identity | **1.4 %** error |
| **Same audio in both files** — track stability | 14 % short runs (truth 9 %) |
| **Same audio in both files** — turn boundaries | 10 % overlapping onsets (truth 17 %) |
| Long-range callbacks | precision 0.967, recall 1.000 |
| Nod detection | precision 1.00, recall 1.00 |
| Nods vs single dips / shakes / drift | 0 false positives each |
| Synchrony false positive | \|z\| 1.06 on independent signals (raw r was 0.32) |
| Synchrony sensitivity | z 10.1, lag recovered exactly |

## Benchmark

`conversation-analyst benchmark` goes further than validation: it measures the
recognizer's **word error rate** against scripted synthetic speech, runs the
full pipeline end-to-end on real .mp4 files and scores the measured turn
counts, backchannel counts, response latencies and question detection
against the script's exact answer key, and times every stage with a cold
and a warm cache. Results land in `workspace/benchmark/` as four CSVs and a
readable `BENCHMARK.md`.

Two honesty notes, stated in the report itself: synthetic speech is the
system's *ceiling*, not its field performance, and the open measurement
remains agreement with human coders on real dyads.

---

# Where the measures come from

The catalogue is grounded in the conversation-science literature, most
directly Cooney & Wheatley's *Conversation* chapter in the Handbook of
Social Psychology (6th ed., 2025), which organizes the field's measurable
constructs. Every measure's codebook entry carries its source citations, so
a number in `measures_all.csv` traces to the paper that defined the
construct. Among the constructs implemented from that review:

- **Turn-gap norms** — the cross-linguistic ~200 ms modal gap and the ±1 s
  band that contains nearly all transitions (Stivers et al. 2009; Heldner &
  Edlund 2010), and fast responses as a connection signal (Templeton et
  al. 2022, 2023).
- **Backchannel function classes** — generic continuers vs. the specific
  assessments that causally shape a partner's storytelling (Bavelas et
  al. 2000; Jefferson 1984).
- **Repair** — self-correction, other-initiated repair and news receipts as
  the machinery of staying understood (Schegloff et al. 1977; Dingemanse et
  al. 2015; Heritage 1985).
- **Follow-up questions** — the specific question type that raises liking
  (Huang et al. 2017; Yeomans et al. 2019).
- **Macro-rhythm** — the 2–5 minute cycles in which partners trade blocks of
  vocal activity (Dabbs 1983; Warner 1979–92).
- **Shared reality** — partner language similarity and its growth across the
  conversation (Rossignac-Milon et al. 2021; Ta et al. 2017).
- **Openings and closings** — greeting exchanges and the pre-closing
  negotiation by which conversations actually end (Schegloff & Sacks 1973;
  Mastroianni et al. 2021).

Validation audio is real speech rendered through the system voices and placed
at exact known times, so the whole chain runs on material every model accepts
while the answer stays known to the millisecond.

**What this does not establish.** Synthetic material cannot show a detector
works on human participants — it has no head turns, no accents, no overlapping
laughter, nobody leaning out of frame. What it does establish is that the path
from a known event to a reported number is correct, which is where silent
errors live: a sign flip, a boundary convention that shifts every latency by a
frame, a threshold that suppresses a whole class of event.

---

# Limitations

Stated plainly, because a measurement tool that oversells itself is worse than
useless.

- **Accuracy on real pairs is unmeasured.** Everything above is synthetic
  ground truth. Hand-coded recordings are the next step.
- **Laughter is under-detected** — quiet and breathy laughter is missed, so
  those rates are lower bounds.
- **Gaze is inferred, not calibrated.** Camera geometry isn't recorded, so "at
  the partner" is estimated from the mode of each person's own gaze
  distribution. It assumes people look at their partner more than anywhere
  else — usually true, and it fails on someone who stared at the table.
- **Duchenne classification is a proxy**, not a sincerity detector.
- **Lexical measures are English-only.** Applied to another language they
  would produce numbers that look valid and are not.
- **Body measures need the torso in frame.** They come from the close-up
  views, where attribution is certain; a tight head-and-shoulders shot yields
  low coverage and withheld measures — the honest outcome, not a bug.
- **These are proxies for behavior, not scores of skill.** Almost none has a
  defensible "higher is better", which is why the codebook marks direction as
  unknown for most.

---

# For developers

The app is a thin shell over a library and a CLI.

```bash
pip install -e ".[semantic,dev]"

conversation-analyst-gui                  # the desktop app
conversation-analyst analyze recordings/ -o out/
conversation-analyst analyze sessions.json -o out/   # explicit manifest
conversation-analyst demo -o out/
conversation-analyst validate-study <study.zip> -w out/   # vs the study's records
conversation-analyst agreement coding-<session>-<coder>.json -w out/
conversation-analyst docs --open          # documentation of record
# `convlab` still works as an alias for one release
conversation-analyst validate                   # ground-truth checks (29+)
conversation-analyst benchmark                  # accuracy incl. WER + cold/warm runtime
conversation-analyst codebook -o docs/measures.md
pytest                             # 341 tests, no models or media needed
```

A manifest is the authoritative route when filenames aren't tidy, and it
carries study variables straight into the output tables. Paths resolve
relative to the manifest, so it can travel with the recordings:

```json
[{"session_id": "dyad012",
  "views": {"close_a": "MVI_0042.MP4", "close_b": "MVI_0117.MP4"},
  "metadata": {"condition": "control", "week": 3}},
 {"session_id": "dyad013",
  "views": {"close_a": "MVI_0208.MP4", "close_b": "MVI_0311.MP4"},
  "metadata": {"condition": "treatment", "week": 3}}]
```

Save it as `sessions.json` next to the videos and point the app or
`conversation-analyst analyze` at that file instead of the folder.

Pipeline order, and why:

```
probe → decode audio → align cameras → voice activity → recording quality
      → face tracking
      → speaker attribution  (level unmixing · lip motion · audio-visual
                              coherence · learned voice model, HMM decoded)
      → turns → transcription → turns again
      → prosody · semantics · body · hesitations · laughter
      → 203 measures → tables · codebook · QC · dashboard
```

Attribution runs *after* face tracking so mouth movement can inform it. Turn
construction runs *twice* because classifying backchannels needs the words and
transcribing needs the speech regions — a real circular dependency, resolved
by doing the cheap pass first.

Load-bearing design decisions:

- **Every stage caches** on a fingerprint of its inputs, the relevant config
  and its code version. Change a turn threshold and face tracking is reused.
- **A failing stage doesn't take the run down.** A corrupt wide camera still
  leaves turn-taking and prosody; failures are recorded, not swallowed.
- **All thresholds live in one typed config**, dumped into every run's
  `manifest.json`, so any number traces back to what produced it.
- **Model weights are pinned by SHA-256.** A silently changed upstream model
  would move every number, and a study spanning that change would contain two
  incomparable halves with nothing to say so.
- **Backchannels are excluded from turn construction.** Counting "mhm" as a
  turn inflates turn counts by about a third and pulls latency medians toward
  zero — the single convention that most changes the headline numbers.
- **A turn is a stretch of holding the floor**, not of speaking. Sorting
  speech by start time and calling every speaker change a boundary makes one
  mid-turn interjection produce both a twenty-second "overlap" and a
  twenty-second "reply", corrupting two response latencies apiece.
- **Speakers are separated even when both files carry the same audio.** A
  voice model is learned per session from labels the visual cues supply, and
  discarded if it fails held-out cross-validation. Without it, conferencing
  exports leave only lip motion and the speaker track flickers.
- **What cannot be measured is withheld, not estimated.** On a shared audio
  feed simultaneous speech has a measured recall ceiling of 0.26 at any
  setting, so the nine overlap and interruption measures come out as missing
  with a reason rather than as numbers. Response latencies on those
  recordings are right-censored at zero and the report says so.

No GPU required, and no `ffmpeg` on `PATH` (PyAV links the libraries directly,
a common silent failure on lab Windows machines).

- [`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md) — **start here**: full walkthrough of every stage and how each measure is defined
- [`docs/METHODS.md`](docs/METHODS.md) — algorithms, thresholds, and their justification
- [`docs/measures.md`](docs/measures.md) — the generated catalogue of all 203 measures

---

Built for the Niedenthal Emotions Lab, UW–Madison.
Participant recordings never leave approved storage; nothing in this
repository contains or requires participant data.
