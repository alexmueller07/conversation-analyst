"""Desktop application.

Built on Tkinter deliberately. It ships with Python on every platform, so
the application adds nothing to the install that a research assistant has to
troubleshoot -- which matters far more here than a more fashionable toolkit
would. The window reads as a three-step form -- recordings, measures,
analyze -- because that is the whole job; the results table and activity
log below it report on how the job went.

Three rules shape the implementation:

* **The pipeline never runs on the UI thread.** A ten-minute dyad takes tens
  of minutes, and a frozen window is indistinguishable from a crashed one.
  Work happens on a worker thread and talks to the interface through a
  queue that the main loop drains on a timer.
* **Stopping is safe.** Cancellation is checked at stage boundaries only, so
  a stopped run never leaves a half-written cache entry that a later run
  would silently trust.
* **Failures are shown, not swallowed.** Anything that goes wrong reaches the
  log pane and the session's status, because a tool that quietly produces
  fewer numbers is worse than one that stops.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont
except ImportError as exc:  # pragma: no cover - headless install
    raise SystemExit(
        "convlab gui needs Tkinter, which is missing from this Python install.\n"
        "On Windows and macOS reinstall Python from python.org; on Debian or "
        "Ubuntu run: sudo apt install python3-tk"
    ) from exc

APP_NAME = "Conversation Analyst"
APP_TAGLINE = "Measure a conversation the way the literature says to"
APP_TITLE = f"{APP_NAME} — multimodal conversation measurement"

MODEL_DIR = Path.home() / ".convlab" / "models"
"""One model cache per user. Keeping it out of the results folder means
changing where results go never re-downloads 27 MB of weights. The path
keeps the engine's historical name so nobody's already-downloaded models
are abandoned by the rebrand."""

PALETTE = {
    # The Conversation Analyst identity: deep ink surfaces, high-contrast
    # type, and the two speakers' colors -- teal for A, amber for B -- as
    # the brand accents, the same pair every report and timeline uses.
    # Verdict colors are tuned for dark ground.
    "bg": "#0D1117",
    "panel": "#161C26",
    "panel_hi": "#1B2330",
    "text": "#E6E8EC",
    "muted": "#8B93A1",
    "accent": "#2DD4BF",
    "accent_dark": "#14B8A6",
    "accent_faint": "#123B37",
    "accent_ink": "#082220",
    "accent2": "#FBBF24",
    "ok": "#4ADE80",
    "warn": "#FBBF24",
    "fail": "#F87171",
    "line": "#232B38",
    "field": "#0F141D",
}

def default_results_dir() -> Path:
    """New installs get the new name; an existing results folder wins.

    Renaming the default out from under someone whose corpus lives in the
    old folder would scatter their results across two directories."""
    old = Path.home() / "convlab-results"
    if old.exists():
        return old
    return Path.home() / "conversation-analyst-results"


SKIPPABLE = (
    ("asr", "Transcribe speech", "Needed for questions, callbacks and style matching"),
    ("face", "Track faces", "Needed for gaze, nods, smiles and expressivity"),
    ("body", "Track body", "Gesture and posture; the slowest stage"),
    ("prosody", "Analyze voice", "Pitch, loudness and entrainment"),
    ("semantics", "Analyze meaning", "Topics, coherence and long-range callbacks"),
    ("laughter", "Detect laughter", "Laughter and shared laughter"),
)


def _draw_icon(size: int = 32) -> tk.PhotoImage:
    """Two overlapping speech bubbles in the accent color.

    Drawn pixel-by-pixel with ``PhotoImage.put`` so the app carries no image
    files and no imaging dependency. 32x32 is what title bars and taskbars
    actually display, so nothing finer would survive scaling anyway.
    """
    image = tk.PhotoImage(width=size, height=size)

    def bubble(x0: int, y0: int, x1: int, y1: int, color: str) -> None:
        radius = 4
        for y in range(y0, y1):
            for x in range(x0, x1):
                dx = max(x0 + radius - x, x - (x1 - 1 - radius), 0)
                dy = max(y0 + radius - y, y - (y1 - 1 - radius), 0)
                if dx * dx + dy * dy <= radius * radius + radius:
                    image.put(color, (x, y))

    def tail(x: int, y: int, color: str, leftward: bool) -> None:
        for step, width in enumerate((4, 3, 2, 1)):
            if leftward:
                image.put(color, to=(x, y + step, x + width, y + step + 1))
            else:
                image.put(color, to=(x - width, y + step, x, y + step + 1))

    # The two speakers' bubbles in the two speakers' colors -- amber behind
    # for person B, teal in front for person A, the same pair every report
    # uses. A two-voice mark for a two-person tool.
    bubble(2, 2, 20, 15, PALETTE["accent2"])
    tail(5, 15, PALETTE["accent2"], leftward=True)
    bubble(12, 12, 30, 25, PALETTE["accent"])
    tail(27, 25, PALETTE["accent"], leftward=False)
    return image


# ----------------------------------------------------------------------
# Worker-to-UI messages
# ----------------------------------------------------------------------


@dataclass
class Message:
    kind: str          # log | progress | session | done | error
    text: str = ""
    level: str = "info"
    value: float = 0.0
    payload: Any = None


class Worker(threading.Thread):
    """Runs the analysis off the UI thread and reports through a queue."""

    def __init__(
        self,
        target: str,
        output: str,
        skip: tuple[str, ...],
        model_dir: str,
        outbox: "queue.Queue[Message]",
        lenient: bool,
        tracking_workers: int | None = None,
    ):
        super().__init__(daemon=True)
        self.target, self.output = target, output
        self.skip, self.model_dir = skip, model_dir
        self.outbox = outbox
        self.lenient = lenient
        self.tracking_workers = tracking_workers
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    # ------------------------------------------------------------------
    def send(self, kind: str, text: str = "", level: str = "info",
             value: float = 0.0, payload: Any = None) -> None:
        self.outbox.put(Message(kind, text, level, value, payload))

    def run(self) -> None:  # pragma: no cover - exercised interactively
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            self.send("log", "".join(traceback.format_exc()), "fail")
            self.send("error", f"{type(exc).__name__}: {exc}", "fail")
        finally:
            self.send("done")

    def _run(self) -> None:
        import pandas as pd

        from conversation_analyst import models
        from conversation_analyst.config import Config
        from conversation_analyst.pipeline import Canceled, analyze_session
        from conversation_analyst.report.codebook import write_codebook
        from conversation_analyst.report.corpus import SessionEntry, write_corpus_report
        from conversation_analyst.report.dashboard import write_dashboard
        from conversation_analyst.report.qc import VERDICT_LABELS, assess_quality
        from conversation_analyst.report.exports import (
            transcript_table, transcript_words_table, write_corpus_exports,
        )
        from conversation_analyst.report.tables import measures_long, measures_wide, write_session_tables
        from conversation_analyst.session import iter_sessions

        config = Config()
        config.model_dir = self.model_dir
        config.tracking_workers = self.tracking_workers

        self.send("log", "Checking model assets...")
        missing = [r for r in models.status(self.model_dir) if not r["valid"]]
        if missing:
            self.send("log", f"Downloading {len(missing)} model file(s), first run only...")
            for row in missing:
                if self.stopping:
                    return
                self.send("log", f"  {row['name']} ({row['size_mb']} MB)")
                models.ensure(row["name"], self.model_dir)
        self.send("log", "Models ready.", "ok")

        # Discover leniently and skip what cannot be analyzed, rather than
        # letting one badly named pair abort the whole batch. Strict mode
        # raises on the first incomplete session, which would contradict the
        # folder scan's promise that such sessions are simply skipped.
        found = list(iter_sessions(self.target, strict=False))
        sessions = [s for s in found if s.has_close_pair]
        skipped = [s for s in found if not s.has_close_pair]
        self.send("log", f"Found {len(found)} session(s) in {self.target}")
        for session in skipped:
            self.send(
                "log",
                f"  skipping {session.session_id}: needs both close-up views, "
                f"has {', '.join(sorted(session.views))}",
                "warn",
            )
        if not sessions:
            self.send("log", "Nothing to analyze.", "warn")
            return

        output = Path(self.output)
        output.mkdir(parents=True, exist_ok=True)
        write_codebook(output / "codebook.csv")

        written: dict[str, Any] = {"codebook": output / "codebook.csv"}
        all_long: list[Any] = []
        all_transcripts: list[Any] = []
        all_words: list[Any] = []
        summary: list[dict] = []
        entries: list[Any] = []

        for index, session in enumerate(sessions, 1):
            if self.stopping:
                self.send("log", "Stopped before the next session.", "warn")
                break

            self.send("log", "")
            self.send("log", f"[{index}/{len(sessions)}] {session.session_id}", "accent")
            self.send("session", payload={"session_id": session.session_id,
                                          "verdict": "running"})

            def on_progress(name: str, i: int, total: int,
                            _index=index, _n=len(sessions)) -> None:
                # Progress spans the whole run, not just this session, so the
                # bar reflects how much work is actually left.
                span = 1.0 / max(_n, 1)
                self.send(
                    "progress",
                    text=f"{session.session_id}: {name.replace('_', ' ')}",
                    value=((_index - 1) * span + span * (i / max(total, 1))) * 100.0,
                )

            try:
                result = analyze_session(
                    session, config, output_root=output, skip=self.skip,
                    progress=on_progress, cancel=lambda: self.stopping,
                )
            except Canceled:
                self.send("log", "  stopped", "warn")
                break
            except Exception as exc:  # noqa: BLE001
                self.send("log", f"  FAILED: {type(exc).__name__}: {exc}", "fail")
                self.send("session", payload={"session_id": session.session_id,
                                              "verdict": "failed"})
                summary.append({"session_id": session.session_id, "verdict": "fail",
                                "error": f"{type(exc).__name__}: {exc}"})
                entries.append(SessionEntry(
                    session_id=session.session_id, verdict="fail",
                    error=f"{type(exc).__name__}: {exc}"))
                continue

            qc = assess_quality(result.context, result.sync)
            write_session_tables(
                result.workspace, session.session_id, result.context, result.measures
            )
            import json

            result.workspace.file("qc.json").write_text(
                json.dumps(qc.to_dict(), indent=2), encoding="utf-8"
            )
            dashboard = write_dashboard(
                result.workspace.file("dashboard.html"),
                result.context, result.measures, qc, result.stages, result.sync,
                video_paths=dict(session.views),
                offsets={r: result.sync.offset(r) for r in session.views}
                if result.sync else None,
            )

            available = sum(1 for m in result.measures if m.available)
            n_turns = len(result.context.turn_set.turns) if result.context.turn_set else 0
            level = {"pass": "ok", "pass_limits": "ok", "review": "warn",
                     "fail": "fail"}.get(qc.verdict, "warn")
            self.send(
                "log",
                f"  {VERDICT_LABELS.get(qc.verdict, qc.verdict.upper())} - "
                f"{available}/{len(result.measures)} values, "
                f"{n_turns} turns",
                level,
            )
            for stage in result.failed_stages:
                self.send("log", f"    stage '{stage.name}' failed: {stage.detail}", "fail")
            for check in qc.failures:
                self.send("log", f"    {check.severity}: {check.message}", "warn")

            self.send("session", payload={
                "session_id": session.session_id,
                "verdict": qc.verdict,
                "dashboard": str(dashboard),
                "turns": n_turns,
                "minutes": result.context.duration / 60.0,
                "values": f"{available}/{len(result.measures)}",
                "note": (qc.failures[0].message if qc.failures else ""),
            })

            entries.append(SessionEntry(
                session_id=session.session_id,
                verdict=qc.verdict,
                duration_s=result.context.duration,
                n_turns=n_turns,
                values_available=available,
                values_total=len(result.measures),
                dashboard=f"{session.session_id}/dashboard.html",
                failures=[(c.name, c.severity, c.message) for c in qc.failures],
                values={(m.id, m.person): m.value
                        for m in result.measures if m.available},
                unavailable={m.id: m.unavailable_reason for m in result.measures
                             if not m.available and m.unavailable_reason},
            ))

            all_long.append(
                measures_long(session.session_id, result.measures, result.context.metadata)
            )
            all_transcripts.append(transcript_table(session.session_id, result.context))
            all_words.append(transcript_words_table(session.session_id, result.context))
            summary.append({
                "session_id": session.session_id, "verdict": qc.verdict,
                "duration_s": round(result.context.duration, 1), "n_turns": n_turns,
                "values_available": available, "values_total": len(result.measures),
            })

        if all_long:
            combined = pd.concat(all_long, ignore_index=True)
            combined.to_csv(output / "measures_all.csv", index=False)
            measures_wide(combined).to_csv(output / "measures_all_wide.csv", index=False)
            self.send("log", "")
            self.send("log", f"Wrote {output / 'measures_all.csv'}", "ok")

            written["measures_all"] = output / "measures_all.csv"

            exports = write_corpus_exports(
                output, combined, sessions=summary,
                transcripts=all_transcripts, words=all_words,
            )
            written.update(exports)
            if "counts" in exports:
                self.send("log", f"Counts per person: {exports['counts']}", "ok")
            if "transcript_all" in exports:
                self.send("log", f"Transcript: {exports['transcript_all']}", "ok")
            self.send(
                "log",
                f"One file per measure family: {output / 'by-measure-family'}",
                "ok",
            )
        if summary:
            pd.DataFrame(summary).to_csv(output / "session_summary.csv", index=False)

        # The whole-run page. This is what a person opens first when they have
        # analyzed more than one conversation, so it is what the Open button
        # points at once it exists.
        if entries:
            report = write_corpus_report(
                output / "index.html", entries, title=Path(self.target).name,
                written=written,
            )
            self.send("log", f"Report for all {len(entries)} session(s): {report}", "ok")
            self.send("report", text=str(report))

        self.send("progress", text="Finished", value=100.0)


# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------


class App:
    """The main window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.queue: "queue.Queue[Message]" = queue.Queue()
        self.worker: Worker | None = None
        self.dashboards: dict[str, str] = {}
        self.report_path: str = ""
        self.verdicts: dict[str, str] = {}

        root.title(APP_TITLE)
        root.geometry("1180x900")
        root.minsize(880, 640)
        root.configure(bg=PALETTE["bg"])
        if sys.platform == "win32":  # dark title bar; harmless if it fails
            try:
                import ctypes

                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    ctypes.windll.user32.GetParent(root.winfo_id()),
                    20,  # DWMWA_USE_IMMERSIVE_DARK_MODE
                    ctypes.byref(ctypes.c_int(1)), 4,
                )
            except Exception:
                pass
        try:
            # Kept as an attribute deliberately: Tk holds no reference of
            # its own, and a garbage-collected PhotoImage blanks the icon.
            self._icon = _draw_icon()
            root.iconphoto(True, self._icon)
        except tk.TclError:  # pragma: no cover - display without RGBA icons
            pass

        self._init_style()
        self._build()
        self.root.after(80, self._drain)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- appearance ----------------------------------------------------
    def _init_style(self) -> None:
        style = ttk.Style()
        # 'clam' is the one theme present on every platform that actually
        # honours color options; the native themes ignore most of them.
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Segoe UI is the Windows system face and the lab machines run
        # Windows; elsewhere the toolkit's own defaults keep the app from
        # requesting a font the platform does not ship.
        if sys.platform == "win32":
            self.font_family, self.font_mono = "Segoe UI", "Consolas"
        else:
            self.font_family = tkfont.nametofont("TkDefaultFont").actual("family")
            self.font_mono = tkfont.nametofont("TkFixedFont").actual("family")
        family = self.font_family

        style.configure(".", background=PALETTE["bg"],
                        foreground=PALETTE["text"], font=(family, 10))
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("Card.TFrame", background=PALETTE["panel"])
        style.configure("TLabel", background=PALETTE["bg"], font=(family, 10))
        style.configure("Wordmark.TLabel", font=(family, 20, "bold"),
                        foreground=PALETTE["text"])
        style.configure("WordmarkAccent.TLabel", font=(family, 20, "bold"),
                        foreground=PALETTE["accent"])
        style.configure("Muted.TLabel", foreground=PALETTE["muted"],
                        font=(family, 9))
        style.configure("Card.TLabel", background=PALETTE["panel"],
                        font=(family, 10))
        style.configure("CardHint.TLabel", background=PALETTE["panel"],
                        foreground=PALETTE["muted"], font=(family, 9))
        style.configure("Summary.TLabel", background=PALETTE["panel"],
                        foreground=PALETTE["muted"], font=(family, 10))
        # The step number renders as a small accent chip: padding turns the
        # label's own background into the badge, no canvas drawing needed.
        # Teal demands dark text; white on teal fails contrast.
        style.configure("StepNumber.TLabel", background=PALETTE["accent"],
                        foreground=PALETTE["accent_ink"],
                        font=(family, 10, "bold"), padding=(7, 1))
        style.configure("StepTitle.TLabel", background=PALETTE["panel"],
                        font=(family, 11, "bold"))
        style.configure("TButton", font=(family, 10), padding=(12, 6),
                        background=PALETTE["panel_hi"],
                        foreground=PALETTE["text"],
                        bordercolor=PALETTE["line"],
                        lightcolor=PALETTE["panel_hi"],
                        darkcolor=PALETTE["panel_hi"], relief="flat")
        style.map("TButton",
                  background=[("active", PALETTE["line"])],
                  bordercolor=[("focus", PALETTE["accent"])])
        style.configure("Ghost.TButton", font=(family, 9), padding=(8, 3),
                        background=PALETTE["panel"])
        style.map("Ghost.TButton", background=[("active", PALETTE["panel_hi"])])
        style.configure("Card.TCheckbutton", background=PALETTE["panel"],
                        font=(family, 10),
                        indicatorbackground=PALETTE["field"],
                        indicatorforeground=PALETTE["accent"],
                        focuscolor=PALETTE["panel"])
        style.map(
            "Card.TCheckbutton",
            background=[("active", PALETTE["panel"])],
            indicatorbackground=[("selected", PALETTE["accent"]),
                                 ("active", PALETTE["accent_faint"])],
            indicatorforeground=[("selected", PALETTE["accent_ink"])],
        )
        style.configure("Card.TRadiobutton", background=PALETTE["panel"],
                        font=(family, 10),
                        indicatorbackground=PALETTE["field"],
                        indicatorforeground=PALETTE["accent"],
                        focuscolor=PALETTE["panel"])
        style.map(
            "Card.TRadiobutton",
            background=[("active", PALETTE["panel"])],
            indicatorbackground=[("selected", PALETTE["accent"]),
                                 ("active", PALETTE["accent_faint"])],
            indicatorforeground=[("selected", PALETTE["accent_ink"])],
        )
        style.configure("TEntry", padding=6,
                        fieldbackground=PALETTE["field"],
                        foreground=PALETTE["text"],
                        insertcolor=PALETTE["accent"],
                        bordercolor=PALETTE["line"],
                        lightcolor=PALETTE["line"], darkcolor=PALETTE["line"])
        style.map("TEntry", bordercolor=[("focus", PALETTE["accent"])],
                  lightcolor=[("focus", PALETTE["accent"])],
                  darkcolor=[("focus", PALETTE["accent"])])
        style.configure("Treeview", background=PALETTE["panel"],
                        fieldbackground=PALETTE["panel"],
                        foreground=PALETTE["text"], borderwidth=0,
                        rowheight=26, font=(family, 9))
        style.configure("Treeview.Heading", background=PALETTE["panel"],
                        foreground=PALETTE["muted"], borderwidth=0,
                        relief="flat", font=(family, 9, "bold"))
        style.map("Treeview.Heading",
                  background=[("active", PALETTE["panel"])])
        style.map("Treeview",
                  background=[("selected", PALETTE["accent_faint"])],
                  foreground=[("selected", PALETTE["text"])])
        style.configure("Horizontal.TProgressbar",
                        background=PALETTE["accent"],
                        troughcolor=PALETTE["field"], borderwidth=0,
                        thickness=8)
        style.configure("Vertical.TScrollbar",
                        background=PALETTE["panel_hi"],
                        troughcolor=PALETTE["bg"], bordercolor=PALETTE["bg"],
                        arrowcolor=PALETTE["muted"])

    # -- layout --------------------------------------------------------
    def _build_menu(self) -> None:
        """Every entry reuses a button's command; the menu adds no logic."""
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Choose recordings…", command=self._pick_input)
        file_menu.add_command(label="Choose results folder…",
                              command=self._pick_output)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Analyze", command=self._start)
        tools.add_command(label="Stop", command=self._stop)
        tools.add_separator()
        tools.add_command(label="Build demo session", command=self._make_demo)
        tools.add_command(label="Open results folder", command=self._open_output)
        tools.add_command(label="Open latest report", command=self._open_dashboard)
        menubar.add_cascade(label="Tools", menu=tools)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="Documentation — every decision, with citations",
            command=self._open_docs)
        help_menu.add_command(label="Code a session by hand…",
                              command=self._explain_coding)
        help_menu.add_command(label=f"About {APP_NAME}", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.configure(menu=menubar)

    def _card(self, parent: ttk.Frame, row: int, title: str,
              number: str | None = None,
              pady: tuple[int, int] = (0, 10)) -> ttk.Frame:
        """A bordered white panel with a heading; returns its content frame.

        The border is tk's highlight ring rather than a themed relief
        because it is the one 1px border whose color every platform that
        runs clam actually honours.
        """
        shell = tk.Frame(
            parent, bg=PALETTE["panel"],
            highlightbackground=PALETTE["line"],
            highlightcolor=PALETTE["line"], highlightthickness=1)
        shell.grid(row=row, column=0, sticky="nsew", pady=pady)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        body = ttk.Frame(shell, style="Card.TFrame", padding=14)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        head = ttk.Frame(body, style="Card.TFrame")
        head.grid(row=0, column=0, sticky="w", pady=(0, 8))
        if number:
            ttk.Label(head, text=number, style="StepNumber.TLabel").pack(side="left")
        ttk.Label(head, text=title, style="StepTitle.TLabel").pack(
            side="left", padx=(8 if number else 0, 0))

        content = ttk.Frame(body, style="Card.TFrame")
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        return content

    def _build(self) -> None:
        self._build_menu()

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        self.outer = outer

        # -- identity ----------------------------------------------------
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        wordmark = ttk.Frame(header)
        wordmark.pack(anchor="w")
        ttk.Label(wordmark, text="Conversation", style="Wordmark.TLabel").pack(side="left")
        ttk.Label(wordmark, text=" Analyst", style="WordmarkAccent.TLabel").pack(side="left")
        ttk.Label(header, text=APP_TAGLINE, style="Muted.TLabel").pack(
            anchor="w", pady=(1, 0))

        # -- step 1: recordings -------------------------------------------
        paths = self._card(outer, row=1, number="1", title="Recordings")
        paths.columnconfigure(1, weight=1)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(default_results_dir()))

        ttk.Label(paths, text="Videos folder", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=4)
        ttk.Entry(paths, textvariable=self.input_var).grid(
            row=0, column=1, sticky="ew", padx=10)
        ttk.Button(paths, text="Browse...", command=self._pick_input).grid(
            row=0, column=2)
        ttk.Button(paths, text="Use demo data", command=self._make_demo).grid(
            row=0, column=3, padx=(6, 0))

        ttk.Label(paths, text="Results folder", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=4)
        ttk.Entry(paths, textvariable=self.output_var).grid(
            row=1, column=1, sticky="ew", padx=10)
        ttk.Button(paths, text="Browse...", command=self._pick_output).grid(
            row=1, column=2)

        ttk.Label(
            paths,
            text="Each conversation is two videos -- one close-up per person; "
                 "both carry both voices, which is expected. Name the pair "
                 "with a shared id and a person token, e.g. "
                 "dyad012_close_a.mp4 and dyad012_close_b.mp4",
            style="CardHint.TLabel",
            # Without a wrap length this label is a single long line, and its
            # requested width becomes the window's minimum -- which pushed the
            # buttons on the right off the edge of the window.
            wraplength=880,
            justify="left",
        ).grid(row=2, column=1, columnspan=3, sticky="w", padx=10, pady=(2, 0))

        # -- step 2: what to measure ---------------------------------------
        options = self._card(outer, row=2, number="2", title="What to measure")
        self.stage_vars: dict[str, tk.BooleanVar] = {}
        for i, (key, label, hint) in enumerate(SKIPPABLE):
            var = tk.BooleanVar(value=True)
            self.stage_vars[key] = var
            row, col = divmod(i, 2)
            options.columnconfigure(col, weight=1, uniform="measure")
            cell = ttk.Frame(options, style="Card.TFrame")
            cell.grid(row=row, column=col, sticky="w", padx=(0, 24), pady=4)
            ttk.Checkbutton(cell, text=label, variable=var,
                            style="Card.TCheckbutton").pack(anchor="w")
            ttk.Label(cell, text=hint, style="CardHint.TLabel").pack(
                anchor="w", padx=(24, 0))

        # Face and body landmarking are 93% of the runtime, and how many run
        # at once is the only large lever on it. The automatic setting reads
        # free memory and is often forced down to one worker by whatever else
        # is open, so the choice is put in front of the person waiting for
        # the run rather than buried in a config file.
        speed = ttk.Frame(options, style="Card.TFrame")
        speed.grid(row=(len(SKIPPABLE) + 1) // 2, column=0, columnspan=2,
                   sticky="w", pady=(12, 0))
        ttk.Label(speed, text="Speed", style="Card.TLabel").pack(anchor="w")
        self.workers_var = tk.StringVar(value="auto")
        choices = ttk.Frame(speed, style="Card.TFrame")
        choices.pack(anchor="w")
        for value, label in (
            ("auto", "Automatic"),
            ("4", "Fastest (4 at once)"),
            ("1", "Gentlest (one at a time)"),
        ):
            ttk.Radiobutton(
                choices, text=label, value=value, variable=self.workers_var,
                style="Card.TRadiobutton",
            ).pack(side="left", padx=(0, 18))
        ttk.Label(
            speed,
            text="Tracking faces and bodies is most of the wait. Running all "
                 "four jobs at once roughly halves it, and needs about 1.5 GB "
                 "free — close other applications first. Automatic picks "
                 "from the memory available when the run starts.",
            style="CardHint.TLabel", wraplength=760, justify="left",
        ).pack(anchor="w", pady=(4, 0))

        # -- step 3: analyze -----------------------------------------------
        run = self._card(outer, row=3, number="3", title="Analyze")
        actions = ttk.Frame(run, style="Card.TFrame")
        actions.grid(row=0, column=0, sticky="ew")

        # A plain tk.Button rather than ttk: the themed engines quietly
        # ignore background requests on half the platforms, and the one
        # button that starts the run is the one place the accent must
        # actually show up.
        self.run_button = tk.Button(
            actions, text="Analyze", command=self._start,
            background=PALETTE["accent"], foreground=PALETTE["accent_ink"],
            activebackground=PALETTE["accent_dark"],
            activeforeground=PALETTE["accent_ink"],
            disabledforeground="#3F6B66",
            font=(self.font_family, 12, "bold"),
            relief="flat", borderwidth=0, cursor="hand2",
            padx=32, pady=8,
        )
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(
            actions, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(12, 0))
        ttk.Button(actions, text="Open results folder",
                   command=self._open_output).pack(side="left", padx=(8, 0))
        # The one thing worth doing after a run, so it sits alone on the right
        # where nothing can crowd it off the edge of the window.
        self.dashboard_button = ttk.Button(
            actions, text="Open report", command=self._open_dashboard, state="disabled")
        self.dashboard_button.pack(side="right")

        self.progress = ttk.Progressbar(run, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        status_row = ttk.Frame(run, style="Card.TFrame")
        status_row.grid(row=2, column=0, sticky="w", pady=(6, 0))
        # The status dot breathes while a run is live -- a 12 px canvas
        # animated by the same ticker that eases the progress bar, so a
        # stalled interface is visibly distinguishable from a working one.
        self.status_dot = tk.Canvas(
            status_row, width=12, height=12, highlightthickness=0,
            background=PALETTE["panel"])
        self.status_dot.pack(side="left", padx=(0, 7))
        self._dot_item = self.status_dot.create_oval(
            2, 2, 10, 10, fill=PALETTE["muted"], outline="")
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_row, textvariable=self.status_var,
                  style="CardHint.TLabel").pack(side="left")

        self._progress_target = 0.0
        self._pulse_phase = 0.0
        self.root.after(33, self._animate_tick)

        # -- results ----------------------------------------------------
        #
        # A running log is the right place for detail and the wrong place for
        # state: by the time eight sessions have finished, "which ones came
        # out badly" has scrolled away. The table holds the answer to that
        # question and stays put; the summary line above it answers it in one
        # glance; the log keeps the detail underneath, folded away.
        outer.rowconfigure(4, weight=3)
        results_card = self._card(outer, row=4, title="Results", pady=(0, 0))
        results_card.rowconfigure(1, weight=1)

        self.summary_var = tk.StringVar(value="No sessions analyzed yet.")
        ttk.Label(results_card, textvariable=self.summary_var,
                  style="Summary.TLabel").grid(row=0, column=0, sticky="w",
                                               pady=(0, 6))

        table = ttk.Frame(results_card, style="Card.TFrame")
        table.grid(row=1, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        columns = ("verdict", "minutes", "turns", "values", "note")
        self.results = ttk.Treeview(
            table, columns=columns, show="tree headings", height=6,
            selectmode="browse",
        )
        self.results.heading("#0", text="Session")
        self.results.column("#0", width=130, stretch=False)
        for key, label, width, anchor in (
            ("verdict", "Verdict", 84, "center"),
            ("minutes", "Minutes", 70, "e"),
            ("turns", "Turns", 60, "e"),
            ("values", "Values", 74, "e"),
            ("note", "What was flagged", 420, "w"),
        ):
            self.results.heading(key, text=label)
            self.results.column(key, width=width, anchor=anchor,
                                stretch=(key == "note"))
        self.results.grid(row=0, column=0, sticky="nsew")
        rscroll = ttk.Scrollbar(table, orient="vertical",
                                command=self.results.yview)
        rscroll.grid(row=0, column=1, sticky="ns")
        self.results.configure(yscrollcommand=rscroll.set)
        self.results.tag_configure("pass", foreground=PALETTE["ok"])
        self.results.tag_configure("review", foreground=PALETTE["warn"])
        self.results.tag_configure("fail", foreground=PALETTE["fail"])
        # The worker reports a crashed session as "failed"; without a tag of
        # its own those rows would render in plain ink and look healthy.
        self.results.tag_configure("failed", foreground=PALETTE["fail"])
        self.results.tag_configure("running", foreground=PALETTE["muted"])
        # Double-clicking a row opens that session's own report.
        self.results.bind("<Double-1>", self._open_selected)

        # -- activity log -------------------------------------------------
        self._log_row = 6
        self.log_toggle = ttk.Button(
            outer, text="Hide activity log", command=self._toggle_log,
            style="Ghost.TButton")
        self.log_toggle.grid(row=5, column=0, sticky="w", pady=(10, 4))

        self.log_shell = tk.Frame(
            outer, bg=PALETTE["panel"],
            highlightbackground=PALETTE["line"],
            highlightcolor=PALETTE["line"], highlightthickness=1)
        self.log_shell.grid(row=self._log_row, column=0, sticky="nsew")
        self.log_shell.columnconfigure(0, weight=1)
        self.log_shell.rowconfigure(0, weight=1)

        self.log = tk.Text(
            self.log_shell, wrap="word", height=10, borderwidth=0,
            font=(self.font_mono, 9), background=PALETTE["panel"],
            foreground=PALETTE["text"], padx=12, pady=10, state="disabled",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.log_shell, orient="vertical",
                               command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

        for tag, color in (("ok", PALETTE["ok"]), ("warn", PALETTE["warn"]),
                            ("fail", PALETTE["fail"]), ("accent", PALETTE["accent"]),
                            ("info", PALETTE["text"])):
            self.log.tag_configure(tag, foreground=color)
        self.log.tag_configure("accent", font=(self.font_mono, 9, "bold"))

        # Collapsed by default so the resting view stays calm. Toggling down
        # from the built (shown) state also puts the right label on the
        # button, so there is exactly one code path for both directions.
        self._log_visible = True
        self._toggle_log()

        self._log("Ready. Choose a folder of recordings, or click "
                  "'Use demo data' to try it without any.", "muted")

    def _toggle_log(self) -> None:
        """Fold the log away; hiding is geometric, nothing is lost.

        The Text widget always exists and always receives lines, so tests
        and the queue pump never care whether it is on screen. The row
        weight has to move with it: an empty weighted row would otherwise
        hold a blank band of window where the log used to be.
        """
        if self._log_visible:
            self.log_shell.grid_remove()
            self.outer.rowconfigure(self._log_row, weight=0)
            self.log_toggle.configure(text="Show activity log")
        else:
            self.log_shell.grid()
            self.outer.rowconfigure(self._log_row, weight=2)
            self.log_toggle.configure(text="Hide activity log")
        self._log_visible = not self._log_visible

    # -- actions -------------------------------------------------------
    def _pick_input(self) -> None:
        path = filedialog.askdirectory(title="Folder containing dyad recordings")
        if path:
            self.input_var.set(path)
            self._scan(path)

    def _scan(self, path: str) -> None:
        """Report what was found as soon as a folder is chosen.

        Filename conventions are the most common thing to get wrong, and
        finding out after a forty-minute run has started is no use. Discovery
        is cheap -- it only stats files -- so it runs immediately.
        """
        from conversation_analyst.session import SessionError, iter_sessions

        try:
            sessions = list(iter_sessions(path, strict=False))
        except SessionError as exc:
            self._log(f"Could not read that folder: {exc}", "fail")
            self.status_var.set("Could not read that folder.")
            return

        if not sessions:
            self._log("No video files found in that folder.", "warn")
            self.status_var.set("No video files found in that folder.")
            return

        complete = [s for s in sessions if s.has_close_pair]
        self._log("")
        self._log(f"Found {len(sessions)} session(s) in {path}", "accent")
        for session in sessions[:12]:
            views = ", ".join(sorted(session.views))
            level = "ok" if session.has_close_pair else "warn"
            note = "" if session.has_close_pair else "   <- needs both close-up views"
            self._log(f"  {session.session_id}: {views}{note}", level)
        if len(sessions) > 12:
            self._log(f"  ... and {len(sessions) - 12} more")

        if len(complete) < len(sessions):
            self._log(
                f"{len(sessions) - len(complete)} session(s) are missing a close-up "
                "view and will be skipped. Speaker attribution needs both.",
                "warn",
            )

        # The log is collapsed by default, so the scan's verdict has to
        # reach the always-visible status line as well.
        self.status_var.set(
            f"Found {len(sessions)} session(s); {len(complete)} ready to analyze."
        )

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Where to write results")
        if path:
            self.output_var.set(path)

    def _make_demo(self) -> None:
        """Generate a synthetic session so the app can be tried with no data."""
        from conversation_analyst.synth import tts_available

        if not tts_available():
            messagebox.showinfo(
                "Demo unavailable",
                "The demo builds a conversation using the system speech voices, "
                "which are only available on Windows.\n\n"
                "Point the app at a folder of real recordings instead.",
            )
            return
        if self.worker and self.worker.is_alive():
            return

        target = Path(self.output_var.get()) / "demo_media"
        self._log("")
        self._log("Building a synthetic conversation (about 30 seconds)...", "accent")
        self.run_button.configure(state="disabled")

        def build() -> None:
            try:
                from conversation_analyst.synth import build_script, render_session
                from conversation_analyst.synth.media import write_session

                session = render_session(plan=build_script(n_turns=16, seed=5), seed=5)
                write_session(
                    session, target, session_id="demo",
                    offsets={"close_a": 0.0, "close_b": 1.7},
                )
                self.queue.put(Message(
                    "log", f"Wrote two views to {target}", "ok"))
                self.queue.put(Message("demo_ready", str(target)))
            except Exception as exc:  # noqa: BLE001
                self.queue.put(Message("log", f"Demo failed: {exc}", "fail"))
                self.queue.put(Message("demo_ready", ""))

        threading.Thread(target=build, daemon=True).start()

    def _start(self) -> None:
        target = self.input_var.get().strip()
        if not target:
            messagebox.showwarning("No recordings", "Choose a folder of recordings first.")
            return
        if not Path(target).exists():
            messagebox.showerror("Not found", f"This path does not exist:\n{target}")
            return

        skip = tuple(k for k, var in self.stage_vars.items() if not var.get())
        self.dashboards.clear()
        self.report_path = ""
        self.verdicts.clear()
        self._update_summary()
        for row in self.results.get_children():
            self.results.delete(row)
        self.dashboard_button.configure(state="disabled")
        self.progress.configure(value=0)
        self._log("")
        self._log("=" * 60, "muted")

        self.worker = Worker(
            target=target,
            output=self.output_var.get().strip() or str(default_results_dir()),
            skip=skip,
            # Models live in one place per user, not beside the results, so
            # that changing the output folder never triggers a re-download.
            model_dir=str(MODEL_DIR),
            outbox=self.queue,
            lenient=False,
            tracking_workers=self._chosen_workers(),
        )
        self.worker.start()
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Starting...")

    def _chosen_workers(self) -> int | None:
        """The Speed setting, as the config wants it: None means automatic."""
        value = getattr(self, "workers_var", None)
        if value is None:
            return None
        raw = value.get()
        return None if raw == "auto" else int(raw)

    def _stop(self) -> None:
        if self.worker:
            self.worker.request_stop()
            self.stop_button.configure(state="disabled")
            self.status_var.set("Stopping after the current stage...")
            self._log("Stop requested; finishing the current stage first.", "warn")

    def _open_output(self) -> None:
        path = Path(self.output_var.get())
        path.mkdir(parents=True, exist_ok=True)
        webbrowser.open(path.as_uri())

    def _open_dashboard(self) -> None:
        """Open the whole-run report, falling back to the latest session.

        The corpus page is the right destination once more than one
        conversation has been analyzed: it links every session and shows the
        distributions, so it answers "how did the run go" rather than "how
        did the last one go".
        """
        if self.report_path:
            webbrowser.open(Path(self.report_path).as_uri())
        elif self.dashboards:
            webbrowser.open(Path(list(self.dashboards.values())[-1]).as_uri())

    def _open_selected(self, _event=None) -> None:
        selected = self.results.focus()
        target = self.dashboards.get(selected)
        if target:
            webbrowser.open(Path(target).as_uri())

    def _animate_tick(self) -> None:
        """One 30 fps frame: ease the bar, breathe the dot.

        Easing rather than jumping is not decoration on a tool whose runs
        take minutes: a bar that glides toward each stage boundary reads as
        alive between updates, where a bar that leaps every forty seconds
        reads as hung the rest of the time.
        """
        try:
            current = float(self.progress["value"])
            target = self._progress_target
            if abs(target - current) > 0.05:
                self.progress.configure(value=current + (target - current) * 0.18)
            running = self.worker is not None and self.worker.is_alive()
            if running:
                import math

                self._pulse_phase = (self._pulse_phase + 0.10) % (2 * math.pi)
                # Breathe between the faint and full accent by mixing the
                # two hex colors -- tkinter has no alpha, so mix by hand.
                mix = 0.5 + 0.5 * math.sin(self._pulse_phase)
                lo = (18, 59, 55)     # accent_faint
                hi = (45, 212, 191)   # accent
                color = "#%02x%02x%02x" % tuple(
                    int(a + (b - a) * mix) for a, b in zip(lo, hi)
                )
                self.status_dot.itemconfigure(self._dot_item, fill=color)
            else:
                self.status_dot.itemconfigure(
                    self._dot_item,
                    fill=PALETTE["ok"] if self._progress_target >= 100
                    else PALETTE["muted"],
                )
        except tk.TclError:  # window closing
            return
        self.root.after(33, self._animate_tick)

    def _open_docs(self) -> None:
        """Generate and open the documentation of record.

        Generated fresh on each open so the page always matches the
        installed version's registry -- every decision, its reasoning, its
        citations, and the full measure catalogue.
        """
        from conversation_analyst.report.documentation import write_documentation

        target = MODEL_DIR.parent / "documentation.html"
        try:
            write_documentation(target)
        except OSError as exc:
            messagebox.showinfo("Documentation", f"Could not write the page: {exc}")
            return
        webbrowser.open(target.resolve().as_uri())

    def _explain_coding(self) -> None:
        messagebox.showinfo(
            "Code a session by hand",
            "Every analyzed session gets a coding.html next to its "
            "dashboard.\n\nOpen it, mark nods, smiles and laughs with "
            "single keys while the video plays (the page shows no "
            "detections, on purpose), export the file, then run:\n\n"
            "  conversation-analyst agreement <exported file>\n\n"
            "to get event F1, onset error and Cohen's kappa against the "
            "detectors. Twenty minutes of coding turns 'does it agree with "
            "a person?' from an open question into a number.",
        )

    def _show_about(self) -> None:
        # Imported here, not at module top: the package __init__ pulls in
        # config and session, which the GUI otherwise defers to the worker.
        from conversation_analyst import __version__

        messagebox.showinfo(
            f"About {APP_NAME}",
            f"{APP_NAME} {__version__}\n\n"
            "Turns paired close-up recordings of a two-person conversation "
            "into a documented table of behavioral measures.\n\n"
            f"{APP_TAGLINE}\n\n"
            "Built on faster-whisper, Silero VAD, MediaPipe, "
            "Praat/parselmouth, sentence-transformers.",
        )

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askokcancel(
                "Quit", "Analysis is still running. Stop it and quit?"
            ):
                return
            self.worker.request_stop()
        self.root.destroy()

    def _update_summary(self) -> None:
        """One line above the table so "how did the run go" needs no scan."""
        if not self.verdicts:
            self.summary_var.set("No sessions analyzed yet.")
            return
        counts: dict[str, int] = {}
        for verdict in self.verdicts.values():
            # The worker says "failed" for a crashed session and "fail" for
            # a quality verdict; the distinction matters in the row's note,
            # not in a headcount.
            key = "fail" if verdict == "failed" else verdict
            counts[key] = counts.get(key, 0) + 1
        total = len(self.verdicts)
        parts = [f"{counts[key]} {key}"
                 for key in ("pass", "review", "fail", "running")
                 if counts.get(key)]
        noun = "session" if total == 1 else "sessions"
        self.summary_var.set(f"{total} {noun} — " + ", ".join(parts))

    # -- queue pump ----------------------------------------------------
    def _log(self, text: str, level: str = "info") -> None:
        tag = "info" if level == "muted" else level
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain(self) -> None:
        try:
            while True:
                message = self.queue.get_nowait()
                self._handle(message)
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    def _handle(self, message: Message) -> None:
        if message.kind == "log":
            self._log(message.text, message.level)
        elif message.kind == "progress":
            self._progress_target = float(message.value)
            self.status_var.set(message.text)
        elif message.kind == "session":
            info = message.payload or {}
            session_id = info["session_id"]
            verdict = info.get("verdict", "running")
            values = (
                verdict.upper() if verdict != "running" else "running...",
                f"{info['minutes']:.1f}" if info.get("minutes") else "",
                str(info.get("turns", "")),
                info.get("values", ""),
                info.get("note", ""),
            )
            if self.results.exists(session_id):
                self.results.item(session_id, values=values, tags=(verdict,))
            else:
                self.results.insert(
                    "", "end", iid=session_id, text=session_id,
                    values=values, tags=(verdict,),
                )
            self.results.see(session_id)
            self.verdicts[session_id] = verdict
            self._update_summary()
            dashboard = info.get("dashboard")
            if dashboard:
                self.dashboards[session_id] = dashboard
                self.dashboard_button.configure(state="normal")
        elif message.kind == "report":
            self.report_path = message.text
            self.dashboard_button.configure(state="normal", text="Open report")
        elif message.kind == "demo_ready":
            self.run_button.configure(state="normal")
            if message.text:
                self.input_var.set(message.text)
                self._scan(message.text)
                self._log("Demo data ready - click Analyze.", "ok")
        elif message.kind == "error":
            messagebox.showerror("Analysis failed", message.text)
        elif message.kind == "done":
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_var.set("Finished.")
            self.worker = None


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``convlab gui`` and the ``convlab-gui`` script.

    Startup failures are caught and shown. The Windows launcher starts the
    app with ``pythonw``, which has no console, so an uncaught exception here
    would produce absolutely nothing -- no window, no message, no traceback.
    "I double-clicked it and nothing happened" is the least diagnosable bug
    report there is, so any crash is written to a log file beside the app and
    surfaced in a dialog.
    """
    try:
        root = tk.Tk()
        App(root)
        root.mainloop()
        return 0
    except Exception:  # noqa: BLE001
        report = traceback.format_exc()
        log_path = Path.home() / ".convlab" / "crash.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(report, encoding="utf-8")
        except OSError:  # pragma: no cover - unwritable home
            log_path = None

        sys.stderr.write(report)
        try:
            messagebox.showerror(
                "convlab could not start",
                f"{report.strip().splitlines()[-1]}\n\n"
                + (f"Full details written to:\n{log_path}" if log_path else ""),
            )
        except Exception:  # pragma: no cover - no display at all
            pass
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
