"""Command line interface.

    conversation-analyst analyze RECORDINGS/ -o workspace/
    conversation-analyst models fetch
    conversation-analyst codebook -o docs/measures.md
    conversation-analyst demo -o workspace/
    conversation-analyst validate -o validation/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from conversation_analyst import __version__


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else (
        logging.INFO if verbosity == 1 else logging.DEBUG
    )
    logging.basicConfig(
        level=level, format="%(levelname)-7s %(name)s: %(message)s", stream=sys.stderr
    )
    for noisy in ("urllib3", "matplotlib", "PIL", "numba", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ----------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    from conversation_analyst.config import Config
    from conversation_analyst.pipeline import analyze_session
    from conversation_analyst.report.codebook import write_codebook
    from conversation_analyst.report.corpus import SessionEntry, write_corpus_report
    from conversation_analyst.report.dashboard import write_dashboard
    from conversation_analyst.report.qc import VERDICT_LABELS, assess_quality
    from conversation_analyst.report.tables import measures_long, write_session_tables
    from conversation_analyst.session import iter_sessions

    import pandas as pd

    config = Config.load(args.config) if args.config else Config()
    if args.model_dir:
        config.model_dir = args.model_dir

    sessions = list(iter_sessions(args.target, strict=not args.lenient))
    print(f"Found {len(sessions)} session(s)")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_codebook(output / "codebook.csv")

    written: dict[str, Path] = {"codebook": output / "codebook.csv"}
    all_long: list[pd.DataFrame] = []
    all_transcripts: list[pd.DataFrame] = []
    all_words: list[pd.DataFrame] = []
    summary: list[dict] = []
    entries: list[SessionEntry] = []

    for index, session in enumerate(sessions, 1):
        print(f"\n[{index}/{len(sessions)}] {session.describe()}")
        started = time.perf_counter()
        try:
            result = analyze_session(
                session, config, output_root=output, skip=tuple(args.skip or ())
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            summary.append(
                {"session_id": session.session_id, "verdict": "fail",
                 "error": f"{type(exc).__name__}: {exc}"}
            )
            entries.append(
                SessionEntry(
                    session_id=session.session_id, verdict="fail",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        qc = assess_quality(result.context, result.sync)
        paths = write_session_tables(
            result.workspace, session.session_id, result.context, result.measures
        )
        result.workspace.file("qc.json").write_text(
            json.dumps(qc.to_dict(), indent=2), encoding="utf-8"
        )
        write_dashboard(
            result.workspace.file("dashboard.html"),
            result.context, result.measures, qc, result.stages, result.sync,
            video_paths=dict(session.views),
            offsets={r: result.sync.offset(r) for r in session.views}
            if result.sync else None,
        )
        from conversation_analyst.report.coding import write_coding_page

        write_coding_page(
            result.workspace.file("coding.html"),
            result.context,
            video_paths=dict(session.views),
            offsets={r: result.sync.offset(r) for r in session.views}
            if result.sync else None,
        )

        all_long.append(
            measures_long(session.session_id, result.measures, result.context.metadata)
        )
        from conversation_analyst.report.exports import (
            transcript_table, transcript_words_table,
        )

        all_transcripts.append(transcript_table(session.session_id, result.context))
        all_words.append(transcript_words_table(session.session_id, result.context))
        available = sum(1 for m in result.measures if m.available)
        elapsed = time.perf_counter() - started
        print(
            f"  {VERDICT_LABELS.get(qc.verdict, qc.verdict.upper())} - "
            f"{available}/{len(result.measures)} values, "
            f"{len(result.context.turn_set.turns) if result.context.turn_set else 0} turns, "
            f"{elapsed:.0f}s"
        )
        for stage in result.failed_stages:
            print(f"    stage '{stage.name}' failed: {stage.detail}")
        for check in qc.failures:
            print(f"    QC {check.severity}: {check.message}")
        print(f"    -> {paths.get('measures')}")

        entries.append(
            SessionEntry(
                session_id=session.session_id,
                verdict=qc.verdict,
                duration_s=result.context.duration,
                n_turns=len(result.context.turn_set.turns)
                if result.context.turn_set else 0,
                values_available=available,
                values_total=len(result.measures),
                seconds=elapsed,
                dashboard=f"{session.session_id}/dashboard.html",
                failures=[
                    (c.name, c.severity, c.message) for c in qc.failures
                ],
                values={
                    (m.id, m.person): m.value
                    for m in result.measures if m.available
                },
                unavailable={
                    m.id: m.unavailable_reason
                    for m in result.measures
                    if not m.available and m.unavailable_reason
                },
            )
        )

        summary.append(
            {
                "session_id": session.session_id,
                "verdict": qc.verdict,
                "duration_s": round(result.context.duration, 1),
                "n_turns": len(result.context.turn_set.turns) if result.context.turn_set else 0,
                "values_available": available,
                "values_total": len(result.measures),
                "seconds": round(elapsed, 1),
                "warnings": len(result.context.warnings),
            }
        )

    if all_long:
        combined = pd.concat(all_long, ignore_index=True)
        combined.to_csv(output / "measures_all.csv", index=False)
        from conversation_analyst.report.exports import write_corpus_exports
        from conversation_analyst.report.tables import measures_wide

        measures_wide(combined).to_csv(output / "measures_all_wide.csv", index=False)
        print(f"\nCombined -> {output / 'measures_all.csv'}")

        written["measures_all"] = output / "measures_all.csv"

        exports = write_corpus_exports(
            output, combined, sessions=summary,
            transcripts=all_transcripts, words=all_words,
        )
        written.update(exports)
        if "counts" in exports:
            print(f"Counts per person -> {exports['counts']}")
        if "transcript_all" in exports:
            print(f"Transcript -> {exports['transcript_all']}")
        print(f"One file per measure family -> {output / 'by-measure-family'}")

    pd.DataFrame(summary).to_csv(output / "session_summary.csv", index=False)

    # One page for the whole run. Opening eight dashboards in turn is how a
    # problem in session five gets missed.
    if entries:
        index = write_corpus_report(
            output / "index.html", entries, title=Path(args.target).name,
            written=written,
        )
        print(f"\nCorpus report -> {index}")

    passed = sum(1 for s in summary if s.get("verdict") in ("pass", "pass_limits"))
    print(f"Summary: {passed}/{len(summary)} sessions passed QC "
          f"-> {output / 'session_summary.csv'}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from conversation_analyst import models

    if args.action == "fetch":
        for name in models.REGISTRY:
            path = models.ensure(name, args.model_dir)
            print(f"  {name:18s} -> {path}")
        return 0

    rows = models.status(args.model_dir)
    width = max(len(r["name"]) for r in rows)
    for row in rows:
        state = "ok" if row["valid"] else ("corrupt" if row["present"] else "missing")
        print(f"  {row['name']:<{width}}  {state:8s} {row['size_mb']:>6.1f} MB  "
              f"{row['purpose']}")
    return 0 if all(r["valid"] for r in rows) else 1


def cmd_codebook(args: argparse.Namespace) -> int:
    from conversation_analyst.report.codebook import build_codebook, codebook_markdown

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".md":
        output.write_text(codebook_markdown(), encoding="utf-8")
    else:
        build_codebook().to_csv(output, index=False)
    frame = build_codebook()
    print(f"{len(frame)} measures in {frame['family'].nunique()} families -> {output}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Generate a synthetic session, write it as video, and analyze it."""
    from conversation_analyst.synth import build_script, render_session, tts_available
    from conversation_analyst.synth.media import write_session

    if not tts_available():
        print("Speech synthesis is unavailable on this platform; demo needs Windows.")
        return 2

    output = Path(args.output)
    media = output / "demo_media"
    print("Rendering synthetic conversation...")
    session_audio = render_session(
        plan=build_script(n_turns=args.turns, seed=args.seed), seed=args.seed
    )
    print(f"  {session_audio.duration:.0f}s, {len(session_audio.turns)} turns")

    faces = {}
    if args.faces:
        faces = {"A": args.faces[0], "B": args.faces[-1], "wide": args.faces[0]}

    roles = ("close_a", "close_b", "wide") if args.wide else ("close_a", "close_b")
    # Cameras started by hand never line up; offsetting the written files
    # means the demo actually exercises the alignment stage.
    write_session(
        session_audio, media, session_id="demo", face_videos=faces, roles=roles,
        offsets={"close_a": 0.0, "close_b": 1.7, "wide": 0.4},
    )
    print(f"  wrote {len(roles)} views to {media}")
    print(
        "  note: the demo's picture is a still placeholder, so the report will\n"
        "        warn that the video is frozen and no face was tracked. That is\n"
        "        the quality checks working; the demo exercises the audio chain."
    )

    args.target = str(media)
    args.lenient = False
    return cmd_analyze(args)


def cmd_gui(args: argparse.Namespace) -> int:
    from conversation_analyst.gui import main as gui_main

    return gui_main()


def cmd_validate(args: argparse.Namespace) -> int:
    from conversation_analyst.validation import run_validation

    report = run_validation(
        output_dir=Path(args.output), seeds=tuple(args.seeds), quick=args.quick
    )
    print(report.render_text())
    return 0 if report.passed else 1


def cmd_validate_study(args: argparse.Namespace) -> int:
    """Compare pipeline output against the study's own records."""
    from conversation_analyst.study import run_study_validation, write_study_report

    openface = args.openface
    if openface is None:
        # The lab's OpenFace export, if it sits where downloads land.
        candidate = Path.home() / "Downloads" / "CSV Files Dyad.zip"
        openface = str(candidate) if candidate.exists() else None
    result = run_study_validation(args.study, args.workspace, openface_zip=openface)
    path = write_study_report(result, args.workspace)

    print(f"Praat convergence over {result.n_sessions} session(s):")
    for _, row in result.convergence_by_stat.iterrows():
        print(
            f"  {row.statistic:6s} r={row.r:+.3f}  "
            f"median |err| {row.median_abs_error_hz:6.2f} Hz  "
            f"(lab {row.lab_mean:6.1f}, ours {row.our_mean:6.1f})"
        )
    if len(result.openface):
        for _, row in (result.openface.groupby("signal")
                       .r_vs_openface.median().reset_index().iterrows()):
            print(f"  OpenFace convergence {row.signal:11s} median |r| = {row.r_vs_openface:.3f}")
    if len(result.criterion_panel):
        consistent = (result.criterion_panel.consistent == "yes").sum()
        directional = (result.criterion_panel.consistent != "n/a").sum()
        print(f"Criterion panel: {consistent}/{directional} in the predicted direction")
    for note in result.notes:
        print(f"  NOTE: {note}")
    print(f"-> {path}")
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    """Write the documentation-of-record page."""
    from conversation_analyst.report.documentation import write_documentation

    path = write_documentation(args.output)
    print(f"-> {path}")
    if args.open:
        import webbrowser

        webbrowser.open(Path(path).resolve().as_uri())
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    """Score a human coder's file against the detectors."""
    from conversation_analyst.agreement import (
        load_human_coding, load_machine_events, score_agreement,
    )

    human, meta = load_human_coding(args.coding_file)
    session_id = meta.get("session", "?")
    session_dir = Path(args.workspace) / str(session_id)
    machine, duration = load_machine_events(session_dir)
    duration = float(meta.get("duration") or duration)

    results = score_agreement(human, machine, duration, tolerance_s=args.tolerance)
    if not results:
        print("No overlapping behaviors between the coder's file and the "
              "machine's events.")
        return 1

    coder = meta.get("coder", "anonymous")
    print(f"Agreement for session {session_id}, coder {coder} "
          f"(tolerance ±{args.tolerance:.1f}s):")
    print(f"  {'behavior':12s} {'who':3s} {'human':>6s} {'machine':>8s} "
          f"{'F1':>6s} {'recall':>7s} {'precision':>10s} {'onset MAE':>10s} {'kappa':>7s}")
    for r in sorted(results, key=lambda x: (x.behavior, x.person)):
        mae = f"{r.boundary_mae_s*1000:.0f} ms" if r.boundary_errors_s else "--"
        print(f"  {r.behavior:12s} {r.person:3s} {r.n_human:6d} {r.n_machine:8d} "
              f"{r.f1:6.2f} {r.recall:7.2f} {r.precision:10.2f} {mae:>10s} "
              f"{r.kappa:7.2f}")

    import pandas as pd

    out = session_dir / f"agreement-{coder}.csv"
    pd.DataFrame([
        {
            "session_id": session_id, "coder": coder,
            "behavior": r.behavior, "person": r.person,
            "n_human": r.n_human, "n_machine": r.n_machine,
            "n_matched": r.n_matched, "f1": r.f1, "recall": r.recall,
            "precision": r.precision, "onset_mae_s": r.boundary_mae_s,
            "kappa": r.kappa, "tolerance_s": args.tolerance,
        }
        for r in results
    ]).to_csv(out, index=False)
    print(f"-> {out}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from conversation_analyst.benchmark import run_benchmark

    report = run_benchmark(
        output_dir=Path(args.output), seeds=tuple(args.seeds),
        quick=args.quick, model_dir=args.model_dir,
    )
    print(report.render_markdown())
    print(f"written to {Path(args.output) / 'BENCHMARK.md'}")
    failed = [r for r in report.accuracy_rows if not r["passed"]]
    failed += [r for r in report.measure_rows if not r["passed"]]
    return 1 if failed else 0


# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conversation-analyst",
        description="Multimodal measurement of dyadic conversation quality.",
    )
    parser.add_argument("--version", action="version", version=f"Conversation Analyst {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze",
                             help="analyze a directory of recordings or a manifest")
    analyze.add_argument("target", help="directory of videos, or a manifest .json")
    analyze.add_argument("-o", "--output", default="workspace")
    analyze.add_argument("-c", "--config", default=None, help="YAML config overrides")
    analyze.add_argument("--model-dir", default=None)
    analyze.add_argument("--lenient", action="store_true",
                         help="allow sessions missing a close-up view")
    analyze.add_argument("--skip", nargs="*", default=[],
                         choices=["face", "body", "asr", "prosody", "semantics", "laughter"],
                         help="stages to skip")
    analyze.set_defaults(func=cmd_analyze)

    models_cmd = sub.add_parser("models", help="download or check model assets")
    models_cmd.add_argument("action", choices=["fetch", "status"], nargs="?",
                            default="status")
    models_cmd.add_argument("--model-dir", default="models")
    models_cmd.set_defaults(func=cmd_models)

    codebook = sub.add_parser("codebook", help="write the measure catalogue")
    codebook.add_argument("-o", "--output", default="codebook.csv")
    codebook.set_defaults(func=cmd_codebook)

    demo = sub.add_parser("demo", help="build a synthetic session and analyze it")
    demo.add_argument("-o", "--output", default="workspace")
    demo.add_argument("--turns", type=int, default=20)
    demo.add_argument("--seed", type=int, default=3)
    demo.add_argument("--faces", nargs="*", default=None,
                      help="optional videos of faces to loop into the close-up views")
    demo.add_argument("--wide", action="store_true",
                      help="also write an optional third wide view")
    demo.add_argument("-c", "--config", default=None)
    demo.add_argument("--model-dir", default=None)
    demo.add_argument("--skip", nargs="*", default=[])
    demo.set_defaults(func=cmd_demo)

    gui = sub.add_parser("gui", help="open the desktop application")
    gui.set_defaults(func=cmd_gui)

    validate = sub.add_parser("validate", help="score detectors against known ground truth")
    validate.add_argument("-o", "--output", default="validation")
    validate.add_argument("--seeds", type=int, nargs="*", default=[3, 7, 11, 17])
    validate.add_argument("--quick", action="store_true",
                          help="fewer seeds and no transcription")
    validate.set_defaults(func=cmd_validate)

    study = sub.add_parser(
        "validate-study",
        help="compare pipeline output against the study's own records "
             "(Praat values, skill groups, partner reports)",
    )
    study.add_argument("study", help="path to the study zip or extracted folder")
    study.add_argument("-w", "--workspace", default="workspace",
                       help="analysis workspace holding measures_all.csv")
    study.add_argument("--openface", default=None,
                       help="zip or folder of the lab's OpenFace CSVs for "
                            "cross-toolchain convergence (auto-detected in "
                            "Downloads when omitted)")
    study.set_defaults(func=cmd_validate_study)

    docs = sub.add_parser(
        "docs", help="write the documentation-of-record page (every "
                     "decision, its reasoning, its citations)",
    )
    docs.add_argument("-o", "--output", default="documentation.html")
    docs.add_argument("--open", action="store_true", help="open in the browser")
    docs.set_defaults(func=cmd_docs)

    agree = sub.add_parser(
        "agreement",
        help="score a human coder's exported file against the detectors",
    )
    agree.add_argument("coding_file", help="JSON exported by the coding page")
    agree.add_argument("-w", "--workspace", default="workspace")
    agree.add_argument("--tolerance", type=float, default=0.5,
                       help="onset matching tolerance in seconds")
    agree.set_defaults(func=cmd_agreement)

    bench = sub.add_parser(
        "benchmark",
        help="accuracy (incl. ASR word error rate) and runtime, cold and warm",
    )
    bench.add_argument("-o", "--output", default="workspace/benchmark")
    bench.add_argument("--seeds", type=int, nargs="*", default=[3, 7, 11, 17])
    bench.add_argument("--quick", action="store_true",
                       help="fewer seeds; skips the slowest parts")
    bench.add_argument("--model-dir", default="models")
    bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
