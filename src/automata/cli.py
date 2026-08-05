"""Command line: `automata plan` decides the edit, `automata render` makes it.

Planning and rendering are separate commands on purpose. Deciding is seconds
and free; rendering is minutes and hot. Splitting them means you can read the
timeline, disagree with it, fix it by hand, and render exactly that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .events import Intent
from .init_project import ProbeError, init_project
from .perception.fixture import load_events, save_events
from .pipeline import editable_window, plan
from .project import Project
from .record import RecordError
from .render import ffmpeg
from .timeline import Timeline


def _record(args: argparse.Namespace) -> int:
    from . import recorder_ui as ui

    try:
        devices = ui.list_devices()
        if args.list:
            ui.print_devices(devices)
            return 0

        if args.check:
            _, camera, microphone = ui.choose(
                devices, args.screen, args.camera, args.audio
            )
            ok = ui.check_framing(camera, microphone, Path(args.directory))
            return 0 if ok else 1

        directory = Path(args.directory)
        if (directory / "screen.mp4").exists() and not args.force:
            print(
                f"{directory}/screen.mp4 already exists; pass --force to overwrite "
                "or choose another directory.",
                file=sys.stderr,
            )
            return 2

        screen, camera, microphone = ui.choose(
            devices, args.screen, args.camera, args.audio
        )
        recording = ui.run_session(directory, screen, camera, microphone)
    except RecordError as exc:
        print(exc, file=sys.stderr)
        return 2

    offset = recording.camera_offset_s
    print(f"  screen   {recording.screen_duration:7.2f}s")
    print(f"  camera   {recording.camera_duration:7.2f}s")
    print(f"  offset   {offset:+7.3f}s  (measured, not guessed)")
    print()

    project = directory / "project.json"
    result = init_project(
        recording.screen_path, recording.camera_path, project,
        talking_head=args.talking_head, camera_offset_s=offset,
    )
    print(result.path)
    for note in result.notes:
        print(f"  note: {note}")
    for warning in result.warnings:
        print(f"  warning: {warning}", file=sys.stderr)

    if args.no_process:
        print(f"\nNext:  automata analyze {project}")
        return 0

    print("\n== analysing ==")
    args.project, args.output, args.force, args.intent, args.no_llm = (
        str(project), None, True, args.intent, False,
    )
    if (code := _analyze(args)) != 0:
        return code

    print("\n== deciding the edit ==")
    args.retag = False
    if (code := _plan(args)) != 0:
        return code

    print("\n== rendering ==")
    args.timeline, args.dry_run, args.ffmpeg = None, False, "ffmpeg"
    args.output = str(directory / "out.mp4")
    return _render(args)


def _init(args: argparse.Namespace) -> int:
    output = Path(args.output or "project.json")
    if output.exists() and not args.force:
        print(f"{output} already exists; pass --force to overwrite.", file=sys.stderr)
        return 2
    try:
        result = init_project(
            args.screen, args.camera, output, talking_head=args.talking_head
        )
    except ProbeError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(result.path)
    print(f"  screen  {result.screen.describe()}")
    print(f"  camera  {result.camera.describe()}")
    for note in result.notes:
        print(f"  note: {note}")
    for warning in result.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    print(f"\nNext:  automata analyze {output}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    project = Project.load(args.project)
    output = Path(args.output) if args.output else _events_path(project, args.project)

    # Checked before the expensive work, not after: transcription can run for
    # minutes, and refusing to write at the end would waste all of it.
    if output.exists() and not args.force:
        print(
            f"{output} already exists; pass --force to overwrite.\n"
            "Events files are meant to be hand-edited -- a corrected transcript "
            "or a written fixture lives here, and re-analysing would discard it.",
            file=sys.stderr,
        )
        return 2

    from .analyze import analyze  # imported late: pulls in the optional deps

    mode = "offline" if args.no_llm else args.intent
    try:
        events, report = analyze(
            project, mode=mode, cache_path=output.with_suffix(".verdicts.json")
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    save_events(events, output)

    print(output)
    for line in report.lines():
        print(f"  {line}")
    return 0


def _load_for_plan(path: Path, *, retag: bool):
    """Load events, optionally discarding the stored intent labels.

    Those labels are derived data, but they are baked into the events file at
    analysis time. That is right for language-model verdicts, which are slow and
    must stay reproducible -- and wrong for the local keyword patterns, which are
    free and deterministic. Without this, editing a regex has no effect until you
    re-transcribe the whole recording.
    """
    events = load_events(path)
    if not retag:
        return events
    return [e for e in events if not isinstance(e, Intent)]


def _window_error(project: Project) -> str:
    """Say which of the two causes it actually is.

    An empty editable window means either the clips genuinely don't overlap, or
    -- far more often on a hand-written project -- nobody declared how long they
    are. Blaming `offset_s` for the second sends people to the wrong field.
    """
    if all(s.duration_s is None for s in project.sources):
        return (
            "no source declares duration_s, so there is no editable window.\n"
            "Add it to each source (seconds), e.g.:\n"
            "  ffprobe -v error -show_entries format=duration "
            "-of csv=p=0 screen.mp4"
        )
    return "sources do not overlap in time; check offset_s and duration_s"


def _events_path(project: Project, project_arg: str) -> Path:
    """Where a project's events live.

    The project file may name one explicitly; otherwise `analyze` writes and
    `plan` reads the same conventional path beside it, so the two commands line
    up without the creator having to wire them together.
    """
    if project.events_path is not None:
        return Path(project.events_path)
    return Path(project_arg).with_suffix(".events.json")


def _plan(args: argparse.Namespace) -> int:
    project = Project.load(args.project)
    events = _events_path(project, args.project)
    if not events.exists():
        print(f"no events at {events}; run `automata analyze` first", file=sys.stderr)
        return 2

    start_t, end_t = editable_window(project.sources)
    if end_t - start_t <= 0:
        print(_window_error(project), file=sys.stderr)
        return 2

    timeline = plan(
        _load_for_plan(events, retag=args.retag),
        config=project.director,
        start_t=start_t,
        end_t=end_t,
    ).apply_cuts()

    output = Path(args.output or Path(args.project).with_suffix(".timeline.json"))
    timeline.save(output)
    _summarise(timeline, output)
    for note in _excluded_footage(project, start_t, end_t):
        print(f"  note: {note}")
    return 0


def _excluded_footage(project: Project, start_t: float, end_t: float) -> list[str]:
    """Report footage outside the editable window.

    The window is the overlap of every source, so a clip that ran longer than
    the others is trimmed away -- correctly, since there is nothing to composite
    it against. But dropping it silently means a camera that stopped two minutes
    early costs two minutes of video with nothing on screen to say so.
    """
    notes = []
    for clip in project.sources:
        end = clip.master_end
        lead = start_t - clip.master_start
        tail = (end - end_t) if end is not None else 0.0
        if lead > 0.05:
            notes.append(f"{clip.id}: first {lead:.2f}s unused (another source starts later)")
        if tail > 0.05:
            notes.append(f"{clip.id}: last {tail:.2f}s unused (another source ends sooner)")
    return notes


def _render(args: argparse.Namespace) -> int:
    project = Project.load(args.project)
    if args.timeline:
        timeline = Timeline.load(args.timeline)
    elif not (events := _events_path(project, args.project)).exists():
        print(f"no events at {events} and no --timeline given", file=sys.stderr)
        return 2
    else:
        start_t, end_t = editable_window(project.sources)
        timeline = plan(
            load_events(events),
            config=project.director,
            start_t=start_t,
            end_t=end_t,
        ).apply_cuts()

    if not timeline.kept_spans:
        print("timeline has no kept spans; run `plan` first", file=sys.stderr)
        return 2

    render = ffmpeg.build(
        timeline, list(project.sources), args.output, project.render, ffmpeg=args.ffmpeg
    )
    for warning in render.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if args.dry_run:
        print(render.write_graph())
        print(" ".join(render.argv))
        return 0

    render.run()
    print(f"wrote {args.output}")
    return 0


def _summarise(timeline: Timeline, output: Path) -> None:
    removed = sum(c.duration for c in timeline.cuts)
    print(f"{output}")
    print(f"  {len(timeline.segments)} shots over {timeline.duration:.1f}s")
    if removed:
        print(f"  {removed:.1f}s removed by cuts")
    for warning in timeline.warnings:
        print(f"  warning: {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="automata", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("record", help="capture screen and camera together, then edit")
    r.add_argument("directory", nargs="?", default=".", help="where to write everything")
    r.add_argument("--list", action="store_true", help="show capture devices and exit")
    r.add_argument("--check", action="store_true",
                   help="record 4s and report whether your framing works for gaze")
    r.add_argument("--screen", type=int, help="screen device index")
    r.add_argument("--camera", type=int, help="camera device index")
    r.add_argument("--audio", type=int, help="microphone device index")
    r.add_argument("--talking-head", action="store_true",
                   help="camera sits on the screen: speech decides, gaze is ignored")
    r.add_argument("--intent", choices=("auto", "offline", "claude", "gemini"), default=None)
    r.add_argument("--no-process", action="store_true",
                   help="just record; do not analyse or render")
    r.add_argument("-f", "--force", action="store_true", help="overwrite an existing recording")
    r.set_defaults(func=_record)

    i = sub.add_parser("init", help="measure two recordings and write a project file")
    i.add_argument("screen", help="the screen capture")
    i.add_argument("camera", help="the webcam recording")
    i.add_argument("-o", "--output", help="where to write it (default project.json)")
    i.add_argument(
        "--talking-head",
        action="store_true",
        help="camera sits on the screen, so you always face it: speech decides "
             "everything and gaze is ignored",
    )
    i.add_argument("-f", "--force", action="store_true", help="overwrite an existing project")
    i.set_defaults(func=_init)

    a = sub.add_parser("analyze", help="transcribe, track gaze, and write an events file")
    a.add_argument("project")
    a.add_argument("-o", "--output")
    a.add_argument(
        "--intent",
        choices=("auto", "offline", "claude", "gemini"),
        default=None,
        help="how to read the transcript: offline keyword patterns, per-sentence "
             "claude, or gemini narrative segmentation. Defaults to $AUTOMATA_INTENT, "
             "else whichever is configured",
    )
    a.add_argument(
        "--no-llm",
        action="store_true",
        help="shorthand for --intent offline; no API calls",
    )
    a.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite an existing events file",
    )
    a.set_defaults(func=_analyze)

    p = sub.add_parser("plan", help="decide the edit and write a timeline")
    p.add_argument("project")
    p.add_argument("-o", "--output")
    p.add_argument(
        "--retag",
        action="store_true",
        help="re-run the local keyword patterns instead of using the stored "
             "labels (discards any language-model labels in the events file)",
    )
    p.set_defaults(func=_plan)

    r = sub.add_parser("render", help="render a timeline with ffmpeg")
    r.add_argument("project")
    r.add_argument("-t", "--timeline", help="use this timeline instead of re-planning")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--ffmpeg", default="ffmpeg")
    r.add_argument("--dry-run", action="store_true", help="print the command, render nothing")
    r.set_defaults(func=_render)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
