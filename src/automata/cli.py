"""Command line: `automata plan` decides the edit, `automata render` makes it.

Planning and rendering are separate commands on purpose. Deciding is seconds
and free; rendering is minutes and hot. Splitting them means you can read the
timeline, disagree with it, fix it by hand, and render exactly that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .perception.fixture import load_events, save_events
from .pipeline import editable_window, plan
from .project import Project
from .render import ffmpeg
from .timeline import Timeline


def _analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze  # imported late: pulls in the heavy optional deps

    project = Project.load(args.project)
    output = Path(args.output) if args.output else _events_path(project, args.project)

    events, report = analyze(
        project,
        use_llm=not args.no_llm,
        cache_path=output.with_suffix(".verdicts.json"),
    )
    save_events(events, output)

    print(output)
    for line in report.lines():
        print(f"  {line}")
    return 0


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
        print("sources do not overlap in time; check offset_s", file=sys.stderr)
        return 2

    timeline = plan(
        load_events(events),
        config=project.director,
        start_t=start_t,
        end_t=end_t,
    ).apply_cuts()

    output = Path(args.output or Path(args.project).with_suffix(".timeline.json"))
    timeline.save(output)
    _summarise(timeline, output)
    return 0


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

    a = sub.add_parser("analyze", help="transcribe, track gaze, and write an events file")
    a.add_argument("project")
    a.add_argument("-o", "--output")
    a.add_argument(
        "--no-llm",
        action="store_true",
        help="keyword intent tagging only; no API calls",
    )
    a.set_defaults(func=_analyze)

    p = sub.add_parser("plan", help="decide the edit and write a timeline")
    p.add_argument("project")
    p.add_argument("-o", "--output")
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
