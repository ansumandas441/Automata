"""The edit decision list: what the orchestrator actually produces.

The director emits a :class:`Timeline` -- layout segments plus cut spans, in
master time. Nothing here touches pixels. That separation is the point: the
same timeline drives a post-hoc ffmpeg render today and live scene switching
later, it can be diffed and unit-tested without hardware, and a creator can
hand-fix a bad decision instead of re-recording.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

EPSILON = 1e-3
"""Anything shorter than a millisecond is rounding noise, not a shot."""


class Layout(Enum):
    SCREEN_FOCUS = "screen_focus"
    """Screen fills the frame, camera in a corner."""

    CAMERA_FOCUS = "camera_focus"
    """Camera fills the frame, screen in a corner."""

    SCREEN_ONLY = "screen_only"
    """Degraded: no camera footage available."""

    CAMERA_ONLY = "camera_only"
    """Degraded: no screen footage available."""


@dataclass(frozen=True)
class Segment:
    t_start: float
    t_end: float
    layout: Layout
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class Cut:
    """A span removed from the output."""

    t_start: float
    t_end: float
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class Timeline:
    segments: tuple[Segment, ...]
    cuts: tuple[Cut, ...]
    duration: float
    start: float = 0.0
    """Master time of the first editable frame.

    Non-zero whenever the sources do not all start together: editing before
    every input has footage means compositing against black. Ignoring it is
    how the renderer ends up trimming each input to a *different* length and
    silently desyncing the concat.
    """

    kept_spans: tuple[tuple[float, float], ...] = ()
    """Master-time spans surviving the cuts, in output order.

    Empty before :meth:`apply_cuts`. The renderer trims its inputs to these.
    """

    warnings: tuple[str, ...] = ()

    def apply_cuts(self) -> Timeline:
        """Remove cut spans and re-time everything onto the output clock.

        This is where the fiddly cases land: cuts that overlap each other,
        cuts that straddle a layout switch (the segment must be split, not
        dropped), and pieces that collapse to zero or negative length.
        """
        merged = _merge_cuts(self.cuts, self.start, self.duration)
        kept = _complement(merged, self.start, self.duration)

        pieces: list[Segment] = []
        cursor = 0.0
        for span_start, span_end in kept:
            for segment in self.segments:
                start = max(segment.t_start, span_start)
                end = min(segment.t_end, span_end)
                if end - start <= EPSILON:
                    continue
                pieces.append(
                    Segment(
                        t_start=cursor + (start - span_start),
                        t_end=cursor + (end - span_start),
                        layout=segment.layout,
                        reason=segment.reason,
                    )
                )
            cursor += span_end - span_start

        return Timeline(
            segments=tuple(_coalesce(pieces)),
            cuts=(),
            duration=cursor,
            start=0.0,  # cuts applied: the output clock always begins at zero
            kept_spans=tuple(kept),
            warnings=self.warnings,
        )

    def spans_for(self, *layouts: Layout) -> list[tuple[float, float]]:
        """Merged time spans occupied by the given layouts."""
        selected = [s for s in self.segments if s.layout in layouts]
        return _merge_intervals((s.t_start, s.t_end) for s in selected)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "duration": round(self.duration, 3),
            "segments": [
                {
                    "start": round(s.t_start, 3),
                    "end": round(s.t_end, 3),
                    "layout": s.layout.value,
                    "reason": s.reason,
                }
                for s in self.segments
            ],
            "cuts": [
                {"start": round(c.t_start, 3), "end": round(c.t_end, 3), "reason": c.reason}
                for c in self.cuts
            ],
            "kept_spans": [[round(a, 3), round(b, 3)] for a, b in self.kept_spans],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Timeline:
        return cls(
            segments=tuple(
                Segment(s["start"], s["end"], Layout(s["layout"]), s.get("reason", ""))
                for s in data["segments"]
            ),
            cuts=tuple(
                Cut(c["start"], c["end"], c.get("reason", "")) for c in data.get("cuts", ())
            ),
            duration=data["duration"],
            start=float(data.get("start", 0.0)),
            kept_spans=tuple((a, b) for a, b in data.get("kept_spans", ())),
            warnings=tuple(data.get("warnings", ())),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Timeline:
        return cls.from_dict(json.loads(Path(path).read_text()))


# -- interval helpers ----------------------------------------------------


def _merge_intervals(spans) -> list[tuple[float, float]]:
    ordered = sorted((a, b) for a, b in spans if b - a > EPSILON)
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + EPSILON:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _merge_cuts(cuts, origin: float, duration: float) -> list[tuple[float, float]]:
    """Clamp cuts into the editable window and merge overlaps.

    Repeated "cut, cut, cut" produces overlapping spans by design; without this
    they would double-subtract and skew every later timestamp.
    """
    clamped = []
    for cut in cuts:
        start = min(max(cut.t_start, origin), duration)
        end = min(max(cut.t_end, origin), duration)
        if end - start > EPSILON:
            clamped.append((start, end))
    return _merge_intervals(clamped)


def _complement(spans, origin: float, duration: float) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    cursor = origin
    for start, end in spans:
        if start - cursor > EPSILON:
            kept.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor > EPSILON:
        kept.append((cursor, duration))
    return kept


def _coalesce(segments: list[Segment]) -> list[Segment]:
    """Join neighbouring segments that ended up with the same layout.

    Cutting a span out can butt two identical shots against each other; leaving
    them split would make the renderer emit a pointless switch.
    """
    out: list[Segment] = []
    for segment in segments:
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.layout is segment.layout
            and abs(prev.t_end - segment.t_start) <= EPSILON
        ):
            out[-1] = Segment(prev.t_start, segment.t_end, prev.layout, prev.reason)
        else:
            out.append(segment)
    return out
