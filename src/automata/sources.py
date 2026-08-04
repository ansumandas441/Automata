"""Recorded sources and their relationship to the master timeline.

The master timeline is in seconds, with ``t=0`` at the notional start of the
recording session. Every source is an independently recorded file that started
at some other moment and may run at a slightly different rate, so each carries
an ``offset_s`` mapping its own local time onto the master timeline::

    master_t = local_t + offset_s

Wall-clock timestamps are only ever used to *derive* that offset once, at
ingest. They are never used for ongoing timing: NTP can step the system clock
mid-recording, and a 45-minute take at a nominal 30fps that is really 29.97fps
drifts by seconds. Downstream code must key off container PTS, never frame
index -- dropped screen-capture frames silently desync anything that counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SourceRole(Enum):
    """What a source contributes to the composition."""

    SCREEN = "screen"
    CAMERA = "camera"


@dataclass(frozen=True)
class SourceClip:
    id: str
    role: SourceRole
    path: str
    offset_s: float = 0.0
    duration_s: float | None = None
    has_audio: bool = False

    def to_master(self, local_t: float) -> float:
        return local_t + self.offset_s

    def to_local(self, master_t: float) -> float:
        return master_t - self.offset_s

    @property
    def master_start(self) -> float:
        return self.offset_s

    @property
    def master_end(self) -> float | None:
        if self.duration_s is None:
            return None
        return self.offset_s + self.duration_s

    def covers(self, master_t: float) -> bool:
        """Whether this clip has footage at the given master time."""
        if master_t < self.master_start:
            return False
        end = self.master_end
        return end is None or master_t <= end


def overlap_window(clips: list[SourceClip]) -> tuple[float, float]:
    """The master-time window in which *every* clip has footage.

    Editing outside this window means at least one input is missing, so the
    director would be compositing against a black frame. Callers should treat
    this as the editable span, not the union of the clips.
    """
    if not clips:
        return (0.0, 0.0)
    start = max(c.master_start for c in clips)
    ends = [c.master_end for c in clips if c.master_end is not None]
    end = min(ends) if ends else start
    return (start, max(start, end))
