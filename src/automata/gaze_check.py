"""Sampling gaze from a clip without building a project around it.

The preflight check needs one number -- how often the detector actually finds a
face -- and nothing else. Going through :func:`automata.perception.gaze.track`
would mean inventing a SourceClip and a master timeline for a four-second
throwaway.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .perception.gaze import GazeConfig
from .sources import SourceClip, SourceRole


def sample_states(clip: Path, config: GazeConfig | None = None) -> Iterator[str]:
    """Yield the gaze state of each sampled frame."""
    from .perception.gaze import track

    source = SourceClip(id="check", role=SourceRole.CAMERA, path=str(clip))
    for sample in track(source, config):
        yield sample.state


__all__ = ["sample_states"]
