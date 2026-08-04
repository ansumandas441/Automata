"""Tuning knobs for the director and the renderer.

Defaults are chosen for a talking-head coding video. They are the difference
between a watchable edit and an unwatchable one, so they live here rather than
scattered as literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .timeline import Layout


@dataclass(frozen=True)
class DirectorConfig:
    default_layout: Layout = Layout.SCREEN_FOCUS

    min_shot_s: float = 2.5
    """Hysteresis. No shot may be shorter than this.

    Without it, a creator who glances between screen and lens produces a
    strobing edit. This single constant does more for watchability than any
    amount of classifier accuracy.
    """

    gaze_debounce_s: float = 0.8
    """Gaze must hold steady this long before it may move the camera.

    A glance is not a shot change.
    """

    intent_hold_s: float = 6.0
    """How long an explicit spoken instruction pins the layout.

    "Let me show you this code" should hold on the screen even if the creator
    keeps glancing at the lens -- but not forever, or gaze never regains
    control for the rest of the video.
    """

    gaze_enabled: bool = True

    # -- cut resolution --

    max_cut_back_s: float = 20.0
    """Absolute backstop on how far a single "cut that" reaches."""

    max_cut_back_utterances: int = 2
    """How many sentences before the trigger a cut may swallow.

    The real limit in practice. Pauses in fluent speech rarely clear
    `silence_gap_s`, so without a count the walk runs to `max_cut_back_s` and
    one word deletes twenty seconds. Two matches how the phrase is used: a bad
    take plus the "no, wait" that follows it.
    """

    silence_gap_s: float = 0.6
    """A pause at least this long is treated as a safe cut boundary."""

    cut_pad_s: float = 0.12
    """Expand a cut into surrounding silence, never into neighbouring speech."""

    def __post_init__(self) -> None:
        for name in ("min_shot_s", "gaze_debounce_s", "intent_hold_s", "max_cut_back_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PipConfig:
    scale: float = 0.26
    """Picture-in-picture width as a fraction of the canvas width."""

    margin: int = 36
    corner: str = "top_right"

    def __post_init__(self) -> None:
        if not 0 < self.scale < 1:
            raise ValueError("pip scale must be between 0 and 1")
        if self.corner not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
            raise ValueError(f"unknown pip corner {self.corner!r}")


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30

    camera_pip: PipConfig = field(default_factory=lambda: PipConfig(corner="top_right"))
    screen_pip: PipConfig = field(
        default_factory=lambda: PipConfig(scale=0.22, corner="bottom_right")
    )

    video_codec: str = "libx264"
    video_preset: str = "medium"
    crf: int = 18
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
