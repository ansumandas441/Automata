"""Tuning knobs for the director and the renderer.

Defaults are chosen for a talking-head coding video. They are the difference
between a watchable edit and an unwatchable one, so they live here rather than
scattered as literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .timeline import Layout


class GazeMode(Enum):
    """How much authority the gaze signal carries.

    The difference is not sensitivity but *direction*: which edges of the signal
    are allowed to move the camera, and which are ignored.
    """

    FOLLOW = "follow"
    """Level-triggered, both ways. Looking at the lens holds camera; looking away
    holds screen. Suits a camera mounted off to one side of the screen, where
    looking away genuinely means "reading the screen"."""

    LATCH = "latch"
    """Edge-triggered, one way. Turning *to* the lens claims the camera and keeps
    it; turning away means nothing. Suits a camera the creator only faces
    deliberately -- the shot is then released by speech, not by a head turn."""

    OFF = "off"
    """Ignore gaze entirely. Suits a rig where the creator always faces the lens
    (camera sitting just above the screen), so gaze carries no information and
    only what they say does."""


def _coerce(enum_cls, value, field: str):
    """Turn a wire string into an enum member, or say what was allowed."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(repr(m.value) for m in enum_cls)
        raise ValueError(f"{field} must be one of {allowed}; got {value!r}") from None


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

    intent_hold_s: float | None = 6.0
    """How long an explicit spoken instruction pins the layout.

    "Let me show you this code" should hold on the screen even if the creator
    keeps glancing at the lens -- but not forever, or gaze never regains
    control for the rest of the video.

    ``None`` means hold until something else claims the shot, which is what you
    want when speech is the only signal you trust.
    """

    gaze_mode: GazeMode = GazeMode.FOLLOW
    """Which gaze transitions may move the camera. See :class:`GazeMode`."""

    return_to_default_after_s: float | None = None
    """Drift back to ``default_layout`` after this long with nothing asserting.

    Without it, a shot claimed by speech or by a gaze latch is held until
    something else claims it -- correct when every switch should be deliberate.
    With it, attention decays: an explicit "look at this" wins the screen, and
    once the creator goes back to ordinary narration the shot returns home.
    ``0.0`` means return as soon as the claim lapses.
    """

    gaze_enabled: bool = True
    """Master switch, kept for existing project files. Prefer
    ``gaze_mode="off"``; setting this to ``False`` forces that regardless."""

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

    @property
    def effective_gaze_mode(self) -> GazeMode:
        return GazeMode.OFF if not self.gaze_enabled else self.gaze_mode

    def __post_init__(self) -> None:
        # Accept the wire spelling as well as the enum, so a project file and a
        # Python caller can both say gaze_mode="latch".
        object.__setattr__(self, "gaze_mode", _coerce(GazeMode, self.gaze_mode, "gaze_mode"))
        object.__setattr__(
            self, "default_layout", _coerce(Layout, self.default_layout, "default_layout")
        )

        for name in ("min_shot_s", "gaze_debounce_s", "max_cut_back_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("intent_hold_s", "return_to_default_after_s"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")


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
