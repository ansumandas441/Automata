"""Timestamped perception events.

Every event carries ``t`` in seconds on the *master* timeline. Producers are
responsible for converting their own source-local timestamps first (see
:meth:`automata.sources.SourceClip.to_master`) -- nothing downstream knows
which file an event came from.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sources import SourceRole


class GazeState:
    """Namespaced constants for :class:`GazeSample`.

    ``UNKNOWN`` is deliberately distinct from ``AWAY``. "No face detected"
    means the creator stepped out of frame, the light dropped, or the detector
    failed -- it does *not* mean they looked away, and the director must not
    treat it as a reason to switch shots.
    """

    LOOKING = "looking"
    AWAY = "away"
    UNKNOWN = "unknown"

    ALL = frozenset({LOOKING, AWAY, UNKNOWN})


class IntentKind:
    """What the creator's speech asked the editor to do."""

    FOCUS_SCREEN = "focus_screen"
    FOCUS_CAMERA = "focus_camera"
    CUT_PREVIOUS = "cut_previous"
    NEUTRAL = "neutral"

    ALL = frozenset({FOCUS_SCREEN, FOCUS_CAMERA, CUT_PREVIOUS, NEUTRAL})


@dataclass(frozen=True)
class Event:
    """Base for everything on the bus. ``t`` is the *occurrence* time."""

    t: float
    producer: str

    @property
    def t_end(self) -> float:
        return self.t


@dataclass(frozen=True)
class Word:
    text: str
    t: float
    t_end: float


@dataclass(frozen=True)
class Utterance(Event):
    """One VAD-bounded run of speech.

    Word-level timings are what make cut boundaries precise; without them a cut
    can only snap to whole-utterance edges.
    """

    text: str
    end: float
    words: tuple[Word, ...] = ()

    @property
    def t_end(self) -> float:
        return self.end


@dataclass(frozen=True)
class GazeSample(Event):
    state: str
    confidence: float = 1.0


@dataclass(frozen=True)
class Intent(Event):
    """A classified utterance. Spans the utterance it was derived from."""

    kind: str
    end: float
    confidence: float = 1.0
    evidence: str = ""

    @property
    def t_end(self) -> float:
        return self.end


@dataclass(frozen=True)
class SourceStatus(Event):
    """A source appearing or dropping out mid-recording (USB dropout, etc.)."""

    source_id: str
    role: SourceRole
    available: bool


@dataclass(frozen=True)
class ScreenActivity(Event):
    """Normalised frame-diff energy on the screen capture, 0.0 - 1.0."""

    level: float
