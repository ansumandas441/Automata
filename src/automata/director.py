"""The director: a deterministic state machine that decides the edit.

The language model is a *sensor*, not the director. It labels one utterance
into an :class:`~automata.events.IntentKind` and stops there; every actual
decision -- which shot, when to switch, what to remove -- is made by the fixed
arbitration below. That keeps the edit reproducible, roughly fifty times
cheaper, and debuggable: when a cut is wrong you can name the rule that made
it, instead of re-rolling a prompt.

Arbitration, highest priority first::

    forced (a source is missing)  >  explicit speech  >  gaze  >  drift  >  hold

with two dampers on top: no shot may be shorter than ``min_shot_s``, and gaze
must hold steady for ``gaze_debounce_s`` before it counts.

What *releases* a shot matters as much as what claims it, and that depends on
the rig -- see :class:`~automata.config.GazeMode`. With the camera off to one
side, looking away is itself a signal (``FOLLOW``). With a camera the creator
only faces deliberately, turning away means nothing and speech has to release
the shot (``LATCH``). With the camera sitting on the screen, gaze carries no
information at all (``OFF``) and only speech decides.

Events must arrive in strict time order -- that is what :class:`.EventBus`
guarantees. The director asserts it rather than trusting it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

from .config import DirectorConfig, GazeMode
from .events import (
    Event,
    GazeSample,
    GazeState,
    Intent,
    IntentKind,
    ScreenActivity,
    SourceStatus,
    Utterance,
)
from .sources import SourceRole
from .timeline import EPSILON, Cut, Layout, Segment, Timeline


class Priority(IntEnum):
    HOLD = 0
    RECOVERY = 1
    GAZE = 2
    EXPLICIT = 3
    FORCED = 4
    """Bypasses hysteresis -- there is no footage to hold on."""


@dataclass(frozen=True)
class _Desire:
    layout: Layout
    reason: str
    priority: Priority
    consumes_edge: bool = False
    """True for a one-shot desire raised by a gaze edge, cleared once acted on."""


_INTENT_LAYOUTS = {
    IntentKind.FOCUS_SCREEN: Layout.SCREEN_FOCUS,
    IntentKind.FOCUS_CAMERA: Layout.CAMERA_FOCUS,
}

_DEGRADED = (Layout.SCREEN_ONLY, Layout.CAMERA_ONLY)


class Director:
    def __init__(
        self,
        config: DirectorConfig | None = None,
        *,
        start_t: float = 0.0,
        horizon: float | None = None,
    ) -> None:
        self.config = config or DirectorConfig()
        self._start_t = start_t
        self._horizon = horizon

        self._layout = self.config.default_layout
        self._reason = "default"
        self._segment_start = start_t
        self._last_switch_t = start_t

        self._gaze = GazeState.UNKNOWN
        self._gaze_since = start_t
        self._gaze_rising = False
        """A turn *to* the lens that has not yet been acted on (LATCH mode)."""

        self._intent: Intent | None = None
        self._intent_expiry = -float("inf")
        self._idle_since: float | None = None
        """When every rule last stopped asserting, for return-to-default."""

        self._utterances: deque[Utterance] = deque()
        self._missing: set[SourceRole] = set()

        self._segments: list[Segment] = []
        self._cuts: list[Cut] = []
        self._warnings: list[str] = []
        self._now = start_t
        self._finished = False

    # -- input -----------------------------------------------------------

    def handle(self, event: Event) -> None:
        if self._finished:
            raise RuntimeError("director already finished")
        if event.t < self._now - EPSILON:
            raise ValueError(
                f"out-of-order event: {event.t:.3f} < {self._now:.3f}; "
                "events must be drained from the bus, not consumed on arrival"
            )
        self._now = max(self._now, event.t)

        if isinstance(event, Utterance):
            self._on_utterance(event)
        elif isinstance(event, Intent):
            self._on_intent(event)
        elif isinstance(event, GazeSample):
            self._on_gaze(event)
        elif isinstance(event, SourceStatus):
            self._on_source(event)
        elif isinstance(event, ScreenActivity):
            pass  # reserved: a tiebreaker once gaze and speech both abstain

        self._reconcile(self._now)

    def advance(self, t: float) -> None:
        """Move time forward with no event.

        Intent expiry and gaze debounce are time-based, so a long silence still
        needs the state machine re-evaluated. Live mode calls this on a timer;
        post-hoc gets it for free from the gaze stream's own cadence.
        """
        if t < self._now - EPSILON:
            raise ValueError(f"time went backwards: {t:.3f} < {self._now:.3f}")
        self._now = max(self._now, t)
        self._reconcile(self._now)

    def finish(self, end_t: float) -> Timeline:
        """Close the open shot and return the timeline in master time."""
        if self._finished:
            raise RuntimeError("director already finished")
        self._finished = True
        end_t = max(end_t, self._segment_start)

        if end_t - self._segment_start > EPSILON:
            self._segments.append(
                Segment(self._segment_start, end_t, self._layout, self._reason)
            )
        elif not self._segments:
            # Recording shorter than a single shot: still emit something valid.
            self._segments.append(
                Segment(self._start_t, max(end_t, self._start_t), self._layout, self._reason)
            )
            self._warnings.append("recording shorter than one shot; emitted a single segment")

        return Timeline(
            segments=tuple(self._segments),
            cuts=tuple(self._cuts),
            duration=max(end_t, self._start_t),
            start=self._start_t,
            warnings=tuple(self._warnings),
        )

    # -- event handlers ---------------------------------------------------

    def _on_utterance(self, utterance: Utterance) -> None:
        self._utterances.append(utterance)
        floor = utterance.t_end - self.config.max_cut_back_s
        while self._utterances and self._utterances[0].t_end < floor:
            self._utterances.popleft()

    def _on_intent(self, intent: Intent) -> None:
        if intent.kind == IntentKind.CUT_PREVIOUS:
            cut = self._resolve_cut(intent)
            if cut is None:
                self._warnings.append(
                    f"cut request at {intent.t:.2f}s resolved to an empty span; ignored"
                )
            else:
                self._cuts.append(cut)
            return

        if intent.kind in _INTENT_LAYOUTS:
            self._intent = intent
            hold = self.config.intent_hold_s
            self._intent_expiry = math.inf if hold is None else intent.t_end + hold
            # Speech supersedes a latched glance. Without this, a latch that was
            # overridden would spring back the moment the instruction expired,
            # and "let me show you this" would bounce to camera mid-sentence.
            self._gaze_rising = False

    def _on_gaze(self, sample: GazeSample) -> None:
        if sample.state not in GazeState.ALL:
            raise ValueError(f"unknown gaze state {sample.state!r}")
        if sample.state != self._gaze:
            turned_to_lens = (
                sample.state == GazeState.LOOKING and self._gaze != GazeState.LOOKING
            )
            self._gaze = sample.state
            self._gaze_since = sample.t
            if turned_to_lens:
                self._gaze_rising = True

    def _on_source(self, status: SourceStatus) -> None:
        if status.available:
            self._missing.discard(status.role)
        elif status.role not in self._missing:
            self._missing.add(status.role)
            self._warnings.append(
                f"{status.role.value} source {status.source_id!r} lost at {status.t:.2f}s"
            )

    # -- arbitration -------------------------------------------------------

    def _desire(self, now: float) -> _Desire | None:
        if SourceRole.SCREEN in self._missing and SourceRole.CAMERA in self._missing:
            return None  # nothing to cut to; hold and let the warning speak
        if SourceRole.SCREEN in self._missing:
            return _Desire(Layout.CAMERA_ONLY, "screen source lost", Priority.FORCED)
        if SourceRole.CAMERA in self._missing:
            return _Desire(Layout.SCREEN_ONLY, "camera source lost", Priority.FORCED)

        if self._intent is not None and now < self._intent_expiry:
            layout = _INTENT_LAYOUTS[self._intent.kind]
            evidence = self._intent.evidence or self._intent.kind
            return _Desire(layout, f"said: {evidence}", Priority.EXPLICIT)

        gaze = self._gaze_desire(now)
        if gaze is not None:
            return gaze

        if self._layout in _DEGRADED:
            # A source came back but nothing is asking for a shot; don't strand
            # the edit in a degraded layout for the rest of the video.
            return _Desire(self.config.default_layout, "sources restored", Priority.RECOVERY)

        # Gaze UNKNOWN (no face in frame) is not a reason to move the camera.
        return None

    def _gaze_desire(self, now: float) -> _Desire | None:
        """What gaze is asking for, if anything.

        The two modes differ in which transitions count. FOLLOW reads the signal
        continuously in both directions. LATCH reads only the rising edge -- a
        turn *to* the lens claims the camera, and turning away says nothing at
        all, leaving the shot to be released by speech instead.
        """
        mode = self.config.effective_gaze_mode
        if mode is GazeMode.OFF:
            return None

        settled = now - self._gaze_since >= self.config.gaze_debounce_s

        if mode is GazeMode.LATCH:
            if self._gaze_rising and self._gaze == GazeState.LOOKING and settled:
                return _Desire(
                    Layout.CAMERA_FOCUS, "turned to lens", Priority.GAZE, consumes_edge=True
                )
            return None

        if self._gaze in (GazeState.LOOKING, GazeState.AWAY) and settled:
            looking = self._gaze == GazeState.LOOKING
            return _Desire(
                Layout.CAMERA_FOCUS if looking else Layout.SCREEN_FOCUS,
                "gaze on lens" if looking else "gaze off lens",
                Priority.GAZE,
            )
        return None

    def _drift_home(self, now: float) -> _Desire | None:
        """Return to the default layout once every claim has lapsed.

        This is what makes ordinary narration a signal in its own right: an
        explicit "look at this" wins the screen, and when the creator goes back
        to talking rather than showing, nothing renews that claim and the shot
        comes home on its own.
        """
        grace = self.config.return_to_default_after_s
        if grace is None or self._layout is self.config.default_layout:
            return None
        if self._idle_since is None or now - self._idle_since < grace:
            return None
        return _Desire(self.config.default_layout, "attention lapsed", Priority.RECOVERY)

    def _reconcile(self, now: float) -> None:
        desire = self._desire(now)
        if desire is None:
            if self._idle_since is None:
                self._idle_since = now
            desire = self._drift_home(now)
        else:
            self._idle_since = None

        if desire is None or desire.layout is self._layout:
            return
        if (
            desire.priority < Priority.FORCED
            and now - self._last_switch_t < self.config.min_shot_s
        ):
            return
        if desire.consumes_edge:
            self._gaze_rising = False
        self._switch(now, desire.layout, desire.reason)

    def _switch(self, now: float, layout: Layout, reason: str) -> None:
        if now - self._segment_start <= EPSILON:
            # Two switches inside the same instant: rewrite the pending shot
            # rather than emitting a zero-length segment.
            self._layout, self._reason = layout, reason
            self._last_switch_t = now
            return
        self._segments.append(Segment(self._segment_start, now, self._layout, self._reason))
        self._segment_start = now
        self._last_switch_t = now
        self._layout, self._reason = layout, reason

    # -- cut resolution ----------------------------------------------------

    def _resolve_cut(self, trigger: Intent) -> Cut | None:
        """Expand "cut that" backwards to the start of the bad take.

        Walks back through recent speech until it finds a pause long enough to
        be a natural boundary, bounded by ``max_cut_back_s``. The trigger
        utterance itself is always removed -- nobody wants "cut that" left in.

        In fluent speech no pause clears the threshold -- inter-sentence gaps
        sit around half a second and vary by less than a tenth. So the reach is
        also capped at ``max_cut_back_utterances``: "cut that" means the last
        thing said, not everything since the last long breath. Without the cap,
        a recording with no real pauses loses ``max_cut_back_s`` of good footage
        to one word, and picking the widest of several near-identical gaps is
        just deciding on noise.
        """
        end = trigger.t_end + self.config.cut_pad_s
        if self._horizon is not None:
            end = min(end, self._horizon)
        floor = max(self._start_t, trigger.t_end - self.config.max_cut_back_s)

        start = trigger.t
        boundary: float | None = None
        consumed = 0

        for utterance in reversed(self._utterances):
            if utterance.t_end > trigger.t:
                continue  # the trigger utterance itself, or anything after it
            if utterance.t_end <= floor:
                break
            if start - utterance.t_end >= self.config.silence_gap_s:
                boundary = utterance.t_end  # a real pause: don't reach past it
                break
            if consumed >= self.config.max_cut_back_utterances:
                boundary = utterance.t_end  # far enough back for one instruction
                break
            consumed += 1
            start = utterance.t

        lower_bound = floor if boundary is None else boundary
        start = max(floor, lower_bound, start - self.config.cut_pad_s)

        if end - start <= EPSILON:
            return None
        evidence = trigger.evidence or "cut"
        return Cut(start, end, f"said: {evidence}")
