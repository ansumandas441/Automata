"""Wiring: perception events in, timeline out.

Post-hoc mode is a degenerate case of the live pipeline -- every producer
closes immediately, so the watermark jumps to infinity and everything drains at
once. Live mode will reuse this exact path with real watermarks and a delay
buffer; nothing here needs to change for it.
"""

from __future__ import annotations

from collections.abc import Iterable

from .bus import DropPolicy, EventBus
from .config import DirectorConfig
from .director import Director
from .events import Event, Utterance
from .perception.intent import IntentPipeline, IntentTagger
from .sources import SourceClip, overlap_window
from .timeline import Timeline

_DROP_POLICIES = {
    "gaze": DropPolicy.OLDEST,
    "screen": DropPolicy.OLDEST,
}
"""Redundant high-rate sensors may shed load. Speech and intents may not."""


def plan(
    events: Iterable[Event],
    *,
    config: DirectorConfig | None = None,
    start_t: float = 0.0,
    end_t: float | None = None,
    taggers: tuple[IntentTagger, ...] = (),
) -> Timeline:
    """Run the director over an event stream and return the master timeline.

    Utterances are tagged on the way in unless the stream already carries its
    own ``Intent`` events (fixtures may supply either).
    """
    events = list(events)
    intents = IntentPipeline(*taggers)

    bus = EventBus()
    registered: set[str] = set()

    def publish(event: Event) -> None:
        if event.producer not in registered:
            bus.register(event.producer, _DROP_POLICIES.get(event.producer, DropPolicy.NEVER))
            registered.add(event.producer)
        bus.publish(event)

    has_own_intents = any(e.producer == "intent" for e in events)
    for event in events:
        publish(event)
        if isinstance(event, Utterance) and not has_own_intents:
            publish(intents.tag(event))

    bus.close_all()

    if end_t is None:
        end_t = max((e.t_end for e in events), default=start_t)
    end_t = max(end_t, start_t)

    director = Director(config, start_t=start_t, horizon=end_t)
    for event in bus.drain():
        if event.t > end_t:
            break  # events past the editable window cannot affect the output
        director.handle(event)
    director.advance(end_t)

    timeline = director.finish(end_t)
    dropped = bus.dropped()
    if dropped:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        timeline = Timeline(
            segments=timeline.segments,
            cuts=timeline.cuts,
            duration=timeline.duration,
            start=timeline.start,
            kept_spans=timeline.kept_spans,
            warnings=(*timeline.warnings, f"dropped events under backpressure: {summary}"),
        )
    return timeline


def editable_window(clips: Iterable[SourceClip]) -> tuple[float, float]:
    """Master-time span in which every clip has footage.

    Editing outside it means compositing against a black frame, so the director
    is never run there.
    """
    return overlap_window(list(clips))
