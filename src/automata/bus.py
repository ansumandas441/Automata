"""Watermarked, time-ordered event bus.

Producers run at wildly different cadences -- speech segments arrive in bursts,
gaze at ~8Hz, the intent classifier at ~0.2Hz with 0.3-1.5s of its own latency
-- so events *arrive* badly out of order. Consuming them in arrival order lets
a late verdict about t=10.0 be applied at t=11.4, which is exactly how an
auto-editor ends up cutting the wrong sentence.

Each producer declares a **watermark**: "I will never emit an event earlier
than this." The bus releases events only at or below the minimum watermark
across all registered producers, restoring strict time order. A producer that
has finished closes, dropping out of the minimum.

Queues are bounded. On overflow the producer's declared policy decides: dense
sensor samples may shed load, but speech and intents never may -- losing one
silently changes the edit, so it raises instead.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterator
from enum import Enum

from .events import Event


class DropPolicy(Enum):
    NEVER = "never"
    """Overflow is a correctness bug -- raise rather than change the edit."""

    OLDEST = "oldest"
    """Lossy. Only for redundant high-rate samples (gaze, screen activity)."""


class BusOverflow(RuntimeError):
    pass


class WatermarkRegression(RuntimeError):
    """A producer emitted or declared a time it had already passed."""


class EventBus:
    def __init__(self, capacity: int = 200_000) -> None:
        self._capacity = capacity
        self._heap: list[tuple[float, int, Event]] = []
        self._seq = 0
        self._watermarks: dict[str, float] = {}
        self._policies: dict[str, DropPolicy] = {}
        self._dropped: dict[str, int] = {}

    # -- registration ----------------------------------------------------

    def register(self, producer: str, drop: DropPolicy = DropPolicy.NEVER) -> None:
        if producer in self._policies:
            raise ValueError(f"producer {producer!r} already registered")
        self._policies[producer] = drop
        self._watermarks[producer] = -math.inf
        self._dropped[producer] = 0

    def close(self, producer: str) -> None:
        """Mark a producer finished so it stops holding back the watermark."""
        self._require(producer)
        self._watermarks[producer] = math.inf

    def close_all(self) -> None:
        for producer in self._policies:
            self._watermarks[producer] = math.inf

    # -- production ------------------------------------------------------

    def publish(self, event: Event) -> None:
        self._require(event.producer)
        if event.t < self._watermarks[event.producer]:
            raise WatermarkRegression(
                f"{event.producer} published t={event.t:.3f} behind its own "
                f"watermark {self._watermarks[event.producer]:.3f}"
            )
        if len(self._heap) >= self._capacity:
            self._overflow(event.producer)
        heapq.heappush(self._heap, (event.t, self._seq, event))
        self._seq += 1

    def advance(self, producer: str, watermark: float) -> None:
        """Declare that `producer` will emit nothing before `watermark`."""
        self._require(producer)
        current = self._watermarks[producer]
        if watermark < current:
            raise WatermarkRegression(
                f"{producer} watermark went backwards: {current:.3f} -> {watermark:.3f}"
            )
        self._watermarks[producer] = watermark

    # -- consumption -----------------------------------------------------

    @property
    def watermark(self) -> float:
        """Latest time at which the ordering of released events is settled."""
        if not self._watermarks:
            return math.inf
        return min(self._watermarks.values())

    def drain(self) -> Iterator[Event]:
        """Yield every settled event in strict time order."""
        limit = self.watermark
        while self._heap and self._heap[0][0] <= limit:
            yield heapq.heappop(self._heap)[2]

    def pending(self) -> int:
        return len(self._heap)

    def dropped(self) -> dict[str, int]:
        """Per-producer drop counts. Non-zero means the edit saw partial data."""
        return {k: v for k, v in self._dropped.items() if v}

    # -- internals -------------------------------------------------------

    def _require(self, producer: str) -> None:
        if producer not in self._policies:
            raise KeyError(f"unregistered producer {producer!r}")

    def _overflow(self, producer: str) -> None:
        if self._policies[producer] is DropPolicy.NEVER:
            raise BusOverflow(
                f"bus full ({self._capacity}) and {producer!r} may not drop events; "
                "the consumer is not keeping up"
            )
        victim = self._evict_oldest_droppable()
        if victim is None:
            raise BusOverflow(
                f"bus full ({self._capacity}) and nothing droppable is queued"
            )
        self._dropped[victim] += 1

    def _evict_oldest_droppable(self) -> str | None:
        """Shed the earliest sheddable sample and re-heapify.

        O(n), but overflow only happens when the consumer has already fallen
        badly behind -- correctness of what remains matters more than the cost
        of the rebuild.
        """
        victim_index = min(
            (
                index
                for index, (_, _, event) in enumerate(self._heap)
                if self._policies[event.producer] is DropPolicy.OLDEST
            ),
            key=lambda index: self._heap[index][:2],
            default=None,
        )
        if victim_index is None:
            return None
        producer = self._heap[victim_index][2].producer
        self._heap.pop(victim_index)
        heapq.heapify(self._heap)
        return producer
