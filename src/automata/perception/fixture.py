"""Replay perception events from JSON.

Step 1 runs the whole orchestrator off fixtures: no camera, no model weights,
no GPU. That is deliberate -- the director is where the bugs live, and it can
only be tested properly if its inputs are cheap to write by hand.

Fixture events are deliberately allowed to be out of order in the file. The bus
puts them back in order, which is exactly the behaviour worth exercising.

Format::

    {"events": [
      {"type": "utterance", "t": 3.0, "end": 5.2, "text": "let me show you this"},
      {"type": "gaze",      "t": 3.1, "state": "away"},
      {"type": "source",    "t": 60.0, "source_id": "cam0", "role": "camera",
                            "available": false}
    ]}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..events import (
    Event,
    GazeSample,
    Intent,
    ScreenActivity,
    SourceStatus,
    Utterance,
    Word,
)
from ..sources import SourceRole

SPEECH_PRODUCER = "stt"
GAZE_PRODUCER = "gaze"
SYSTEM_PRODUCER = "system"


def _utterance(raw: dict) -> Utterance:
    return Utterance(
        t=float(raw["t"]),
        producer=raw.get("producer", SPEECH_PRODUCER),
        text=raw["text"],
        end=float(raw.get("end", raw["t"])),
        words=tuple(
            Word(w["text"], float(w["t"]), float(w["end"])) for w in raw.get("words", ())
        ),
    )


def _gaze(raw: dict) -> GazeSample:
    return GazeSample(
        t=float(raw["t"]),
        producer=raw.get("producer", GAZE_PRODUCER),
        state=raw["state"],
        confidence=float(raw.get("confidence", 1.0)),
    )


def _intent(raw: dict) -> Intent:
    return Intent(
        t=float(raw["t"]),
        producer=raw.get("producer", "intent"),
        kind=raw["kind"],
        end=float(raw.get("end", raw["t"])),
        confidence=float(raw.get("confidence", 1.0)),
        evidence=raw.get("evidence", ""),
    )


def _source(raw: dict) -> SourceStatus:
    return SourceStatus(
        t=float(raw["t"]),
        producer=raw.get("producer", SYSTEM_PRODUCER),
        source_id=raw["source_id"],
        role=SourceRole(raw["role"]),
        available=bool(raw["available"]),
    )


def _activity(raw: dict) -> ScreenActivity:
    return ScreenActivity(
        t=float(raw["t"]),
        producer=raw.get("producer", "screen"),
        level=float(raw["level"]),
    )


_BUILDERS = {
    "utterance": _utterance,
    "gaze": _gaze,
    "intent": _intent,
    "source": _source,
    "activity": _activity,
}


def load_events(path: str | Path) -> list[Event]:
    data = json.loads(Path(path).read_text())
    events: list[Event] = []
    for index, raw in enumerate(data["events"]):
        kind = raw.get("type")
        builder = _BUILDERS.get(kind)
        if builder is None:
            raise ValueError(f"events[{index}]: unknown event type {kind!r}")
        events.append(builder(raw))
    return events


def save_events(events: Iterable[Event], path: str | Path) -> None:
    """Write events in the same format `load_events` reads.

    Real perception writes this file; planning reads it. Keeping the two ends of
    the pipeline on one hand-editable format means a bad transcription can be
    corrected in a text editor instead of re-run.
    """
    ordered = sorted(events, key=lambda e: (e.t, e.producer))
    payload = {"events": [_encode(e) for e in ordered]}
    Path(path).write_text(json.dumps(payload, indent=1) + "\n")


def _encode(event: Event) -> dict:
    if isinstance(event, Utterance):
        raw = {"type": "utterance", "t": round(event.t, 3), "end": round(event.end, 3),
               "text": event.text}
        if event.words:
            raw["words"] = [
                {"text": w.text, "t": round(w.t, 3), "end": round(w.t_end, 3)}
                for w in event.words
            ]
        return raw
    if isinstance(event, GazeSample):
        return {"type": "gaze", "t": round(event.t, 3), "state": event.state,
                "confidence": round(event.confidence, 3)}
    if isinstance(event, Intent):
        return {"type": "intent", "t": round(event.t, 3), "end": round(event.end, 3),
                "kind": event.kind, "confidence": round(event.confidence, 3),
                "evidence": event.evidence}
    if isinstance(event, SourceStatus):
        return {"type": "source", "t": round(event.t, 3), "source_id": event.source_id,
                "role": event.role.value, "available": event.available}
    if isinstance(event, ScreenActivity):
        return {"type": "activity", "t": round(event.t, 3), "level": round(event.level, 3)}
    raise TypeError(f"cannot serialise {type(event).__name__}")
