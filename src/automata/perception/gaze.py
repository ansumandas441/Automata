"""Is the creator looking at the lens?

Deliberately the cheapest useful detector: a face landmarker at ~8Hz on
downscaled frames, reduced to two numbers -- how far the head is turned, and how
far the eyes are looking off-axis. That is enough to separate "talking to the
viewer" from "reading the screen", and it leaves the machine free to record.

Both numbers come from the model rather than from geometry we invent:
`output_facial_transformation_matrixes` gives a real head pose, and the
`eyeLook*` blendshapes give eye direction independent of the head. Someone whose
head is square to the camera while their eyes track code on a second monitor is
the common case, and head pose alone would call that "looking at the lens".

Two things matter more than the accuracy of the estimate:

**Frames are timed by PTS, never by index.** Screen and webcam capture drop
frames, and macOS capture is variable-frame-rate; counting frames and
multiplying by 1/fps drifts silently, and by the end of a long recording the
gaze track no longer lines up with the speech.

**A missing face is `UNKNOWN`, not `AWAY`.** Leaving frame, poor light, or a
hand across the face all produce no detection. Reported as "looking away", each
one would swing the camera; reported as unknown, the director holds the shot.
"""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..events import GazeSample, GazeState
from ..sources import SourceClip

PRODUCER = "gaze"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_CACHE = Path.home() / ".cache" / "automata" / "face_landmarker.task"

_EYE_LOOK = (
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
)
_BLINK = ("eyeBlinkLeft", "eyeBlinkRight")


@dataclass(frozen=True)
class GazeConfig:
    sample_hz: float = 8.0
    """Well above the director's 0.8s debounce, far below the frame rate.
    Sampling every frame would trade a lot of CPU for no extra decisions."""

    frame_width: int = 480
    """The landmarker does not need a 4K webcam frame."""

    max_yaw_deg: float = 16.0
    """Head turn away from the lens. Beyond this they are addressing the screen."""

    max_pitch_deg: float = 14.0
    """Head tilt. Catches looking down at a keyboard or a lower monitor."""

    max_eye_offset: float = 0.34
    """Eye direction independent of the head, from the blendshape scores.
    Generous: this is a tiebreaker, not the primary signal."""

    max_blink: float = 0.6
    """Above this the eyes are shut, so eye direction is meaningless -- fall back
    to head pose rather than reading a blink as a glance away."""

    min_detection_confidence: float = 0.5
    model_path: Path | None = None
    """Defaults to the cached bundle, downloading it once if absent."""


def track(clip: SourceClip, config: GazeConfig | None = None) -> list[GazeSample]:
    """Sample gaze across a clip, timestamped on the master timeline."""
    config = config or GazeConfig()
    landmarker = _load(config)
    try:
        return [
            GazeSample(
                t=clip.to_master(local_t),
                producer=PRODUCER,
                state=state,
                confidence=confidence,
            )
            for local_t, state, confidence in _samples(clip, landmarker, config)
        ]
    finally:
        landmarker.close()


def _samples(
    clip: SourceClip, landmarker, config: GazeConfig
) -> Iterator[tuple[float, str, float]]:
    import av
    import mediapipe as mp

    interval = 1.0 / config.sample_hz
    container = av.open(clip.path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        next_sample = 0.0
        last_ms = -1

        for frame in container.decode(stream):
            if frame.pts is None:
                continue  # no presentation time: cannot place it on the timeline
            t = frame.pts * time_base
            if t + 1e-6 < next_sample:
                continue
            next_sample = t + interval

            # VIDEO mode tracks across calls and requires strictly increasing
            # timestamps; duplicate PTS would otherwise raise.
            timestamp_ms = max(int(t * 1000), last_ms + 1)
            last_ms = timestamp_ms

            height = max(2, round(frame.height * config.frame_width / max(frame.width, 1)))
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame.to_ndarray(format="rgb24", width=config.frame_width, height=height),
            )
            yield (t, *_classify(landmarker.detect_for_video(image, timestamp_ms), config))
    finally:
        container.close()


def _classify(result, config: GazeConfig) -> tuple[str, float]:
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if not matrices:
        return GazeState.UNKNOWN, 1.0

    yaw, pitch = _head_pose(matrices[0])
    if yaw is None:
        return GazeState.UNKNOWN, 1.0

    scores = _blendshapes(result)
    eye = _eye_offset(scores, config)

    off_axis = max(
        abs(yaw) / config.max_yaw_deg,
        abs(pitch) / config.max_pitch_deg,
        eye / config.max_eye_offset,
    )
    state = GazeState.LOOKING if off_axis <= 1.0 else GazeState.AWAY
    # Confidence falls off near the threshold, where the call is a coin flip.
    return state, round(min(1.0, abs(off_axis - 1.0) + 0.5), 3)


def _head_pose(matrix) -> tuple[float | None, float | None]:
    """Yaw and pitch in degrees from the 4x4 facial transformation matrix."""
    try:
        r = [[float(matrix[row][col]) for col in range(3)] for row in range(3)]
    except (TypeError, IndexError, ValueError):  # pragma: no cover -- unexpected shape
        return None, None
    yaw = math.degrees(math.atan2(-r[2][0], math.hypot(r[2][1], r[2][2])))
    pitch = math.degrees(math.atan2(r[2][1], r[2][2]))
    return yaw, pitch


def _blendshapes(result) -> dict[str, float]:
    groups = getattr(result, "face_blendshapes", None)
    if not groups:
        return {}
    return {c.category_name: c.score for c in groups[0]}


def _eye_offset(scores: dict[str, float], config: GazeConfig) -> float:
    """How far the eyes look off-axis, ignoring blinks.

    Returns 0.0 when blendshapes are unavailable, leaving head pose to decide.
    """
    if not scores:
        return 0.0
    if max((scores.get(name, 0.0) for name in _BLINK), default=0.0) > config.max_blink:
        return 0.0
    return max((scores.get(name, 0.0) for name in _EYE_LOOK), default=0.0)


def _load(config: GazeConfig):
    try:
        import mediapipe  # noqa: F401
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )
    except ImportError as exc:  # pragma: no cover -- optional dependency
        raise RuntimeError(
            "gaze tracking needs mediapipe: pip install 'automata[perception]'"
        ) from exc

    return FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(config.model_path))),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_detection_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
    )


def ensure_model(path: Path | None = None) -> Path:
    """Return the landmarker bundle, fetching it once into the cache if absent."""
    target = Path(path) if path else MODEL_CACHE
    if target.exists():
        return target
    if path is not None:
        raise FileNotFoundError(f"face landmarker bundle not found: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)  # noqa: S310 -- constant https URL
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download the face landmarker bundle from {MODEL_URL}. "
            f"Fetch it manually and pass GazeConfig(model_path=...): {exc}"
        ) from exc
    partial.replace(target)  # atomic, so an interrupted run can't cache a stub
    return target
