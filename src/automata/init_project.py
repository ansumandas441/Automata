"""Turning two recordings into a project file, with every value measured.

Hand-writing a project means running ffprobe for durations, checking which track
actually carries speech, and knowing the schema. That is three chances to get it
wrong before anything has run, and one of them -- pointing ``audio_from`` at a
silent track -- produces an empty transcript and a silent render with no error
to explain it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .probe import MediaInfo, ProbeError, probe

DURATION_GAP_WARN_S = 30.0
"""A gap this large usually means a recorder was started or stopped separately,
which is worth saying out loud -- the excess is unusable footage."""


@dataclass
class InitResult:
    path: Path
    project: dict
    screen: MediaInfo
    camera: MediaInfo
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def init_project(
    screen_path: str | Path,
    camera_path: str | Path,
    output: str | Path,
    *,
    talking_head: bool = False,
    camera_offset_s: float | None = None,
) -> InitResult:
    """Measure both files and write a project ready for ``analyze``.

    ``camera_offset_s`` is supplied by :mod:`automata.record`, which can measure
    it exactly. Left as ``None`` it stays 0.0, which is right only when both
    files came from one recorder started once.
    """
    screen = probe(screen_path)
    camera = probe(camera_path)
    output = Path(output)

    notes: list[str] = []
    warnings: list[str] = []

    audio_from = _pick_audio(screen, camera, notes, warnings)
    _check_camera(camera, warnings)
    if camera_offset_s is None:
        _check_durations(screen, camera, warnings)
    else:
        # A measured offset explains the duration difference, so reporting it as
        # a mismatch would be noise.
        notes.append(
            f"camera offset {camera_offset_s:+.3f}s, measured from the recording"
        )

    # The screen is the detail-critical source -- text and scrolling show judder
    # far more than a face does -- so the output matches its frame rate and
    # resolution and the camera absorbs any conversion.
    fps = int(round(screen.fps)) if screen.fps else 30
    width = screen.width or 1920
    height = screen.height or 1080
    if screen.fps and abs(screen.fps - fps) > 0.01:
        notes.append(
            f"screen is {screen.fps:.3f} fps; output rounded to {fps} and both "
            "sources are resampled to it"
        )

    project = {
        "_note": (
            "Durations, frame rate and the audio track were measured, not guessed. "
            + (
                "offset_s was measured by stopping both recorders at the same "
                "instant, so it is exact."
                if camera_offset_s is not None
                else "offset_s is UNVERIFIED -- 0.0 is right when both files came "
                "from one recorder started once."
            )
        ),
        "sources": [
            _source("screen", "screen", screen, output.parent, audio_from, 0.0),
            _source("cam0", "camera", camera, output.parent, audio_from,
                    camera_offset_s or 0.0),
        ],
        "audio_from": audio_from,
        "director": _director(talking_head),
        "render": {
            "width": width,
            "height": height,
            "fps": fps,
            "crf": 18,
            "video_preset": "medium",
        },
    }

    output.write_text(json.dumps(project, indent=2) + "\n")
    return InitResult(output, project, screen, camera, notes, warnings)


def _source(
    source_id: str, role: str, info: MediaInfo, base: Path, audio_from: str, offset_s: float
) -> dict:
    try:
        path = str(info.path.resolve().relative_to(base.resolve()))
    except ValueError:
        path = str(info.path.resolve())  # not under the project dir; keep absolute
    return {
        "id": source_id,
        "role": role,
        "path": path,
        "offset_s": round(offset_s, 3),
        "duration_s": round(info.duration_s, 3),
        "has_audio": source_id == audio_from,
    }


def _pick_audio(
    screen: MediaInfo, camera: MediaInfo, notes: list[str], warnings: list[str]
) -> str:
    """Choose the track carrying speech, and say why."""
    candidates = [("screen", screen), ("cam0", camera)]
    speaking = [(name, info) for name, info in candidates if info.has_speech]

    if len(speaking) == 1:
        name, info = speaking[0]
        other = camera if name == "screen" else screen
        if other.has_audio_stream:
            notes.append(
                f"audio_from={name} ({info.mean_volume_db:.1f} dB); the other track "
                "exists but is silent"
            )
        return name

    if not speaking:
        warnings.append(
            "neither track carries speech. Transcription will find nothing -- "
            "check the recorder's microphone routing before running analyze."
        )
        return "cam0" if camera.has_audio_stream else "screen"

    # Both have speech: usually one mic captured twice, or system audio alongside
    # it. The camera is where a person's microphone normally sits.
    notes.append(
        f"both tracks carry speech (screen {screen.mean_volume_db:.1f} dB, "
        f"camera {camera.mean_volume_db:.1f} dB); chose cam0 -- change audio_from "
        "if the better microphone is on the screen recording"
    )
    return "cam0"


def _check_durations(screen: MediaInfo, camera: MediaInfo, warnings: list[str]) -> None:
    gap = abs(screen.duration_s - camera.duration_s)
    if gap < DURATION_GAP_WARN_S:
        return
    longer = "screen" if screen.duration_s > camera.duration_s else "camera"
    warnings.append(
        f"the two clips differ by {gap:.0f}s ({longer} is longer). Only the "
        "overlap is editable, so that much of it will be dropped -- if they were "
        "meant to start together, check offset_s."
    )


def _check_camera(camera: MediaInfo, warnings: list[str]) -> None:
    if camera.width and camera.width < 640:
        warnings.append(
            f"the camera is only {camera.width}px wide; face detection may "
            "struggle below about 640."
        )


def _director(talking_head: bool) -> dict:
    if talking_head:
        # Camera on top of the screen: the creator always faces it, so gaze
        # carries no information and only what they say decides.
        return {
            "default_layout": "camera_focus",
            "gaze_mode": "off",
            "intent_hold_s": None,
            "min_shot_s": 2.5,
        }
    return {
        "min_shot_s": 2.5,
        "gaze_debounce_s": 0.8,
        "intent_hold_s": 6.0,
        "gaze_mode": "follow",
    }


__all__ = ["InitResult", "ProbeError", "init_project"]
