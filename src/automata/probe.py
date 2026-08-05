"""Reading what is actually in a media file, so nothing has to be guessed.

Every value in a project file should be measured. Two of these measurements have
each caught a run-ruining problem in practice: a screen capture whose audio
stream exists but contains only silence, and a pair of clips whose durations
differ enough that part of one is unusable.

Only ffprobe and ffmpeg are used, so this stays in the core with no new
dependencies -- and no frame is ever decoded for pixels.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SILENCE_DB = -60.0
"""Below this a track carries no usable speech.

A muted or mis-routed recorder produces roughly -91 dB, the AAC noise floor.
Real speech sits nearer -25 dB, so the threshold is nowhere near either.
"""

PROBE_WINDOW_S = 45.0
"""How much audio to measure. Sampled from the middle, because the opening
seconds of a recording are often silent while someone settles in."""


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_s: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio_stream: bool = False
    mean_volume_db: float | None = None

    @property
    def has_speech(self) -> bool:
        """Whether the audio track carries anything worth transcribing."""
        if not self.has_audio_stream or self.mean_volume_db is None:
            return False
        return self.mean_volume_db > SILENCE_DB

    def describe(self) -> str:
        parts = [f"{self.duration_s:.2f}s"]
        if self.width and self.height:
            parts.append(f"{self.width}x{self.height}")
        if self.fps:
            parts.append(f"{self.fps:.3f} fps".replace(".000 ", " "))
        if not self.has_audio_stream:
            parts.append("no audio")
        elif self.mean_volume_db is None:
            parts.append("audio (unmeasured)")
        elif self.has_speech:
            parts.append(f"audio {self.mean_volume_db:.1f} dB")
        else:
            parts.append(f"audio {self.mean_volume_db:.1f} dB = silent")
        return " · ".join(parts)


def probe(path: str | Path, *, measure_audio: bool = True) -> MediaInfo:
    """Measure one media file. Raises :class:`ProbeError` if it cannot be read."""
    path = Path(path)
    if not path.exists():
        raise ProbeError(f"no such file: {path}")
    _require("ffprobe")

    raw = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ProbeError(f"{path.name}: ffprobe returned nothing usable") from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _duration(data.get("format", {}), video, audio)
    if duration is None:
        raise ProbeError(f"{path.name}: no duration in the container")

    volume = None
    if audio is not None and measure_audio:
        volume = _mean_volume(path, duration)

    return MediaInfo(
        path=path,
        duration_s=duration,
        width=_int(video, "width"),
        height=_int(video, "height"),
        fps=_fps(video),
        has_audio_stream=audio is not None,
        mean_volume_db=volume,
    )


# -- internals -----------------------------------------------------------


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise ProbeError(f"{tool} not found on PATH; install ffmpeg")


def _run(argv: list[str]) -> str:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ProbeError(f"could not run {argv[0]}: {exc}") from exc
    return done.stdout


def _duration(fmt: dict, video: dict | None, audio: dict | None) -> float | None:
    # Container duration is the honest answer; stream durations are the fallback
    # for files whose container header is incomplete.
    for source in (fmt, video or {}, audio or {}):
        value = source.get("duration")
        if value is None:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return seconds
    return None


def _int(stream: dict | None, key: str) -> int | None:
    if not stream:
        return None
    try:
        return int(stream[key])
    except (KeyError, TypeError, ValueError):
        return None


def _fps(stream: dict | None) -> float | None:
    if not stream:
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if not value or "/" not in value:
            continue
        num, _, den = value.partition("/")
        try:
            num, den = float(num), float(den)
        except ValueError:
            continue
        if den:
            rate = num / den
            if rate > 0:
                return rate
    return None


def _mean_volume(path: Path, duration: float) -> float | None:
    """Mean loudness over a window from the middle of the file."""
    _require("ffmpeg")
    start = max(0.0, (duration - PROBE_WINDOW_S) / 2)
    done = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin",
            "-ss", f"{start:.3f}", "-t", f"{PROBE_WINDOW_S:.3f}",
            "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )
    for line in done.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except (IndexError, ValueError):
                return None
    return None
