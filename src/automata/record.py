"""Capturing a screen and a camera at once, and knowing how far apart they began.

Two recorders never start at the same instant. Device warm-up differs, and the
gap is not small: measured repeatedly on a MacBook it lands between 0.3 and 0.9
seconds, which is the same order as the director's gaze debounce. Left
unmeasured it silently shifts every gaze sample against the words.

So both recorders are **stopped at the same instant** instead. Whichever started
earlier ends up with the longer file, and the difference is exactly the offset
between them::

    screen  7.633s   ]
    camera  6.952s   ]  camera began 0.681s later  ->  its offset_s

That turns the one value nobody can eyeball into a measurement, and it is the
reason this module exists rather than leaving people to clap on camera.
"""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

SCREEN_WIDTH = 1920
"""Retina screens capture at 3456x2234 or larger, which is far more data than
the pipeline needs and slows every later stage. 1080p keeps text legible."""

CAMERA_WIDTH = 1280
FRAMERATE = 30
STOP_TIMEOUT_S = 30.0


class RecordError(RuntimeError):
    pass


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    kind: str  # "video" | "audio"

    def __str__(self) -> str:
        return f"[{self.index}] {self.name}"


@dataclass(frozen=True)
class Devices:
    video: tuple[Device, ...]
    audio: tuple[Device, ...]

    def screen(self) -> Device | None:
        """avfoundation exposes displays as video devices named 'Capture screen'."""
        return next((d for d in self.video if "capture screen" in d.name.lower()), None)

    def camera(self) -> Device | None:
        """The first real camera, skipping displays and Continuity extras.

        'Desk View' is a second virtual camera pointed at the desk; it is never
        the head-and-shoulders shot gaze needs.
        """
        for device in self.video:
            low = device.name.lower()
            if "capture screen" in low or "desk view" in low:
                continue
            return device
        return None

    def microphone(self) -> Device | None:
        return self.audio[0] if self.audio else None


@dataclass(frozen=True)
class Recording:
    screen_path: Path
    camera_path: Path
    screen_duration: float
    camera_duration: float

    @property
    def camera_offset_s(self) -> float:
        """Master-time offset for whichever clip started later.

        Positive means the camera began after the screen, which is the usual
        case since a webcam takes longer to wake than a display grab.
        """
        return round(self.screen_duration - self.camera_duration, 3)


def list_devices() -> Devices:
    """Ask avfoundation what it can see."""
    _require("ffmpeg")
    done = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True, check=False,
    )
    video: list[Device] = []
    audio: list[Device] = []
    bucket = None
    for line in done.stderr.splitlines():
        if "AVFoundation video devices" in line:
            bucket = video
            continue
        if "AVFoundation audio devices" in line:
            bucket = audio
            continue
        match = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if match and bucket is not None:
            bucket.append(
                Device(int(match.group(1)), match.group(2),
                       "video" if bucket is video else "audio")
            )
    if not video:
        raise RecordError(
            "avfoundation reported no video devices. On macOS this usually means "
            "the terminal lacks Screen Recording permission -- grant it in "
            "System Settings > Privacy & Security > Screen Recording, then "
            "restart the terminal."
        )
    return Devices(tuple(video), tuple(audio))


class Session:
    """Two recorders running together, stopped together."""

    def __init__(
        self,
        directory: str | Path,
        *,
        screen: Device,
        camera: Device,
        microphone: Device | None,
        framerate: int = FRAMERATE,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.screen_path = self.directory / "screen.mp4"
        self.camera_path = self.directory / "camera.mp4"
        self._screen = screen
        self._camera = camera
        self._microphone = microphone
        self._framerate = framerate
        self._procs: list[subprocess.Popen] = []
        self._started: float | None = None

    def start(self) -> None:
        if self._procs:
            raise RecordError("already recording")
        _require("ffmpeg")

        # The screen goes first: it is the faster device to open, so the camera's
        # warm-up shows up as a positive offset rather than a negative one.
        self._procs = [
            self._spawn(
                f"{self._screen.index}:none", self.screen_path,
                ["-capture_cursor", "1"], SCREEN_WIDTH, audio=False,
            ),
            self._spawn(
                f"{self._camera.index}:"
                f"{self._microphone.index if self._microphone else 'none'}",
                self.camera_path, [], CAMERA_WIDTH,
                audio=self._microphone is not None,
            ),
        ]
        self._started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started is None else time.monotonic() - self._started

    def stop(self) -> Recording:
        """Stop both at the same instant, then measure what each captured."""
        if not self._procs:
            raise RecordError("not recording")

        # The loop below is the measurement. Anything slow between these two
        # signals would be indistinguishable from device warm-up and corrupt the
        # offset, so nothing else belongs here.
        for proc in self._procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)

        for proc in self._procs:
            try:
                proc.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._procs = []

        for path in (self.screen_path, self.camera_path):
            if not path.exists() or path.stat().st_size == 0:
                raise RecordError(f"{path.name} was not written; nothing was captured")

        from .probe import probe

        return Recording(
            screen_path=self.screen_path,
            camera_path=self.camera_path,
            screen_duration=probe(self.screen_path, measure_audio=False).duration_s,
            camera_duration=probe(self.camera_path, measure_audio=False).duration_s,
        )

    def abort(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
        self._procs = []

    # -- internals -------------------------------------------------------

    def _spawn(self, source: str, out: Path, extra: list[str], width: int, *, audio: bool):
        argv = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "avfoundation", "-framerate", str(self._framerate),
            *extra, "-i", source,
            # Even dimensions, because H.264 4:2:0 requires them.
            "-vf", f"scale={width}:-2",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-pix_fmt", "yuv420p",
        ]
        if audio:
            argv += ["-c:a", "aac", "-b:a", "192k"]
        argv += ["-y", str(out)]
        return subprocess.Popen(
            argv, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise RecordError(f"{tool} not found on PATH; install ffmpeg")
