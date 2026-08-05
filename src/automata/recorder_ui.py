"""The terminal interface for a recording session.

Kept apart from :mod:`automata.record` so the capture logic stays testable
without a terminal, and so the display can be replaced without touching the
measurement that matters.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path

from .record import Device, Devices, RecordError, Recording, Session, list_devices

DOT = "●"


def choose(devices: Devices, screen_i: int | None, camera_i: int | None, audio_i: int | None):
    """Resolve the three devices, preferring anything named explicitly."""
    screen = _by_index(devices.video, screen_i) or devices.screen()
    camera = _by_index(devices.video, camera_i) or devices.camera()
    microphone = (
        _by_index(devices.audio, audio_i) if audio_i is not None else devices.microphone()
    )

    if screen is None:
        raise RecordError("no screen capture device found; pass --screen with an index")
    if camera is None:
        raise RecordError("no camera found; pass --camera with an index")
    if screen.index == camera.index:
        raise RecordError("the screen and camera cannot be the same device")
    return screen, camera, microphone


def _by_index(pool, index: int | None) -> Device | None:
    if index is None:
        return None
    match = next((d for d in pool if d.index == index), None)
    if match is None:
        raise RecordError(f"no device with index {index}; run `automata record --list`")
    return match


def print_devices(devices: Devices) -> None:
    print("video devices")
    for device in devices.video:
        tag = ""
        if device == devices.screen():
            tag = "  <- screen default"
        elif device == devices.camera():
            tag = "  <- camera default"
        print(f"  {device}{tag}")
    print("\naudio devices")
    for device in devices.audio:
        tag = "  <- default" if device == devices.microphone() else ""
        print(f"  {device}{tag}")
    if not devices.audio:
        print("  (none found; the recording will be silent)")


def run_session(
    directory: Path,
    screen: Device,
    camera: Device,
    microphone: Device | None,
) -> Recording:
    """Record until the user stops, showing elapsed time as it goes."""
    session = Session(directory, screen=screen, camera=camera, microphone=microphone)

    print(f"  screen   {screen.name}")
    print(f"  camera   {camera.name}")
    print(f"  audio    {microphone.name if microphone else 'none -- no speech will be captured'}")
    print(f"  writing  {directory}/")
    print()

    session.start()
    # Let both recorders open their devices before promising anything: a
    # permission failure surfaces within the first second.
    time.sleep(1.0)
    _check_alive(session)

    stop = threading.Event()
    # A redrawing timer only makes sense on a terminal; piped to a file it would
    # write one line per tenth of a second.
    ticker = None
    if sys.stdout.isatty():
        ticker = threading.Thread(target=_tick, args=(session, stop), daemon=True)
        ticker.start()
    else:
        print("  recording; send a newline on stdin to stop")

    try:
        _wait_for_stop()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop.set()
        if ticker is not None:
            ticker.join(timeout=1.0)
            _clear_line()

    print("  stopping both recorders at once...")
    return session.stop()


def _wait_for_stop() -> None:
    """Block until the user stops the recording.

    A newline on stdin ends it, whether typed at a terminal or piped in by a
    script. Only when stdin is closed outright -- a daemon, or ``</dev/null`` --
    is there nothing to wait for, and then the signal handler has to do it.
    """
    try:
        input()
    except EOFError:
        threading.Event().wait()


def _tick(session: Session, stop: threading.Event) -> None:
    while not stop.is_set():
        elapsed = session.elapsed
        stamp = f"{int(elapsed // 60)}:{elapsed % 60:04.1f}"
        sys.stdout.write(f"\r  {DOT} recording  {stamp}    press ENTER to stop ")
        sys.stdout.flush()
        stop.wait(0.1)


def _clear_line() -> None:
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()


CHECK_SECONDS = 4.0


def check_framing(camera: Device, microphone: Device | None, directory: Path) -> bool:
    """Record a few seconds and report whether the camera is usable.

    Framing is the single thing most likely to waste a take: gaze needs a face
    it can actually find, and a face that sits in a corner of a tall frame is
    invisible to the detector even though a person can see it perfectly well.
    Four seconds now is cheaper than discovering it after a long recording.
    """
    from .probe import probe
    from .record import Session as _Session

    directory.mkdir(parents=True, exist_ok=True)
    session = _Session(directory, screen=camera, camera=camera, microphone=microphone)
    clip = directory / "_check.mp4"

    print(
        f"  checking for {CHECK_SECONDS:.0f}s -- look at the lens and say something\n"
    )
    proc = session._spawn(  # noqa: SLF001
        f"{camera.index}:{microphone.index if microphone else 'none'}",
        clip, [], 1280, audio=microphone is not None,
    )
    time.sleep(CHECK_SECONDS)
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=20)

    if not clip.exists() or clip.stat().st_size == 0:
        print("  could not capture from the camera at all.")
        return False

    info = probe(clip)
    states = _gaze_states(clip)
    clip.unlink(missing_ok=True)

    total = sum(states.values()) or 1
    seen = 100 * (total - states.get("unknown", 0)) / total
    print(f"  camera   {info.width}x{info.height}"
          + ("  (portrait -- unusual for a webcam)" if info.height > info.width else ""))
    # Someone speaking into a laptop mic lands near -25 dB. Room tone with
    # nobody talking sits around -55, which passes the silence test used for
    # picking a track but is not evidence the microphone heard a voice.
    speech_db = info.mean_volume_db
    if speech_db is not None:
        if speech_db > -45:
            verdict = "ok"
        elif speech_db > -70:
            verdict = "very quiet -- room tone, not a voice?"
        else:
            verdict = "SILENT -- the microphone captured nothing"
        print(f"  audio    {speech_db:.1f} dB  {verdict}")
    else:
        print("  audio    no track -- speech signals will not work")
    print(f"  face     found in {seen:.0f}% of frames")

    if seen >= 60:
        print("\n  Framing looks good. Record away.")
        return True

    print(
        "\n  The detector cannot find your face reliably. Gaze will report\n"
        "  'unknown' and hold one shot for the whole video.\n"
        "\n  Try: sit so your head and shoulders fill the middle of the frame,\n"
        "  tilt the screen so the lens points at your face rather than past it,\n"
        "  and add light in front of you.\n"
        "\n  Or skip gaze entirely -- `--talking-head` lets speech decide."
    )
    return False


def _gaze_states(clip: Path) -> dict[str, int]:
    from collections import Counter

    from .gaze_check import sample_states

    return Counter(sample_states(clip))


def _check_alive(session: Session) -> None:
    for proc in session._procs:  # noqa: SLF001 -- the UI owns this session
        if proc.poll() is None:
            continue
        detail = ""
        if proc.stderr is not None:
            detail = proc.stderr.read().decode("utf-8", "replace").strip()
        session.abort()
        raise RecordError(
            "a recorder exited immediately.\n"
            + (f"  {detail}\n" if detail else "")
            + "  On macOS, grant Screen Recording and Camera permission to this "
            "terminal in System Settings > Privacy & Security, then restart it."
        )


__all__ = ["choose", "list_devices", "print_devices", "run_session"]
