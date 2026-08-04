"""Turn a timeline into one ffmpeg invocation.

Two ideas keep this tractable:

**Cuts first, composition second.** The kept spans are trimmed out of each
input and concatenated, so by the time compositing happens we are already on
the output clock and every timestamp in the filter graph means what it says.

**Fixed-size graph.** Compositing is a three-overlay stack whose *timing* comes
from ``enable=`` expressions, not from one filter chain per shot. A 40-minute
video with three hundred shot changes builds the same graph as a 40-second one;
only the expressions get longer. Naively concatenating per-shot chains would
produce thousands of filter instances.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import PipConfig, RenderConfig
from ..sources import SourceClip, SourceRole
from ..timeline import Layout, Timeline

_CORNERS = {
    "top_left": "{m}:{m}",
    "top_right": "W-w-{m}:{m}",
    "bottom_left": "{m}:H-h-{m}",
    "bottom_right": "W-w-{m}:H-h-{m}",
}


@dataclass(frozen=True)
class RenderPlan:
    argv: list[str]
    filter_graph: str
    graph_path: Path
    warnings: tuple[str, ...] = ()

    def write_graph(self) -> Path:
        self.graph_path.write_text(self.filter_graph)
        return self.graph_path

    def run(self) -> subprocess.CompletedProcess:
        """Materialise the graph next to the output and invoke ffmpeg.

        The graph goes to a file because ``-filter_complex`` as an argument
        blows past the OS argument limit on any recording long enough to have
        interesting edits.
        """
        if shutil.which(self.argv[0]) is None:
            raise FileNotFoundError(f"{self.argv[0]} not found on PATH")
        self.write_graph()
        return subprocess.run(self.argv, check=True)


def build(
    timeline: Timeline,
    sources: list[SourceClip],
    output: str | Path,
    config: RenderConfig | None = None,
    *,
    ffmpeg: str = "ffmpeg",
) -> RenderPlan:
    """Build the render for a timeline that has already had cuts applied."""
    config = config or RenderConfig()
    if not timeline.kept_spans:
        raise ValueError("timeline has no kept_spans; call Timeline.apply_cuts() first")

    screen = next((s for s in sources if s.role is SourceRole.SCREEN), None)
    camera = next((s for s in sources if s.role is SourceRole.CAMERA), None)
    if screen is None and camera is None:
        raise ValueError("no video sources")

    inputs = [clip for clip in (screen, camera) if clip is not None]
    index = {clip.id: position for position, clip in enumerate(inputs)}
    audio = next((clip for clip in inputs if clip.has_audio), None)

    chunks: list[str] = []
    warnings: list[str] = []
    spans = list(timeline.kept_spans)

    if screen is not None:
        chunks.append(
            _normalised_video(index[screen.id], "screen", screen, spans, config, warnings)
        )
    if camera is not None:
        chunks.append(
            _normalised_video(index[camera.id], "camera", camera, spans, config, warnings)
        )
    if audio is not None:
        chunks.append(_normalised_audio(index[audio.id], audio, spans, warnings))
    else:
        warnings.append("no source is marked has_audio; rendering silent")

    video_out, composite = _composite(timeline, screen, camera, config)
    chunks.append(composite)

    graph_path = _graph_path(output)
    argv = [ffmpeg, "-y"]
    for clip in inputs:
        argv += ["-i", clip.path]
    argv += ["-filter_complex_script", str(graph_path)]
    argv += ["-map", video_out]
    if audio is not None:
        argv += ["-map", "[aout]", "-c:a", config.audio_codec, "-b:a", config.audio_bitrate]
    argv += [
        "-c:v", config.video_codec,
        "-preset", config.video_preset,
        "-crf", str(config.crf),
        "-pix_fmt", "yuv420p",
        "-r", str(config.fps),
        "-movflags", "+faststart",
        str(output),
    ]

    return RenderPlan(
        argv=argv,
        filter_graph=";\n".join(c for c in chunks if c) + "\n",
        graph_path=graph_path,
        warnings=tuple(warnings),
    )


def _graph_path(output: str | Path) -> Path:
    return Path(output).with_suffix(".filtergraph.txt")


# -- input normalisation --------------------------------------------------


def _normalised_video(
    stream: int,
    label: str,
    clip: SourceClip,
    spans: list[tuple[float, float]],
    config: RenderConfig,
    warnings: list[str],
) -> str:
    """Canvas-fit the whole input once, then trim and concat the kept spans.

    ``fps=`` is not optional: screen capture is variable-frame-rate on macOS,
    and ``concat`` requires a consistent timebase across its inputs.
    """
    head = (
        f"[{stream}:v]fps={config.fps},"
        f"scale={config.width}:{config.height}:force_original_aspect_ratio=decrease,"
        f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1"
    )
    return _trim_and_concat(head, label, clip, spans, warnings, audio=False)


def _normalised_audio(
    stream: int,
    clip: SourceClip,
    spans: list[tuple[float, float]],
    warnings: list[str],
) -> str:
    head = f"[{stream}:a]aresample=async=1:first_pts=0"
    return _trim_and_concat(head, "aout_src", clip, spans, warnings, audio=True)


def _trim_and_concat(
    head: str,
    label: str,
    clip: SourceClip,
    spans: list[tuple[float, float]],
    warnings: list[str],
    *,
    audio: bool,
) -> str:
    trim, setpts = ("atrim", "asetpts") if audio else ("trim", "setpts")
    out = "[aout]" if audio else f"[{label}_full]"
    count = len(spans)

    lines: list[str] = []
    if count == 1:
        lines.append(f"{head}[{label}_src]")
        pieces = [f"{label}_src"]
    else:
        split = "asplit" if audio else "split"
        pads = "".join(f"[{label}_src{i}]" for i in range(count))
        lines.append(f"{head},{split}={count}{pads}")
        pieces = [f"{label}_src{i}" for i in range(count)]

    for i, ((start, end), pad) in enumerate(zip(spans, pieces, strict=True)):
        local_start = clip.to_local(start)
        local_end = clip.to_local(end)
        if local_start < 0:
            warnings.append(
                f"{clip.id}: kept span starts {abs(local_start):.2f}s before the clip; clamped"
            )
            local_start = 0.0
        lines.append(
            f"[{pad}]{trim}=start={local_start:.3f}:end={max(local_end, local_start):.3f},"
            f"{setpts}=PTS-STARTPTS[{label}_p{i}]"
        )

    if count == 1:
        # Rename rather than run a pointless single-input concat.
        lines[-1] = lines[-1].replace(f"[{label}_p0]", out)
    else:
        joined = "".join(f"[{label}_p{i}]" for i in range(count))
        kind = "v=0:a=1" if audio else "v=1:a=0"
        lines.append(f"{joined}concat=n={count}:{kind}{out}")

    return ";\n".join(lines)


# -- composition ----------------------------------------------------------


def _composite(
    timeline: Timeline,
    screen: SourceClip | None,
    camera: SourceClip | None,
    config: RenderConfig,
) -> tuple[str, str]:
    if camera is None:
        return "[screen_full]", ""
    if screen is None:
        return "[camera_full]", ""

    camera_full_spans = timeline.spans_for(Layout.CAMERA_FOCUS, Layout.CAMERA_ONLY)
    camera_pip_spans = timeline.spans_for(Layout.SCREEN_FOCUS)
    screen_pip_spans = timeline.spans_for(Layout.CAMERA_FOCUS)

    # Only fan out the branches a layout actually uses -- an unconsumed pad is
    # a hard ffmpeg error, not a warning.
    screen_pads = ["screen_base"] + (["screen_pip_src"] if screen_pip_spans else [])
    camera_pads = (["camera_base"] if camera_full_spans else []) + (
        ["camera_pip_src"] if camera_pip_spans else []
    )

    lines: list[str] = []
    lines += _fan_out("screen_full", screen_pads)
    lines += _fan_out("camera_full", camera_pads or ["camera_base"])
    if not camera_pads:
        lines.append("[camera_base]nullsink")  # camera never shown; keep the pad connected

    if screen_pip_spans:
        width = int(config.width * config.screen_pip.scale)
        lines.append(f"[screen_pip_src]scale={width}:-2[screen_pip]")
    if camera_pip_spans:
        width = int(config.width * config.camera_pip.scale)
        lines.append(f"[camera_pip_src]scale={width}:-2[camera_pip]")

    current = "screen_base"
    stack = [
        ("camera_base", camera_full_spans, "0:0"),
        ("camera_pip", camera_pip_spans, _position(config.camera_pip)),
        ("screen_pip", screen_pip_spans, _position(config.screen_pip)),
    ]
    step = 0
    for overlay, spans, position in stack:
        if not spans:
            continue
        step += 1
        label = f"comp{step}"
        lines.append(
            f"[{current}][{overlay}]overlay={position}:"
            f"enable='{_enable(spans)}':eof_action=pass[{label}]"
        )
        current = label

    if step == 0:
        return "[screen_base]", ";\n".join(lines)
    return f"[{current}]", ";\n".join(lines)


def _fan_out(source: str, pads: list[str]) -> list[str]:
    if len(pads) == 1:
        return [f"[{source}]null[{pads[0]}]"]
    return [f"[{source}]split={len(pads)}" + "".join(f"[{p}]" for p in pads)]


def _position(pip: PipConfig) -> str:
    return _CORNERS[pip.corner].format(m=pip.margin)


def _enable(spans: list[tuple[float, float]]) -> str:
    """`between()` terms summed together -- ffmpeg treats non-zero as true."""
    return "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in spans)
