"""A project file: which recordings, how they line up, how to edit them.

Sources are ingested, never captured. The creator records with OBS multi-track
or QuickTime and gives us the files plus a sync anchor; we stay out of the
device, permission and driver business entirely.

The ``offset_s`` on each source is the one place wall-clock time is allowed in.
Derive it once -- from a clap, from OBS's shared start timestamp, from file
mtimes if you must -- and everything downstream is monotonic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import DirectorConfig, RenderConfig
from .sources import SourceClip, SourceRole
from .timeline import Layout


@dataclass(frozen=True)
class Project:
    root: Path
    sources: tuple[SourceClip, ...]
    audio_from: str
    events_path: Path | None = None
    director: DirectorConfig = field(default_factory=DirectorConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def source(self, role: SourceRole) -> SourceClip | None:
        return next((s for s in self.sources if s.role is role), None)

    def require(self, role: SourceRole) -> SourceClip:
        clip = self.source(role)
        if clip is None:
            raise ValueError(f"project has no {role.value} source")
        return clip

    @property
    def audio_clip(self) -> SourceClip:
        clip = next((s for s in self.sources if s.id == self.audio_from), None)
        if clip is None:
            raise ValueError(f"audio_from={self.audio_from!r} matches no source")
        return clip

    @classmethod
    def load(cls, path: str | Path) -> Project:
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        root = path.parent

        sources = tuple(
            SourceClip(
                id=raw["id"],
                role=SourceRole(raw["role"]),
                path=str((root / raw["path"]).resolve()),
                offset_s=float(raw.get("offset_s", 0.0)),
                duration_s=_optional_float(raw.get("duration_s")),
                has_audio=bool(raw.get("has_audio", False)),
            )
            for raw in data["sources"]
        )
        if not sources:
            raise ValueError("project has no sources")
        _reject_duplicates(sources)

        audio_from = data.get("audio_from") or next(
            (s.id for s in sources if s.has_audio), sources[0].id
        )
        events = data.get("events")

        return cls(
            root=root,
            sources=sources,
            audio_from=audio_from,
            events_path=(root / events).resolve() if events else None,
            director=_director_config(data.get("director", {})),
            render=_render_config(data.get("render", {})),
        )

    def with_director(self, **overrides) -> Project:
        return replace(self, director=replace(self.director, **overrides))


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _reject_duplicates(sources: tuple[SourceClip, ...]) -> None:
    seen: set[str] = set()
    for clip in sources:
        if clip.id in seen:
            raise ValueError(f"duplicate source id {clip.id!r}")
        seen.add(clip.id)
    # One camera for now. The layout enum and renderer already index by role,
    # so a second camera is a config change plus a new Layout member.
    for role in SourceRole:
        count = sum(1 for s in sources if s.role is role)
        if count > 1:
            raise ValueError(f"multiple {role.value} sources are not supported yet")


def _director_config(raw: dict) -> DirectorConfig:
    if "default_layout" in raw:
        raw = {**raw, "default_layout": Layout(raw["default_layout"])}
    return DirectorConfig(**raw)


def _render_config(raw: dict) -> RenderConfig:
    from .config import PipConfig

    raw = dict(raw)
    for key in ("camera_pip", "screen_pip"):
        if key in raw:
            raw[key] = PipConfig(**raw[key])
    return RenderConfig(**raw)
