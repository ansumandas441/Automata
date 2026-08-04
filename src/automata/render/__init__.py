"""Renderers: consumers that turn a timeline into something watchable."""

from .ffmpeg import RenderPlan, build

__all__ = ["RenderPlan", "build"]
