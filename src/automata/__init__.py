"""automata -- an orchestrator that edits multi-camera screencasts for you.

The orchestrator never touches pixels. It consumes timestamped perception
events (speech, gaze, source health) and emits a :class:`~automata.timeline.Timeline`
-- shot decisions plus cut spans. Rendering is a separate, replaceable
consumer, so the same decisions can drive a post-hoc ffmpeg render today and
live scene switching later.
"""

from .config import DirectorConfig, PipConfig, RenderConfig
from .director import Director
from .pipeline import plan
from .project import Project
from .timeline import Cut, Layout, Segment, Timeline

__all__ = [
    "Cut",
    "Director",
    "DirectorConfig",
    "Layout",
    "PipConfig",
    "Project",
    "RenderConfig",
    "Segment",
    "Timeline",
    "plan",
]

__version__ = "0.1.0"
