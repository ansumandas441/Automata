"""Perception producers: things that turn footage into timestamped events.

Step 1 ships a fixture replayer and a keyword intent tagger. Step 2 adds
VAD-gated speech-to-text and a gaze detector behind these same interfaces; the
director never learns which one it is talking to.
"""

from .fixture import load_events
from .intent import IntentPipeline, IntentTagger, KeywordIntentTagger

__all__ = ["IntentPipeline", "IntentTagger", "KeywordIntentTagger", "load_events"]
