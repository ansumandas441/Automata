"""Footage in, events out.

Analysis is a separate command from planning on purpose. It is the only slow,
non-deterministic, network-touching part of the system: minutes of transcription
and face tracking, and whatever the classifier decides. Planning is
milliseconds and reproducible.

Splitting them means tuning `min_shot_s` is instant, a bad transcription can be
hand-corrected in the events file rather than re-run, and the expensive half
happens exactly once per recording.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .events import Event, IntentKind
from .perception import gaze as gaze_module
from .perception import stt as stt_module
from .perception.intent import IntentPipeline, KeywordIntentTagger
from .perception.llm import ClaudeIntentTagger, VerdictCache
from .perception.llm import available as llm_available
from .project import Project
from .sources import SourceRole


@dataclass
class AnalysisReport:
    utterances: int = 0
    gaze_samples: int = 0
    intents: int = 0
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"{self.utterances} utterances, {self.gaze_samples} gaze samples, "
            f"{self.intents} intents"
        ]
        out.extend(f"  {note}" for note in self.notes)
        return out


def analyze(
    project: Project,
    *,
    stt_config: stt_module.SttConfig | None = None,
    gaze_config: gaze_module.GazeConfig | None = None,
    use_llm: bool = True,
    cache_path=None,
) -> tuple[list[Event], AnalysisReport]:
    """Run every perception producer over a project's sources."""
    report = AnalysisReport()
    audio = project.audio_clip
    camera = project.source(SourceRole.CAMERA)

    # Whisper and MediaPipe both spend their time in native code, so running
    # them on two threads genuinely overlaps rather than fighting over the GIL.
    with ThreadPoolExecutor(max_workers=2) as pool:
        speech = pool.submit(stt_module.transcribe, audio, stt_config)
        looking = (
            pool.submit(gaze_module.track, camera, gaze_config) if camera is not None else None
        )
        utterances = speech.result()
        gaze_samples = looking.result() if looking is not None else []

    if camera is None:
        report.notes.append("no camera source; gaze is unavailable and speech decides alone")

    keywords = KeywordIntentTagger()
    taggers = [keywords]
    llm: ClaudeIntentTagger | None = None
    if use_llm and not llm_available():
        # An unconfigured classifier is an absence, not a failure: say so once
        # here rather than reporting a failed call per ambiguous utterance.
        report.notes.append(
            "classifier unavailable (no anthropic SDK or credentials); keywords only"
        )
    elif use_llm:
        llm = ClaudeIntentTagger(cache=VerdictCache(cache_path))
        # Warm concurrently first, so the ordered pass below is all cache hits.
        # Only what the free tagger could not resolve is worth a request.
        llm.warm(u for u in utterances if keywords.tag(u) is None)
        taggers.append(llm)

    pipeline = IntentPipeline(*taggers)
    intents = [pipeline.tag(u) for u in utterances]

    if llm is not None:
        llm.save()
        report.notes.append(f"classifier: {llm.stats.summary()}")
        if llm.stats.failed:
            report.notes.append(
                f"{llm.stats.failed} classifier calls failed and were treated as neutral"
            )

    report.utterances = len(utterances)
    report.gaze_samples = len(gaze_samples)
    report.intents = sum(1 for i in intents if i.kind != IntentKind.NEUTRAL)

    return [*utterances, *gaze_samples, *intents], report
