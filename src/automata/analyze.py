"""Footage in, events out.

Analysis is a separate command from planning on purpose. It is the only slow,
non-deterministic, network-touching part of the system: minutes of transcription
and face tracking, and whatever the classifier decides. Planning is
milliseconds and reproducible.

Splitting them means tuning `min_shot_s` is instant, a bad transcription can be
hand-corrected in the events file rather than re-run, and the expensive half
happens exactly once per recording.

Gaze is identical in every mode -- only the reading of the transcript changes.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .events import Event, Intent, IntentKind
from .perception import gaze as gaze_module
from .perception import gemini as gemini_module
from .perception import stt as stt_module
from .perception.intent import IntentPipeline, KeywordIntentTagger
from .perception.llm import ClaudeIntentTagger, VerdictCache
from .perception.llm import available as claude_available
from .project import Project
from .sources import SourceRole

MODES = ("auto", "offline", "claude", "gemini")

CUT_DEDUPE_S = 2.0
"""How close two cut requests must be to count as the same one."""


@dataclass
class AnalysisReport:
    mode: str = "offline"
    utterances: int = 0
    gaze_samples: int = 0
    intents: int = 0
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"{self.utterances} utterances, {self.gaze_samples} gaze samples, "
            f"{self.intents} intents  [{self.mode}]"
        ]
        out.extend(f"  {note}" for note in self.notes)
        return out


def resolve_mode(requested: str | None = None) -> str:
    """Pick the intent mode, falling back to what is actually configured.

    Order: the explicit argument, then ``AUTOMATA_INTENT``, then whichever
    remote path has credentials, then offline. Nothing here ever fails -- an
    unconfigured AI mode degrades to the deterministic taggers and says so.
    """
    mode = (requested or os.environ.get("AUTOMATA_INTENT") or "auto").lower()
    if mode not in MODES:
        raise ValueError(f"intent mode must be one of {', '.join(MODES)}; got {mode!r}")
    if mode != "auto":
        return mode
    if gemini_module.available():
        return "gemini"
    if claude_available():
        return "claude"
    return "offline"


def analyze(
    project: Project,
    *,
    mode: str | None = None,
    stt_config: stt_module.SttConfig | None = None,
    gaze_config: gaze_module.GazeConfig | None = None,
    cache_path=None,
) -> tuple[list[Event], AnalysisReport]:
    """Run every perception producer over a project's sources."""
    mode = resolve_mode(mode)
    report = AnalysisReport(mode=mode)
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

    intents = _read_transcript(utterances, mode, cache_path, report)

    report.utterances = len(utterances)
    report.gaze_samples = len(gaze_samples)
    report.intents = sum(1 for i in intents if i.kind != IntentKind.NEUTRAL)

    return [*utterances, *gaze_samples, *intents], report


def _read_transcript(utterances, mode: str, cache_path, report: AnalysisReport) -> list[Intent]:
    """Turn speech into intent events, by whichever route was selected."""
    keywords = KeywordIntentTagger()

    if mode == "gemini":
        return _segment_with_gemini(utterances, keywords, cache_path, report)

    taggers = [keywords]
    if mode == "claude":
        if not claude_available():
            report.notes.append(
                "claude requested but not configured (no SDK or credentials); keywords only"
            )
        else:
            llm = ClaudeIntentTagger(cache=VerdictCache(cache_path))
            # Warm concurrently first, so the ordered pass below is all cache
            # hits. Only what the free tagger could not resolve is worth a call.
            llm.warm(u for u in utterances if keywords.tag(u) is None)
            taggers.append(llm)
            report.notes.append(f"classifier: {llm.stats.summary()}")

    pipeline = IntentPipeline(*taggers)
    intents = [pipeline.tag(u) for u in utterances]
    if taggers[-1] is not keywords:
        taggers[-1].save()
    return intents


def _segment_with_gemini(
    utterances, keywords: KeywordIntentTagger, cache_path, report: AnalysisReport
) -> list[Intent]:
    """Read the transcript as a narrative and keep only the state changes.

    The deterministic cut detector still runs alongside. Cuts destroy footage,
    so the precise anchored patterns stay in the loop as a safety net rather
    than handing that one decision entirely to a model.
    """
    if not gemini_module.available():
        report.notes.append(
            "gemini requested but not configured (set GOOGLE_CLOUD_PROJECT and "
            "GOOGLE_APPLICATION_CREDENTIALS); keywords only"
        )
        return [IntentPipeline(keywords).tag(u) for u in utterances]

    namespace = f"{gemini_module.MODEL}\x00{gemini_module.PROMPT_VERSION}"
    segmenter = gemini_module.GeminiSegmenter(
        cache=VerdictCache(cache_path, namespace=namespace)
    )
    intents = segmenter.segment(list(utterances))
    segmenter.save()
    report.notes.append(f"segmenter: {segmenter.stats.summary()}")

    if segmenter.stats.failed and not intents:
        report.notes.append("gemini returned nothing usable; falling back to keywords")
        return [IntentPipeline(keywords).tag(u) for u in utterances]

    merged = _merge_cuts(intents, utterances, keywords, report)
    return merged


def _merge_cuts(intents, utterances, keywords, report: AnalysisReport) -> list[Intent]:
    known = [i.t for i in intents if i.kind == IntentKind.CUT_PREVIOUS]
    added = 0
    for utterance in utterances:
        found = keywords.tag(utterance)
        if found is None or found.kind != IntentKind.CUT_PREVIOUS:
            continue
        if any(abs(found.t - t) <= CUT_DEDUPE_S for t in known):
            continue
        intents.append(found)
        known.append(found.t)
        added += 1
    if added:
        report.notes.append(f"{added} cut(s) found by keyword that the model missed")
        intents.sort(key=lambda i: (i.t, i.end))
    return intents
