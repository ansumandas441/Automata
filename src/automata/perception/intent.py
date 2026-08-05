"""Turning speech into intent.

Taggers *abstain* by returning ``None`` rather than guessing. A pipeline tries
them in order and falls through to ``NEUTRAL``, so a tagger that is unsure --
or a remote one that times out -- can never stall or hijack the edit.

That abstention is what makes the language model affordable in step 2: the
keyword tagger below resolves the obvious majority of lines locally and for
free, and only genuinely ambiguous utterances are worth an API call.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..events import Intent, IntentKind, Utterance

PRODUCER = "intent"


class IntentTagger(Protocol):
    def tag(self, utterance: Utterance) -> Intent | None:
        """Classify, or return ``None`` to abstain."""


def _phrases(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


_FILLER = r"(?:\b(?:ok(?:ay)?|alright|right|um+|uh+|so|and|well|yeah)\b[\s,]*)*"

# "Cut" is a normal English verb -- "we cut the array here" must not delete a
# take. Only an imperative aimed at the editor counts, so the object is pinned
# to a deictic pronoun.
_CUT = _phrases(
    rf"^{_FILLER}(?:let'?s\s+|can\s+you\s+|please\s+)?cut\s+(?:that|this|it|the\s+last\s+(?:bit|part|take))\b",
    rf"^{_FILLER}(?:scratch|kill|drop|delete|redo)\s+(?:that|this|it)\b",
    rf"^{_FILLER}take\s+(?:two|three|2|3)\b",
    rf"^{_FILLER}(?:cut,?\s*){{2,}}",
)

# Widened against a real 201-sentence screencast transcript. People do not
# narrate the way test fixtures do: the idealised "let me show you this
# function" is rare, while "here you can see", "so you just click on it" and
# "I'm going to show you" carry the same meaning and are what actually gets
# said. Deixis plus a UI verb is the reliable shape.
_FOCUS_SCREEN = _phrases(
    r"\b(?:let me|lemme|i'?ll|i'?m going to|i'?m gonna|going to|gonna)\s+show you\b",
    r"\b(?:look|looking) at (?:this|that|the (?:code|screen|terminal|file|editor))\b",
    r"\bcheck (?:this|it|that) out\b",
    r"\b(?:here'?s|this is) the (?:code|file|function|error|output|part|bit)\b",
    r"\byou(?:'ll| will| can)? see\b",
    r"\b(?:click|clicking|tap|tapping|press|pressing) on\b",
    r"\bscroll (?:down|up|over|back|to)\b",
    r"\b(?:over|down|up|right|in|back) (?:here|there)\b",
    r"\bon (?:my|the) screen\b",
    r"\bif (?:we|you) (?:run|open|hit|look at|go to) (?:this|it|that)\b",
    r"\bthis (?:tab|button|panel|menu|window|option|setting|section|dropdown|field)\b",
    r"\bnotice (?:how|that|the)\b",
)

_FOCUS_CAMERA = _phrases(
    r"\bback to me\b",
    r"\blet me explain\b",
    r"\b(?:before|now) (?:we|i) (?:get into|dive in|start)\b",
    r"\bthe (?:idea|point|reason) (?:here )?is\b",
    r"\blet'?s talk about\b",
    r"\bstep back (?:for a|a) (?:second|moment|sec)\b",
)


class KeywordIntentTagger:
    """Local, free, deterministic. Handles the unambiguous majority."""

    def tag(self, utterance: Utterance) -> Intent | None:
        text = utterance.text.strip()
        if not text:
            return None

        for pattern, kind in (
            (_CUT, IntentKind.CUT_PREVIOUS),
            (_FOCUS_SCREEN, IntentKind.FOCUS_SCREEN),
            (_FOCUS_CAMERA, IntentKind.FOCUS_CAMERA),
        ):
            match = pattern.search(text)
            if match:
                return Intent(
                    t=utterance.t,
                    producer=PRODUCER,
                    kind=kind,
                    end=utterance.t_end,
                    confidence=0.9,
                    evidence=match.group(0).strip(),
                )
        return None


class IntentPipeline:
    """Runs taggers in order; first one that commits wins.

    Falls through to ``NEUTRAL`` rather than raising. An intent classifier that
    can block the pipeline is worse than one that is occasionally wrong.
    """

    def __init__(self, *taggers: IntentTagger) -> None:
        self._taggers = taggers or (KeywordIntentTagger(),)

    def tag(self, utterance: Utterance) -> Intent:
        for tagger in self._taggers:
            try:
                intent = tagger.tag(utterance)
            except Exception:  # noqa: BLE001 -- a sensor must never stop the edit
                continue
            if intent is not None:
                return intent
        return Intent(
            t=utterance.t,
            producer=PRODUCER,
            kind=IntentKind.NEUTRAL,
            end=utterance.t_end,
            confidence=1.0,
        )
