"""Speech to text, gated on voice activity.

The VAD gate is not an optimisation. Whisper hallucinates confidently over
silence -- "Thank you.", "Thanks for watching!", "Subscribe to the channel" are
its house favourites, and a screencast has long quiet stretches while the
creator types. Ungated, those phantom utterances reach the intent classifier and
can move the camera or delete a take during a silence.

So the output passes three filters: the model's own VAD, its per-segment
`no_speech_prob`, and a denylist of known hallucinations. Anything the model is
unsure about is dropped rather than guessed at -- a missed utterance costs one
shot decision; a phantom one costs a wrong cut.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..events import Utterance, Word
from ..sources import SourceClip

PRODUCER = "stt"

HALLUCINATIONS = frozenset(
    {
        "thank you.",
        "thank you",
        "thanks for watching!",
        "thanks for watching.",
        "thank you for watching.",
        "please subscribe",
        "subscribe to the channel",
        "you",
        ".",
        "bye.",
        "okay.",
    }
)
"""Whisper's canonical silence outputs. Matched on the whole utterance only --
"thank you" inside a real sentence is real speech."""


@dataclass(frozen=True)
class SttConfig:
    model_size: str = "small.en"
    """Comfortably faster than real time on Apple Silicon, and accurate enough
    that the keyword tagger's patterns still match."""

    device: str = "auto"
    compute_type: str = "int8"
    language: str | None = "en"

    max_no_speech_prob: float = 0.6
    min_avg_logprob: float = -1.0
    min_duration_s: float = 0.15

    vad_min_silence_ms: int = 400
    vad_speech_pad_ms: int = 120

    split_sentences: bool = True
    """Break each segment at sentence boundaries.

    Whisper segments are bounded by breath and prosody, not by sentences: a
    single one routinely reads "No wait, that is wrong. Cut that. Alright, back
    to me." That is wrong twice over -- the anchored cut patterns never match
    mid-segment, and if they did, the cut would take the good sentence after it
    along with the bad ones. Splitting restores the unit the rest of the system
    assumes, and the word timings make the new boundaries exact.
    """

    sentence_gap_s: float = 0.35
    """A pause between words this long also ends a sentence, punctuation or not.
    Transcripts of speech are unreliably punctuated; silence is not."""


def transcribe(clip: SourceClip, config: SttConfig | None = None) -> list[Utterance]:
    """Transcribe a clip and return utterances on the *master* timeline."""
    config = config or SttConfig()
    model = _load(config)

    segments, _info = model.transcribe(
        clip.path,
        language=config.language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": config.vad_min_silence_ms,
            "speech_pad_ms": config.vad_speech_pad_ms,
        },
        # Each utterance is classified on its own, so carrying decoded text
        # forward buys nothing -- and it is what lets Whisper fall into a
        # repetition loop after one bad segment.
        condition_on_previous_text=False,
    )

    utterances: list[Utterance] = []
    for segment in segments:
        if not _plausible(segment, config):
            continue
        words = tuple(
            Word(w.word.strip(), clip.to_master(w.start), clip.to_master(w.end))
            for w in (segment.words or ())
            if w.word.strip()
        )
        for piece in _split(segment, words, clip, config):
            if piece.t_end - piece.t >= config.min_duration_s:
                utterances.append(piece)
    return utterances


_SENTENCE_END = ".!?"


def _split(segment, words, clip, config: SttConfig) -> list[Utterance]:
    """One utterance per sentence, timed from the words that compose it."""
    whole = Utterance(
        t=clip.to_master(segment.start),
        producer=PRODUCER,
        text=segment.text.strip(),
        end=clip.to_master(segment.end),
        words=words,
    )
    if not config.split_sentences or len(words) < 2:
        return [whole]

    pieces: list[Utterance] = []
    current: list[Word] = []
    for index, word in enumerate(words):
        current.append(word)
        following = words[index + 1] if index + 1 < len(words) else None
        ends_sentence = word.text.rstrip("\"')]").endswith(tuple(_SENTENCE_END))
        pauses = following is not None and following.t - word.t_end >= config.sentence_gap_s
        if following is None or ends_sentence or pauses:
            pieces.append(_utterance(current))
            current = []

    return pieces or [whole]


def _utterance(words: list[Word]) -> Utterance:
    return Utterance(
        t=words[0].t,
        producer=PRODUCER,
        text=" ".join(w.text for w in words),
        end=words[-1].t_end,
        words=tuple(words),
    )


def _plausible(segment, config: SttConfig) -> bool:
    text = segment.text.strip()
    if not text or text.lower() in HALLUCINATIONS:
        return False
    if segment.end - segment.start < config.min_duration_s:
        return False
    if getattr(segment, "no_speech_prob", 0.0) > config.max_no_speech_prob:
        return False
    return getattr(segment, "avg_logprob", 0.0) >= config.min_avg_logprob


def _load(config: SttConfig):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover -- optional dependency
        raise RuntimeError(
            "speech-to-text needs faster-whisper: pip install 'automata[perception]'"
        ) from exc
    return WhisperModel(config.model_size, device=config.device, compute_type=config.compute_type)
