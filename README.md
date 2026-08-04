# automata

Auto-edits multi-source screencasts. You record a camera, your screen and your
voice; it decides shot by shot whether the screen or your face should fill the
frame, and removes the takes you asked it to remove -- from what you *said* and
where you *looked*, with no post-editing pass.

**Status: step 2 of 3.** The orchestration core and real perception both work
end to end — speech, gaze and intent are extracted from footage and rendered to
a cut, composited video. Live mode is still to come.

## The one design decision that matters

**The orchestrator never touches pixels.** It emits a *timeline* -- shot
decisions plus cut spans -- and rendering is a separate consumer.

```
sources ─► perception ─► watermarked bus ─► director ─► timeline ─► renderer
           (events)      (time order)      (state machine)  (EDL)   (ffmpeg)
```

That buys three things:

- the same decisions drive a post-hoc ffmpeg render today and live OBS scene
  switching later, with no second brain;
- the director is unit-testable without a camera, a GPU or a person -- see
  `tests/test_director.py`, which is the real specification of the product;
- when the edit is wrong you open a JSON file and fix one number, instead of
  re-recording.

## The language model is a sensor, not the director

The classifier labels one utterance into
`FOCUS_SCREEN | FOCUS_CAMERA | CUT_PREVIOUS | NEUTRAL` and stops there. Every
decision is made by fixed arbitration in `director.py`:

```
forced (a source is missing)  >  explicit speech  >  gaze  >  hold
```

damped by two constants that matter more than classifier accuracy ever will:
no shot shorter than `min_shot_s` (2.5s), and gaze must hold for
`gaze_debounce_s` (0.8s) before it may move the camera. Without them, a creator
glancing between screen and lens produces a strobing, unwatchable edit.

Cost follows from the same split: `KeywordIntentTagger` resolves the
unambiguous majority locally and for free, and taggers *abstain* rather than
guess, so only genuinely ambiguous lines are worth an API call in step 2.

## Try it

```sh
pip install -e '.[dev,perception,llm]'
examples/demo/make_media.sh                      # synthetic screen + camera clips
automata analyze examples/demo/project.json      # slow: transcribe + track gaze
automata plan    examples/demo/project.json      # fast: decide the edit
automata render  examples/demo/project.json -o examples/demo/out.mp4
```

`analyze` is separated from `plan` because it is the only slow,
non-deterministic, network-touching step. Tuning `min_shot_s` should not mean
re-running Whisper, and a mistranscribed line should be fixable in a text editor
rather than by re-recording.

Add `--no-llm` to `analyze` for keyword intent tagging only — no API calls. With
neither the SDK nor credentials configured it degrades to that automatically and
says so.

The bundled fixture exercises the interesting paths without any footage: speech
overriding gaze, a "cut that" that walks back over two bad takes but stops at a
natural pause, the word "cut" used innocently in narration (`we cut the array in
half`), a face leaving frame, and a camera that starts 0.4s late.

`examples/live/` goes further and builds a recording with **real speech** (via
macOS `say`), so the full perception path runs without a webcam.

**See [EXPERIMENTING.md](EXPERIMENTING.md)** for how to actually poke at this:
writing fixtures, sweeping the director's constants, calibrating gaze on your
own setup, and reading the timeline when an edit comes out wrong.

## What the hard parts actually were

**Sync.** Wall-clock time is only allowed in once, to derive each source's
`offset_s`. It is never used for ongoing timing: NTP can step the clock
mid-recording, and 45 minutes at a nominal 30fps that is really 29.97 drifts by
seconds. Everything downstream keys off PTS, never frame index -- dropped
screen-capture frames silently desync anything that counts frames, and macOS
ScreenCaptureKit is variable-frame-rate.

**Out-of-order arrival.** Producers run at wildly different cadences, and an
intent verdict about t=10.0 arrives at t=11.4. Consuming events in arrival
order is exactly how an auto-editor cuts the wrong sentence. `bus.py` buffers
on *event* time behind a watermark and only releases what is settled; the
director asserts the ordering rather than trusting it.

**`UNKNOWN` is not `AWAY`.** No face detected means the creator leaned out of
frame or the light dropped. Treating that as "looked away" would swing the
camera every time they reach for coffee, so unknown holds the current shot.

**Cuts are nastier than they look.** A cut must delete the trigger phrase
itself, snap back to a real pause rather than a fixed number of seconds, merge
with overlapping cuts (repeated "cut, cut, cut" is normal), and re-split any
shot it straddles. And "we cut the array here" is narration -- the imperative
patterns pin the object to a deictic pronoun so ordinary speech can't delete a
take.

**Fixed-size filter graph.** Compositing timing lives in ffmpeg `enable=`
expressions, not one filter chain per shot. A 40-minute video with 300 shot
changes builds the same graph as a 40-second one.

**Whisper segments are not sentences.** They are bounded by breath and prosody,
so one routinely reads *"No wait, that is wrong. Cut that. Alright, back to
me."* — which breaks the anchored cut patterns, and would delete the good
sentence along with the bad ones. Segments are re-split on sentence punctuation
and inter-word pauses, using the word timings already collected.

**Two sentences is the real cut bound, not silence.** Inter-sentence pauses in
fluent speech run about half a second and vary by less than a tenth, so almost
none clear a silence threshold. Reaching back until one does means a single
"cut that" deletes twenty seconds; picking the widest of several near-identical
gaps is deciding on noise. So the reach is capped at two sentences — which is
what the phrase means anyway.

## Roadmap

- [x] **1 -- Core.** Clock and offset model, event types, watermarked bus,
  director, timeline algebra, ffmpeg renderer, CLI. Zero runtime dependencies.
- [x] **2 -- Real perception.** VAD-gated `faster-whisper` split into sentences
  on word timings; MediaPipe Tasks gaze at 8Hz using head pose and eye-look
  blendshapes; Haiku 4.5 fallback behind `IntentTagger`, schema-constrained to
  four labels, at temperature 0 with an on-disk verdict cache.
- [ ] **3 -- Live mode.** Same director behind a delay buffer, emitting OBS
  scene switches instead of an EDL.

Multi-camera is deliberately deferred but not designed out: sources are keyed
by role and the renderer indexes by role, so a second camera is a config change
plus one new `Layout` member.

## Layout

| file | role |
| --- | --- |
| `sources.py` | clips, offsets, the editable overlap window |
| `events.py` | timestamped perception events |
| `bus.py` | watermarked reordering, bounded queues, drop policy |
| `director.py` | the state machine -- all decisions live here |
| `timeline.py` | the EDL and its interval algebra |
| `pipeline.py` | wiring; post-hoc is the degenerate case of live |
| `analyze.py` | footage to events: the slow, non-deterministic half |
| `perception/stt.py` | VAD-gated transcription, split into sentences |
| `perception/gaze.py` | head pose + eye-look at 8Hz, PTS-timed |
| `perception/intent.py` | the free keyword tagger and the tagger chain |
| `perception/llm.py` | escalation to Haiku 4.5, gated, cached, fail-open |
| `render/ffmpeg.py` | timeline to one ffmpeg invocation |
