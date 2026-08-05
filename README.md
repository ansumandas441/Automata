# automata

Auto-edits multi-source screencasts. You record a camera, your screen and your
voice; it decides shot by shot whether the screen or your face should fill the
frame, and removes the takes you asked it to remove — from what you *said* and
where you *looked*, with no post-editing pass.

**Status: step 2 of 3.** The orchestration core and real perception both work end
to end — speech, gaze and intent are extracted from footage and rendered to a
cut, composited video. Live mode is still to come.

---

## Inspiration

Anyone who records coding videos knows the shape of the problem. You set up a
camera and a screen capture, you talk for forty minutes, and then you spend three
hours in an editor doing the same two things over and over: deciding whether the
screen or your face should fill the frame, and deleting the takes where you
fumbled a sentence and said "ugh, cut that."

Both decisions are already encoded in the recording. You *said* "let me show you
this function." You *looked* at the lens when you started talking to the viewer.
You literally said the word "cut." The information an editor needs is sitting in
the footage — it just isn't in a form software can act on.

So the idea wasn't "use AI to edit videos." It was narrower and more tractable:
**the creator is already directing; capture the direction.**

## What it does

You record camera, screen and audio separately — OBS, QuickTime, whatever you
already use. Automata then transcribes your speech with word-level timing,
tracks whether you're looking at the lens, decides the edit, and renders it as a
single `ffmpeg` command.

Say *"let me show you this function"* and the screen goes full-frame with your
camera in the corner. Look at the lens and talk to the viewer, and it flips. Say
*"cut that"* and the last couple of sentences — plus the phrase itself —
disappear from the finished video.

### The one design decision that matters

**The orchestrator never touches pixels.** It emits a *timeline* — an edit
decision list of shots and cut spans — and rendering is a separate consumer.

```
sources ─► perception ─► watermarked bus ─► director ─► timeline ─► renderer
           (events)      (time order)      (state machine)  (EDL)   (ffmpeg)
```

That buys three things:

- the same decisions drive a post-hoc ffmpeg render today and live OBS scene
  switching later, with no second brain;
- the director is unit-testable without a camera, a GPU or a person — see
  `tests/test_director.py`, which is the real specification of the product;
- when the edit is wrong you open a JSON file and fix one number, instead of
  re-recording.

## How to run it

### Install

```sh
git clone <this repo> && cd automata
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,perception,llm]'
```

`ffmpeg` must be on `PATH` (`brew install ffmpeg`).

Install in tiers if you only need part of it — the core has **zero runtime
dependencies**, so the director runs without a single model weight:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `[dev]` | pytest, ruff | the director, timeline, bus |
| `[perception]` | faster-whisper, mediapipe, av | `analyze` |
| `[llm]` | anthropic | LLM intent classification (optional — `--no-llm` works) |

On first gaze run a 3.6 MB face-landmarker model downloads to
`~/.cache/automata/`.

### The three commands

```sh
.venv/bin/automata analyze project.json           # footage  -> events.json
.venv/bin/automata plan    project.json           # events   -> timeline.json
.venv/bin/automata render  project.json -o out.mp4  # timeline -> video
```

| Command | Cost | Deterministic? | Run it |
| --- | --- | --- | --- |
| `analyze` | seconds–minutes | no | once per recording |
| `plan` | milliseconds | yes | every time you change a setting |
| `render` | minutes | yes | when you're happy with the timeline |

They're separate because `analyze` is the only slow, non-deterministic,
network-touching step. Tuning `min_shot_s` shouldn't mean re-running Whisper.

Useful flags: `analyze --no-llm` (no API calls), `analyze --force` (overwrite an
existing events file), `render --dry-run` (print the ffmpeg command, render
nothing), `render -t timeline.json` (render a specific, possibly hand-edited
timeline).

### The 30-second version — no models needed

`examples/demo/` ships a hand-written events file, so you can skip perception
entirely and go straight to the decisions:

```sh
examples/demo/make_media.sh                        # synthetic screen + camera clips
.venv/bin/automata plan   examples/demo/project.json
.venv/bin/automata render examples/demo/project.json -o examples/demo/out.mp4
```

That fixture exercises the interesting paths without any footage: speech
overriding gaze, a "cut that" that walks back over two bad takes but stops at a
natural pause, the word "cut" used innocently in narration (`we cut the array in
half`), a face leaving frame, and a camera that starts 0.4s late.

### The full pipeline — with real speech

`examples/live/` builds a recording with actual spoken audio (via macOS `say`),
so transcription, gaze and intent all run for real:

```sh
examples/live/make_media.sh
.venv/bin/automata analyze examples/live/project.json    # slow: transcribe + track gaze
.venv/bin/automata plan    examples/live/project.json    # fast: decide the edit
.venv/bin/automata render  examples/live/project.json -o examples/live/out.mp4
```

19.3s of footage → 14.2s output, in about 3s wall.

Add `--no-llm` to `analyze` for keyword intent tagging only — no API calls. With
neither the SDK nor credentials configured it degrades to that automatically and
says so.

> `analyze` refuses to overwrite an existing events file without `--force`.
> That file is meant to be hand-edited — fixing a mistranscribed line there is
> the intended workflow, and re-analysing would silently discard it.

### On your own recording

Record camera, screen and audio separately (OBS multi-track, QuickTime,
whatever you use), then write a `project.json` beside the files:

```json
{
  "sources": [
    {"id": "screen", "role": "screen", "path": "screen.mp4",
     "duration_s": 1800.0, "has_audio": true},
    {"id": "cam0",   "role": "camera", "path": "camera.mp4",
     "duration_s": 1800.0, "offset_s": 0.42}
  ],
  "audio_from": "screen",
  "director": {"min_shot_s": 2.5},
  "render": {"width": 1920, "height": 1080, "fps": 30}
}
```

Then the same three commands. Two fields do the real work:

**`duration_s`** is required — together the durations define the *editable
window*, the span where every source has footage. Get it from ffprobe:

```sh
ffprobe -v error -show_entries format=duration -of csv=p=0 screen.mp4
```

**`offset_s`** maps a clip's own time onto the master timeline
(`master = local + offset_s`). It is the only place wall-clock time is allowed
in, and getting it wrong lands every gaze event on the wrong words. Either clap
once on camera at the start and align the transients, or — if both files came
from one OBS recording — leave both at `0.0`.

Everything else is optional. `director` accepts any field from `DirectorConfig`
and `render` any field from `RenderConfig`; a typo tells you so rather than
silently defaulting.

### Running the tests

```sh
.venv/bin/python -m pytest -q          # 52 tests, no network, no footage
.venv/bin/python -m pytest -k cut -v   # just the cut behaviour
.venv/bin/ruff check .
```

`tests/test_director.py` is the real specification — the test names state the
product decisions and the docstrings say why.

**Open [pipeline.html](pipeline.html)** in a browser for an illustrated teardown of
the whole path — what each stage produces, what the numbers mean, and the two
damping constants caught in real gaze data. Every figure in it comes from one
nine-minute run.

**See [EXPERIMENTING.md](EXPERIMENTING.md)** for how to actually poke at this:
writing fixtures, sweeping the director's constants, calibrating gaze on your own
setup, and reading the timeline when an edit comes out wrong.

## How we built it

**The language model is a sensor, not the director.** This is the load-bearing
choice. Claude Haiku 4.5 labels one sentence into a four-value enum —
`FOCUS_SCREEN | FOCUS_CAMERA | CUT_PREVIOUS | NEUTRAL` — and stops there. Every
decision is made by fixed arbitration in `director.py`:

```
forced (a source is missing)  >  explicit speech  >  gaze  >  hold
```

damped by two constants that matter more than classifier accuracy ever will: no
shot shorter than `min_shot_s` (2.5s), and gaze must hold for `gaze_debounce_s`
(0.8s) before it may move the camera. Without them, a creator glancing between
screen and lens produces a strobing, unwatchable edit.

Keeping the model out of the director makes the system roughly fifty times
cheaper, reproducible, and debuggable — when a cut is wrong you can name the rule
that made it.

**Cost is controlled in three layers.** A free local regex tagger resolves the
unambiguous majority of lines. A keyword gate skips plain narration entirely.
Only what survives both is worth an API call, and every verdict is cached to disk
keyed by `(model, prompt version, text)` — which exists less for cost than for
reproducibility. Without it, re-running analysis on the same footage could
produce a *different edit*. Taggers *abstain* rather than guess, so one that is
unsure — or that times out — can never stall or hijack the edit.

**Analysis is separated from planning.** Transcription and face tracking are
slow, non-deterministic and network-touching; deciding the edit is milliseconds
and pure. Splitting them into two commands means tuning `min_shot_s` doesn't mean
re-running Whisper, and a mistranscribed line is fixable in a text editor.

**The filter graph is fixed-size.** Compositing timing lives in ffmpeg `enable=`
expressions rather than one filter chain per shot, so a 40-minute video with 300
shot changes builds the same graph as a 40-second one.

The core has **zero runtime dependencies** — the director can be exercised
without a single model weight.

## Challenges we ran into

Every real problem here was found by running the thing, not by thinking about it.

**Sync is not "match the timestamps."** Wall-clock time is only allowed in once,
to derive each source's `offset_s`. It is never used for ongoing timing: NTP can
step the clock mid-recording, and 45 minutes at a nominal 30fps that is really
29.97 drifts by seconds. Everything downstream keys off PTS, never frame index —
dropped screen-capture frames silently desync anything that counts frames, and
macOS ScreenCaptureKit is variable-frame-rate.

**A silent desync that passed every test.** The cut logic assumed the timeline
started at zero. With a camera that started 0.4s late, the renderer trimmed
camera to 14.48s and screen to 14.88s — the concat would have drifted apart,
silently, and worse the further into the video you got. No unit test would have
found it; only generating the real ffmpeg filter graph and reading the trim
values did.

**Out-of-order arrival.** Producers run at wildly different cadences, and an
intent verdict about t=10.0 arrives at t=11.4. Consuming events in arrival order
is exactly how an auto-editor cuts the wrong sentence. `bus.py` buffers on
*event* time behind a watermark and only releases what is settled; the director
asserts the ordering rather than trusting it.

**`UNKNOWN` is not `AWAY`.** No face detected means the creator leaned out of
frame or the light dropped. Treating that as "looked away" would swing the camera
every time they reach for coffee, so unknown holds the current shot.

**Whisper segments are not sentences.** The first run on real speech returned
*"No wait, that is wrong. Cut that. Alright, back to me for a second."* as a
single eight-second utterance. That breaks the anchored cut patterns, and cutting
that utterance would delete the good closing line. Fixed by re-splitting on
sentence punctuation and inter-word pauses using word timings we already had.

**The cut walked back too far, and our first fix was wrong.** With sentences
fixed, a "cut that" reached back and swallowed the good opening line. The cause
was that *no pause in real speech cleared the 0.6s threshold* — actual
inter-sentence gaps measured 0.48–0.58s — so the backward walk never stopped. In
a continuous monologue, one phrase would have deleted the full twenty-second
bound.

Our first fix snapped to the widest gap found. We threw it away: choosing between
0.58s and 0.48s is deciding on noise. The right bound turned out to be simpler
and more honest — **two sentences**, because that is what "cut that" means
anyway.

**An API had moved underneath us.** MediaPipe shipped 1.0 and deleted the
`solutions` API our gaze detector was written against; it didn't even import. The
replacement Tasks API turned out better: it exposes a real head-pose matrix and
`eyeLook*` blendshapes. That second signal matters — someone whose head is square
to the camera while their eyes track a second monitor is the common case, and our
original landmark heuristic would have confidently called that "looking at the
lens."

**Reading the docs beat trusting our instincts.** We were about to add prompt
caching to the classifier. Haiku 4.5's minimum cacheable prefix is 4096 tokens
and our system prompt is a fraction of that, so the marker would have silently
done nothing while charging the write premium. There's now a test asserting it
stays absent.

## Accomplishments that we're proud of

**We proved the edit actually happened, in the output.** The strongest check
isn't a passing test — it's running the *rendered* video back through
transcription. The timeline claimed three things were deleted, and the finished
audio says only:

> "Okay, so let me show you this function down here." … "Alright, back to me for
> a second."

The bad takes and the `Cut that.` trigger are physically gone. Frames from either
side of the switch confirm the layout flip, driven purely by a phrase Whisper
transcribed and the tagger labeled.

**"Cut" as a verb doesn't delete your footage.** `"we cut the array in half right
here"` is narration; `"cut that"` is an instruction. The imperative patterns pin
the object to a deictic pronoun so ordinary speech can't destroy a take — and
there's a test for it, because this is the failure mode that would make the tool
untrustworthy.

**Hysteresis, demonstrated rather than asserted.** On a fixture of a creator
glancing back and forth every 1.2 seconds:

```
min_shot_s=  0.0 -> 26 shots      <- unwatchable
min_shot_s=  2.5 ->  9 shots
min_shot_s= 10.0 ->  3 shots      <- sluggish
```

**Failure modes are designed, not discovered.** A classifier that times out,
refuses, returns malformed JSON, or isn't configured at all fails open to neutral
and says so in one line. The camera dropping out mid-recording degrades to
screen-only rather than crashing. Most of the 49 tests are failure paths.

**It's fast enough to iterate on.** 19.3 seconds of footage analyzed in about 3
seconds wall — transcription and gaze tracking overlapping on two threads, with
speech-to-text at roughly 7× realtime.

## What we learned

**Separate the brain from the hands, early.** Emitting a timeline instead of
pixels was the decision everything else hung on. It made the director testable
without hardware, made a wrong edit a one-line JSON fix, and means live mode is
the same code behind a delay buffer rather than a second implementation.

**Constrain the model's job until it's boring.** A four-value enum, schema-
constrained by the API rather than by parsing, temperature zero, cached to disk.
The exciting version — let the model decide the edit — would have been
non-reproducible, expensive and impossible to debug.

**Prompt injection has a structural answer.** The classifier's input is whatever
the creator said out loud, and someone reading a document aloud can utter an
instruction. Because the output is schema-constrained to four labels, the worst a
malicious phrase can do is mislabel one sentence — which hysteresis then absorbs.

**Real speech doesn't look like your fixtures.** We assumed sentence boundaries
and clean pauses. Real inter-sentence gaps sit around half a second and vary by
less than a tenth, which invalidated a threshold we'd have sworn was
conservative. Synthesizing actual audio with `say` and running it end to end
found in one afternoon what fixtures alone never would have.

**Generate the artifact, don't just model it.** The desync bug lived in code that
passed every test. It surfaced only when we printed the ffmpeg command and read
the numbers.

**When erring, err recoverable.** Cutting too little leaves an easy manual fix;
cutting too much destroys footage. That asymmetry decided the cut algorithm, and
it's the right default for any tool that edits someone's work.

## What's next for Automata

- [x] **1 — Core.** Clock and offset model, event types, watermarked bus,
  director, timeline algebra, ffmpeg renderer, CLI. Zero runtime dependencies.
- [x] **2 — Real perception.** VAD-gated `faster-whisper` split into sentences on
  word timings; MediaPipe Tasks gaze at 8Hz using head pose and eye-look
  blendshapes; Haiku 4.5 fallback behind `IntentTagger`, schema-constrained to
  four labels, at temperature 0 with an on-disk verdict cache.
- [ ] **3 — Live mode.** Same director behind a delay buffer, emitting OBS scene
  switches instead of an EDL.

**Calibrate gaze against real faces.** This is the honest gap. The decode path,
PTS timing, sampling and unknown-state handling are all exercised — but the yaw,
pitch and eye-offset *thresholds* have never seen an actual person. They depend
on camera position and how far you sit from the screen. The next step is an
`automata calibrate` command: look at the lens for five seconds, look at your
screen for five seconds, and let it fit the thresholds to your setup.

**Screen activity as a tiebreaker.** Frame-diff energy on the screen capture is
nearly free and answers "is something happening on screen right now?" The event
type already flows through the bus; the director's handler is a documented no-op.
It's the obvious signal when speech and gaze both abstain.

**A second camera.** Deliberately deferred but not designed out: sources are
keyed by role and the renderer indexes by role, so a second camera is a config
change plus one new `Layout` member.

**A timeline editor.** The EDL is already hand-editable JSON, which is the right
substrate for a scrubber that shows shot boundaries and cuts and lets you drag
them. The architecture was chosen partly to make this possible.

**Better cut vocabulary.** Right now "cut that" removes the last two sentences.
"Cut the last thirty seconds," "cut back to where I started the demo" and
"actually keep that" are all things creators say, and all things the timeline
model can already express.

## Layout

| file | role |
| --- | --- |
| `sources.py` | clips, offsets, the editable overlap window |
| `events.py` | timestamped perception events |
| `bus.py` | watermarked reordering, bounded queues, drop policy |
| `director.py` | the state machine — all decisions live here |
| `timeline.py` | the EDL and its interval algebra |
| `pipeline.py` | wiring; post-hoc is the degenerate case of live |
| `analyze.py` | footage to events: the slow, non-deterministic half |
| `perception/stt.py` | VAD-gated transcription, split into sentences |
| `perception/gaze.py` | head pose + eye-look at 8Hz, PTS-timed |
| `perception/intent.py` | the free keyword tagger and the tagger chain |
| `perception/llm.py` | escalation to Haiku 4.5, gated, cached, fail-open |
| `render/ffmpeg.py` | timeline to one ffmpeg invocation |
