# Walkthrough — how this system works and why

Written to be read cold, months later, before an interview. `DECISIONS.md` is
the terse log of choices; this is the explanation underneath them.

Grows as the project does. Day 1 content is complete.

---

## Part 1 — Why the system has this shape

Everything follows from one constraint: **you cannot search audio.** There is
no way to look at a waveform and find a phone number. So the only route is:

```
audio → text → find PII in text → map back to audio → destroy those samples
```

Every hard problem here is a consequence of that detour.

**Consequence A — two coordinate systems.** The detector finds PII at
*character positions* in a string. The redactor must destroy *array indices*
in a sample buffer. Different universes. Something must translate, and that
translation is the most error-prone part of the system.

**Consequence B — invisible error propagation.** The detector only ever sees
the ASR's output. If Whisper mishears "Mathew" as "Mustard", the detector
never sees a name, nothing is beeped, and the name plays in the clear. The
first model's mistakes are *invisible* to the second.

The architecture is organised around those two facts: keep the coordinate
systems strictly separate, bridge them in exactly one function, and make it
trivial to swap the text the detector sees — so error propagation is
measurable.

---

## Part 2 — What audio actually is here

Sound is continuously varying pressure. To store it you measure that pressure
at a fixed rate; each measurement is a **sample**. We use **16,000 samples per
second**.

Why 16 kHz? Nyquist: to represent frequencies up to *f* you must sample at
2*f*. Speech energy is mostly below 8 kHz, so 16 kHz covers it. (Music uses
44.1 kHz because instruments go higher.) Whisper was trained at 16 kHz and
requires it — a hard input requirement, not a preference.

Each sample is a 16-bit signed integer, −32768 to +32767. In `audio.py`:

```python
np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
```

`<i2` = little-endian 2-byte signed int. Dividing by 32767 gives **float32 in
[−1, 1]**, which is what every audio library and model expects and what does
not overflow under arithmetic. We convert back to int16 only when writing a
file.

**The one equation that matters:**

```
sample_index = time_in_seconds × 16000
```

That is the whole bridge between "2.5 seconds in" and "array position 40,000".
A 30.74 s dialogue is an array of 491,840 float32 values. "Destroying" audio
is assigning over a slice: `out[198400:238400] = beep(40000)`.

---

## Part 3 — The ground truth problem, and the trick

To measure anything you need the right answer. For audio PII that is two
things: **what** the PII is, and **when** it was spoken. No such dataset
exists for Indian conversational speech, so we generate one.

The text is easy — we write the script, so we know where the PII is. The
**timings** are the hard part. The standard tool is **forced alignment**: a
model takes known text plus audio and works out which frames correspond to
which words.

We rejected it for two reasons. It is another model and install. More
importantly the proposal also wanted alignment as a *pipeline stage*
(sharpening Whisper's timings) — so the same component would be both the thing
under test and the definition of truth. The timing metric would partly measure
the aligner against itself. Circular.

**The trick:** author a dialogue as alternating segments, render each one
separately.

```json
{"text": "It is"},
{"text": "nine eight seven six five four three two one zero", "pii": "PHONE"}
```

Segment 1 renders to 0.84 s. Segment 2 renders to 2.46 s. Concatenate. The
phone number starts at 0.84 s and ends at 3.30 s — **exactly, by
construction.** No alignment, no model, no circularity.

`tts.py:render_dialogue` is a loop maintaining a counter `n_samples`; when it
meets a segment carrying a `"pii"` key it records the current offset as the
start time.

**The cost, declared honestly:** `say` gives each segment utterance-final
prosody, so there is an unnatural seam at every boundary. A pause at a word
boundary is a cue real speech would not give the ASR, so our numbers are
*optimistic*. The human-recorded subset exists to measure that gap.

### The silence detail

`say` pads every utterance with silence — measured at **210 ms on a 2.4 s
utterance, ~9%**. Left in, a gold span would claim the PII starts before it is
audible, and every coverage metric would be scored against boundaries that are
partly silence.

`audio.py:trim_silence` finds the first and last sample exceeding 2% of the
clip's peak and slices there, keeping 15 ms so consonant attacks are not
clipped. Trim first, *then* concatenate.

### Ground truth in both systems

```python
start: float      # seconds into audio      → scoring whether the beep landed
char_start: int   # offset into transcript  → scoring the detector on text
```

Both are needed, for different metrics. Character offsets let you score
detection without involving audio at all — which is what makes the
error-attribution experiment cheap.

---

## Part 4 — The pipeline, module by module

### `taxonomy.py` — the ontology

PII is a tree, not a flat label set. Eight leaves, three branches:

```
PERSON     ← GIVEN_NAME, SURNAME
LOCATION   ← ADDRESS, LANDMARK
IDENTIFIER ← AADHAAR, PAN, PHONE, EMAIL
```

The structure earns its place because the branches need different machinery.
`IDENTIFIER` has rigid surface forms, so regex is viable. `PERSON` is an open
class — no pattern exists, and names collide with ordinary English words
(Rose, Prince, Happy). `LOCATION` is landmark-relative in Indian speech, so
neither regex nor a gazetteer suffices. Scoring rolls leaves up to branches
for per-category reporting.

### `asr.py` — speech to text, and the bridge

```python
model.transcribe(samples, beam_size=5, word_timestamps=True,
                 language="en", vad_filter=False)
```

**`beam_size=5`.** Whisper emits one token at a time. Greedy decoding takes
the single highest-probability token each step and cannot recover from a bad
early choice. Beam search keeps the 5 best partial sequences alive, extends
each, keeps the best 5 again. Wider beam = better transcription, more compute.
This is the Unit 2 "best-first search" content, and sweeping it is the
latency/quality experiment.

**`word_timestamps=True`.** Whisper does not natively emit word timings.
faster-whisper derives them from the **cross-attention weights**: when the
decoder emits a word, attention reveals which audio frames it attended to, and
dynamic time warping over that attention matrix yields a monotonic word-to-time
alignment. This is also *why* the timings are only good to a few hundred
milliseconds — attention is a soft, diffuse signal, not a boundary detector.

**`vad_filter=False`.** Voice Activity Detection drops audio it judges
non-speech *before* transcription. For a redaction tool that is dangerous:
silently skipped audio is un-redacted audio.

### `detect/` — character spans, never audio

Every level returns `Detection(pii, char_start, char_end, text, source, score)`
over whatever string it was handed. No detector knows audio exists. See Part 5.

### `redact.py` — destroying the audio

**Overwrite, never attenuate.** Ducking the volume or mixing a tone on top
leaves the original signal recoverable. Assigning over the samples destroys
it. This is the difference between a redaction tool and a cosmetic one.
Verified: correlation between original and redacted inside a beeped region is
**−0.0086**.

**The margin is a reported parameter, not a tuned constant.** Whisper's
timings are good to a few hundred ms and the errors are asymmetric in cost: a
late start leaks a name's first syllable, an early end leaks the last digit.
Default 150 ms each side. We report the trade-off curve — more margin means
better coverage of sensitive audio but more ordinary speech destroyed.

The beep has 5 ms raised-cosine fades at each edge; without them the jump from
speech to sine clicks audibly.

---

## Part 5 — The two abstractions that matter

### 5.1 What a `Detection` is

A **dataclass** — a small typed record. `detect(text: str) -> list[Detection]`:
hand it a string, get back one record per thing found.

```
INPUT:  'Hi, it is 9876543210 and my PAN is ABCDE1234F.'
         0123456789...

OUTPUT: [ Detection(pii='PAN',   char_start=35, char_end=45,
                    text='ABCDE1234F', source='regex:pan_std',     score=1.0),
          Detection(pii='PHONE', char_start=10, char_end=20,
                    text='9876543210', source='regex:phone_digits', score=1.0) ]
```

Each field:

| field | what it is | why it exists |
|---|---|---|
| `pii` | leaf category from the taxonomy | per-category scoring |
| `char_start`, `char_end` | a slice into the input string | **the only location info** — `text[35:45]` gives it back |
| `text` | the matched substring | debugging and reports |
| `source` | which rule or model fired | §2 traceability: every beep has a nameable reason |
| `score` | confidence, 0–1 | regex is always 1.0; a neural model gives a real probability, which is what you threshold on |

A `Detection` is **a pointer into a string, plus a label**. Nothing more. It
does not know what audio is, when anything was said, or where it came from.

Why all four comparison levels return this identical shape:

1. Detectors are unit-testable on plain strings — no audio fixtures.
2. All four levels produce directly comparable output, so §4.3's table is fair.
3. **The error-attribution experiment becomes a one-line change** —
   `detect(gold_transcript)` vs `detect(asr_transcript)`, then diff.

### 5.2 How the two coordinate systems actually join

The two systems:

* **Character space** — positions in the transcript string. Where detection happens.
* **Time space** — seconds (hence sample indices) in the audio. Where redaction happens.

Transcription does **not** join them. Transcription *creates* character space.
If all Whisper returned was a string, there would be no way back to the audio
at all.

**What joins them is the word-level timestamps.** Each `Word` carries a
position in *both* systems simultaneously:

```
word               CHAR SPACE         TIME SPACE
'9876543210.'         187-198        12.38-14.80
'Got'                 199-202        15.14-15.40
'it.'                 203-206        15.40-15.66
'Could'               207-212        15.94-16.06
```

**The list of `Word` objects is a translation table between the two coordinate
systems**, and `char_to_time` is a lookup in it:

```python
hit = [w for w in self.words
       if w.char_end > char_start and w.char_start < char_end]
return min(w.start for w in hit), max(w.end for w in hit)
```

"Which words overlap characters 187–197? Just `'9876543210.'`, which lives at
12.38–14.80 s." Then × 16000 → samples 198,080 to 236,800. Overwrite.

Two consequences worth being able to state:

* **The table has word granularity, not character granularity.** A detection
  covering half a word beeps the whole word. Usually harmless, occasionally
  over-redacts.
* **The timestamps are approximate** (attention + DTW, a few hundred ms), which
  is exactly why `redact.py` adds a safety margin.

---

## Part 6 — Build order, and what verification caught

A **bad but complete** pipeline came before any part was good. Day 1's
detector finds four identifier types by regex and cannot find a name at all.
That is fine — it is *complete*.

The alternative — build each module properly, integrate on Day 5 — is how
student projects die: every interface mismatch surfaces at once with no time
left.

| Check | Result | Why it mattered |
|---|---|---|
| `transcript[char_start:char_end] == span.text` | 6/6 consistent | an off-by-one would silently corrupt every score all week |
| RMS inside each gold span | 0.10–0.14 | confirms trimming worked; a span over silence means wrong timing |
| `say --data-format` direct WAV | works | removed ffmpeg from the render loop (~1000 calls saved) |
| faster-whisper on Python 3.13 | works | would have forced a downgrade on Day 4 |
| corr(original, redacted) in beeped region | **−0.0086** | **this is the privacy claim** — near-zero means gone, not quieter |
| name region byte-identical | True | we destroy only what we detect |

### A bug worth remembering

The first `asr.py` had `char_to_time` reading `w.char_start` / `w.char_end` —
fields `Word` **did not have**. It would have crashed the moment detection was
wired up, but did not crash earlier because nothing called it. That is the
characteristic failure of building components in isolation: broken interfaces
stay invisible until integration. Building the spine end-to-end on day one
surfaced it within the hour.

---

## Part 7 — Findings so far

**F1. Whisper normalises spoken digits to numerals.** "nine eight seven six
five four three two one zero" → `9876543210`. This partly contradicts §1's
premise that regex fails on spoken numbers: for Whisper, the ASR does the
normalisation *for* the regex. The open question is whether it is
*consistent* — unpredictable normalisation is worse than reliable failure,
because you cannot write a rule against it. *(n=1, not yet a result.)*

**F2. Half-normalisation destroys structured formats.** "rose dot mathew at
gmail dot com" → `rose .matthew at gmail .com`, matching neither the standard
email pattern nor the spoken one. Regex found it on the gold transcript and
lost it on the ASR transcript — the error-attribution experiment in miniature.

**F3. Name corruption confirmed.** "Mathew" → "Matthew"; "petrol pump" →
"petrol firm". Both §1 predictions holding up.

---

## Part 8 — Questions an interviewer would actually ask

**"How do you know where in the audio to beep?"**
Word-level timestamps. Whisper does not emit them natively — faster-whisper
recovers them from the decoder's cross-attention weights via dynamic time
warping, giving each word a start and end time. Each word then carries a
position in both character space and time space, so the word list is a
translation table. A detection in character space is looked up against it.
They are approximate to a few hundred ms, so we add a safety margin.

**"Why not just use a regex for phone numbers?"**
Two reasons, and one of them turned out to be wrong, which is the interesting
part. Regex cannot find names or addresses at all — that stands. But we
*predicted* regex would fail on spoken-out digits, and when we tested it
Whisper had already normalised "nine eight seven…" to `9876543210`, so the
regex worked. What it actually broke on was half-normalised email. So the real
finding is that the ASR performs a hidden, inconsistent normalisation step
that silently decides whether your downstream regex works.

**"How do you know your redaction actually removes the information?"**
We measured the correlation between the original and redacted signal inside
each beeped region: −0.0086. Near zero means the original is gone. If we had
attenuated the volume or mixed a tone on top — both common implementations —
correlation would stay high and the audio would be trivially recoverable.

**"Where does your system lose the most accuracy?"**
That is the experiment the project is built around: run the identical detector
twice, once on the ground-truth transcript and once on Whisper's, and diff.
The gap is error contributed by speech recognition; the remainder is the
detector's own. It tells you which half of the pipeline deserves the effort.

**"How did you get ground-truth timings without hand-labelling?"**
Rendered each dialogue segment-by-segment and accumulated the sample offset,
so the offset where a PII segment lands *is* its true start time. The
alternative, forced alignment, would have been circular — the aligner was also
going to be a pipeline stage, so it would have been both the thing under test
and the definition of truth.

**"What is the weakest part of your evaluation?"**
The corpus is mostly TTS, and because segments are rendered separately there
is a slight pause at every PII boundary that real speech would not have. That
is a cue the ASR gets for free, so our numbers are optimistic. We report it,
and the human-recorded subset measures the size of the gap.
