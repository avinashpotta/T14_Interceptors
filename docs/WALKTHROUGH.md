# Walkthrough — how this system works and why

Written to be read cold, months later, before an interview. `DECISIONS.md` is
the terse log of choices; this is the explanation underneath them.

Grows as the project does. Complete through Day 5.

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

## Part 9 — Building a corpus you can trust (Day 2)

### Data leakage, and why template-level splitting

Every dialogue generated from one template shares its *phrasing*; only the
slot fills differ. Split the 152 dialogues randomly and `t04_00` lands in dev
while `t04_01` lands in test — near-identical sentences. Anything keyed on
phrasing (a tuned threshold, a rule, a model) is then scored on material it
has effectively already seen, and the test number is inflated and meaningless.

Whole templates are assigned to a split: 15 templates / 60 dialogues dev,
23 / 92 test, **zero overlap**, stratified by scenario.

The objection you will get: *"you train nothing, so why does leakage matter?"*
Because we still **tune** — the detection threshold, the redaction margin, the
capitalisation filter are all chosen on dev. Tuning is fitting. Same risk,
same fix.

### Speaking style as an experimental variable

Day 1's finding (Whisper normalising spoken digits) promoted "how was the
number read aloud" from an incidental detail to the independent variable.
Every numeric identifier is generated with a known canonical value and spoken
in one of four realistic styles, balanced at 24 each:

```
canonical 6983444174
  digit_by_digit  six nine eight three four four four one seven four
  with_doubles    six nine eight three double four four one seven four
  paired          sixty nine, eighty three, forty four, forty one, seventy four
  grouped         six nine eight, three four four, four one seven, four
```

`GoldSpan` carries `canonical` alongside the spoken text, so "did Whisper
recover the right digits?" is separable from "how were they spoken?".

### Two bugs that would have silently corrupted everything

**`say` speaks punctuation.** Not a pause — `.` produces 0.96 s of audio, `?`
produces 1.10 s. There were 288 punctuation-only segments, nearly all
immediately after a PII span. That is ~4 minutes of spurious audio injected
exactly where timing accuracy matters most.

**`say` exits 0 while writing silence.** Five PII segments came back as
0.017-0.032 s — the trim padding alone. A silent gold span is PII **no
detector can ever find**: an unfixable miss that depresses recall unequally
across the four levels, which is how a comparison table ends up lying.

The general lesson: **a zero exit code means the program did not crash, not
that it did the job.** When a tool's real output is a file, validate the
artefact, not the status.

---

## Part 10 — Measuring it honestly (Day 3)

### The metric trap, found before any detector ran

The first scoring harness used span-counted precision. Tested against a
detector that beeps **the entire recording**, it reported:

```
P=1.00  R=1.00  F2=1.00
```

Perfect — while destroying the whole call. One giant span overlaps every gold
span, so it records zero false positives.

The fix is that **recall and precision need different units**:

* **Recall is span-level** — "was each sensitive item protected?" A person is
  either protected or not, so items are the unit.
* **Precision is duration-level** — "how much of what we destroyed needed
  destroying?" Seconds are the unit, because one enormous beep is far worse
  than one small one.

Under duration-based precision the same beep-everything detector scores F2
0.62 with over-redaction flagged at 100%.

### F2, and why precision is not free

F2 = 5PR / (4P + R). Recall counts four times as heavily because missing an
identifier is a breach and beeping a spare word is an annoyance. That does
**not** make precision irrelevant:

| approach | recall | precision | F2 |
|---|---|---|---|
| aggressive gazetteer | 0.95 | 0.25 | **0.61** |
| conservative | 0.70 | 0.90 | **0.73** |

The aggressive version *loses* despite 25 more points of recall. And `over_redaction`
captures a cost F2 misses entirely: a call where a third of the words are
tones is not redacted, it is ruined.

### Bootstrap over dialogues, not spans

Spans inside one dialogue share a speaker, a template and a transcription, so
they are not independent. Resampling spans would give intervals several times
too narrow and manufacture significance that is not there. The dialogue is
the independent unit.

### Overlapping intervals are not a significance test

Reading two separately-computed CIs and checking for overlap is **strictly
more conservative** than testing whether the difference is zero, and it
routinely reports "no difference" for real ones. Both systems score the same
dialogues, so per-dialogue noise is shared and cancels when the pair is
resampled together.

It changed a conclusion here. `rules + NER` [0.67-0.76] vs `NER only`
[0.62-0.72] looks indistinguishable from the intervals. Paired:
**+0.049 [+0.026, +0.077]** — clearly real.

### The gazetteer needs a stoplist, and the measurement proves it

Using only 51 ambiguous names (a real list has ~100k), on our own transcripts:
**24% precision**. The false fires are the lesson:

```
"May I know who I'm speaking with?"   <- may   (the most common agent phrase)
"I'll need your PAN card details"     <- card
"Your bill has been corrected"        <- bill
```

Hence stoplist + context cues + capitalisation — which is exactly the
predicate logic §2 specifies: *redact if it is in a name list and not a common
English word, OR if it follows a name-introducing phrase*. The second clause
is what catches a name no list contains.

### The ORG trap

spaCy emits PERSON, GPE, LOC, FAC, ORG. `ORG` is deliberately **excluded**:
our dialogues are full of "Nexa Telecom", "CityFirst Bank" — the company being
*called*, which is not the customer's PII. Tagging it would beep a brand name
in nearly every dialogue for no privacy gain.

### Detection success is not redaction success

The single most useful diagnostic. PAN was *detected* 81% of the time and
*fully redacted* only 38%:

```
spoken  "M W P H M eight one three three G"   2.13 s
ASR     "MWPHM8133G"                          0.52 s   <- 24% of the duration
```

A collapsed token gets one narrow attention peak from the DTW alignment, so
its timestamp covers a quarter of the audio it represents. The detector was
right; the clock was wrong. Expanding each detection to the neighbouring
undetected words took PAN 38% -> 81% and PHONE 54% -> 68%, for 7.5%
over-redaction instead of 3.7%.

**Only a coverage metric kept separate from a detection metric can tell you
which of the two failed.**

### The attribution experiment

Run the identical detector on the reference transcript and on Whisper's, and
diff. Every item falls in exactly one of four cells:

| | found on ASR | not found on ASR |
|---|---|---|
| **found on reference** | both (62%) | **lost to ASR** (13%) |
| **not found on reference** | **rescued by ASR** (13%) | neither (12%) |

Of the 67 failures: **52% destroyed by the ASR, 48% the detector could not
find anyway.** Neither half of the pipeline dominates.

The fourth cell was discovered by fixing broken accounting that reported
"detector never had it: 101%". For PAN, AADHAAR and PHONE the ASR **rescues**
items — spoken identifiers are harder to pattern-match than normalised ones,
so a perfect transcript would make things worse.

---

## Part 11 — Degradation, and a negative result (Day 4)

### Pink noise, not white

White noise has flat power per hertz, so most of its energy sits above the
speech band and an ASR shrugs it off — it would make the test look easy. Pink
noise falls at 1/f, roughly how room hum, traffic and chatter are actually
distributed, so it competes with speech. Level is set by **SNR against the
speech itself**, not a fixed amplitude, so a quiet dialogue and a loud one are
degraded comparably.

### It made no difference, and that is the result

| condition | ΔF2 vs clean | significant |
|---|---|---|
| pink noise, 20 dB | +0.008 | no |
| pink noise, 10 dB | −0.015 | no |
| room echo | +0.001 | no |

Whisper was trained on 680,000 hours of noisy web audio. Our first attempt
simply tested the wrong thing, and "we added noise and nothing happened" is a
finding about robustness rather than a failed experiment.

### The degradation that does matter

A support call is not degraded by pink noise; it is degraded by **the phone
line**. Band-limiting to 300-3400 Hz plus 8-bit μ-law companding removes
frequency bands outright rather than burying them.

| condition | ΔF2 | significant |
|---|---|---|
| telephone band | −0.031 | borderline |
| **telephone + noise** | **−0.053** | **yes** |

Email degrades worst (75% → 56%) because it survives as a *multi-token
structure* — `neha .goli at gmail .com` — and the pattern needs nearly all of
it intact. A name needs one token to survive; an email needs five.

---

## Part 12 — Human speech, and where synthesis actually lied (Day 5)

### The paired design

Readers recorded scripts **already in the TTS corpus**, so each human
recording has a synthetic twin with identical text and identical PII. Only the
voice source varies, so any difference is attributable. New dialogues would
have confounded voice with text.

### Detection accuracy: no measurable penalty

Human minus TTS detection rate: **−3.1%, 95% CI [−10.7%, +3.9%]** — not
significant at 40 recordings. Speaker-to-speaker variation (P2 −13%, P3 +9%)
*exceeds* the TTS/human difference.

D1 declared that segment-rendered TTS would flatter the ASR. On detection
accuracy we cannot establish that it did.

### But it badly understated the latency budget

| type | TTS median | human median | human max |
|---|---|---|---|
| AADHAAR | 3.91s | 5.80s | 8.06s |
| PAN | 2.31s | 4.98s | **11.76s** |
| GIVEN_NAME | 0.49s | 0.49s | 1.16s |

Names take the same time. Identifiers take 50-115% longer, because people slow
down, group digits and pause when reading out an account number. A synthesiser
does not.

```
delay for 99% of spans :  TTS 4.7s   human 8.0s
```

**The corpus understates the required delay by ~70%.** The bias was real and
large, and it appeared in a dimension nobody was watching — not in accuracy.

### A metric that measured my own code

The first version of this analysis also reported a "span survived in the
transcript" rate. It read **100% for both conditions**, which was not
believable — and checking showed `project_span` returns `None` for 0 of 129
spans, because positional mapping of a `replace` block returns a location even
when the word is entirely wrong. It measured alignment behaviour, not ASR
survival. Removed rather than published.

**The habit worth keeping: a number that is suspiciously clean is a bug report
until proven otherwise.**

---

## Part 13 — The streaming constraint

To redact a span the ASR must see **all** of it, which cannot happen until
capture passes the span's end. At that moment playback sits `delay` behind. So
a span survives only if **its spoken duration is under the delay budget** —
regardless of hardware.

Confirmed directly: at a 3 s delay a 5.24 s phone number leaked, while every
name in the same dialogue was redacted.

Two constraints, different natures:

| | nature | value |
|---|---|---|
| latency floor | structural, hardware-independent | ~8 s for human speech |
| throughput | engineering, solvable | 1.17× realtime on an 8 GB M2 |

Television profanity delay is 5-10 seconds for exactly this reason, which is
some evidence the constraint is real rather than an artefact.

**De-duplication across overlapping windows is free if you use a mask.**
Rather than reconciling word sequences across passes, keep a boolean mask over
absolute sample positions and OR each detection in. Idempotent,
order-independent, and a span missed on one pass can still be caught on the
next.

---

## Part 14 — Comparing fairly against someone else's system

Presidio is configured **charitably on purpose**, on the same principle that
stopped us writing a strawman regex baseline:

* keep every entity type that could land on real PII, **including accidental
  matches** — a 12-digit Aadhaar returns as `US_BANK_NUMBER`, and that counts
  as a hit, because redaction scoring is type-agnostic: the person is protected
  whatever label caused it
* drop only the four types with no counterpart in our gold, which helps its
  precision, not ours
* run it on the same spaCy model our own NER uses, so the comparison is not
  confounded by model size

Result: F2 0.500 vs our 0.719, paired **+0.220 [+0.166, +0.271]**. It beats
our regex level by +0.180, so it is a real mid-tier baseline.

It is **competitive on names** (84% surname vs our 89% — shared spaCy
backbone) and **collapses on identifiers** (EMAIL 14%, AADHAAR 30%, PAN 12%),
because its recognisers expect typed text. That is the same mismatch our own
patterns were built around, measured on somebody else's system.

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

**"How do you know your comparison between approaches is fair?"**
Three things. The gazetteers come from outside our own vocabulary — building
them from the generation vocab would have given the baseline perfect
knowledge of every corpus name, an oracle rather than a baseline. The regex
baseline is written against what the ASR actually emits, which is worth 49
points of recall, so a naive version would have flattered every later level by
that much. And splits are template-level, so no phrasing appears in both.

**"How do you know a difference between two systems is real?"**
Paired bootstrap over dialogues. Not overlapping confidence intervals — that
is strictly more conservative and reports "no difference" for real ones. Not
resampling spans — spans within a dialogue share a speaker and a
transcription, so that would understate the interval badly.

**"Would a bigger name list have worked instead of a model?"**
We measured it. A raw list scored 24% precision on our transcripts because a
large international gazetteer contains *may*, *will*, *card*, *bill*. With a
stoplist and context rules it reaches F2 0.637; the NER alone reaches 0.672;
neither is significantly better than the other. But each adds significantly on
top of the other — +0.049 and +0.083 — so they are complementary, not
redundant. On given names the rules get 78%, the NER 69%, and together 85%:
they recover different names.

**"What was the hardest bug?"**
A metric bug, not a code bug. Span-counted precision scored a detector that
beeps the entire recording at a perfect 1.00. It was caught by testing the
scorer against deliberately absurd detectors before any real one existed.

**"Where does your system fail?"**
Landmark-relative addresses — "opposite the Forum Mall petrol pump". Regex 0%,
gazetteers 4-7%, NER 4%, everything combined 7%. It defeats patterns, name
lists and an OntoNotes NER alike. That is §1's claim about Indian addresses,
confirmed from the failure side.

**"If you had another week, what would you do?"**
Attribution says it would be split evenly — 52% of remaining failures are
transcription, 48% detection — so there is no single lever. Concretely:
phonetic matching for the recoverable near-misses (`Rithwik` -> `Ritwik`), and
something for landmark phrases, which currently nothing handles.

**"Your noise experiment showed nothing. Was it wasted?"**
No — it showed Whisper is robust to additive noise and reverb at realistic
levels, which is worth knowing. It also told us we had tested the wrong thing,
which is why we then tested telephony band-limiting: that *is* significant
(−0.053 F2), and it is what a real support call does to speech.

**"Can this run in real time?"**
Two separate questions. Throughput is an engineering problem — 1.17× realtime
on an 8 GB laptop, fixable with a smaller model or a GPU. The latency floor is
not: you cannot redact what has not been said yet, so the delay must exceed
the longest PII utterance, which we measured at ~8 seconds for human speech.
That is fine for redacting a recording before it is stored, and unusable for a
two-way conversation. Broadcast profanity delay is 5-10 seconds for the same
reason.

**"How did you avoid flattering your own system?"**
Deliberately strengthening everything we compare against. The regex baseline
is written for ASR output rather than typed text, worth 49 points of recall.
The gazetteers come from outside our generation vocabulary. Presidio keeps
hits that land for the wrong reason. Each of those choices makes our margin
smaller.

**"What surprised you most?"**
That for structured identifiers the ASR *helps* — a perfect transcript would
score worse, because spoken digits are harder to pattern-match than
normalised ones. And that synthetic speech, which we expected to flatter our
accuracy, did not measurably do so — it understated the streaming latency
budget by 70% instead.
