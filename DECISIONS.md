# Decisions log

Every non-obvious choice, what we rejected, and why. Written as we go.

---

## D1. Ground-truth timings come from segment-wise TTS, not forced alignment
**Date:** 9 Nov · **Affects:** §4.1, §4.2, §5

A dialogue is authored as alternating ordinary/PII segments. Each is rendered
as its own utterance, silence-trimmed, and concatenated while accumulating a
sample offset. The offset where a PII segment lands *is* its true start time.

**Rejected:** forced alignment (WhisperX + wav2vec2) against a known script.

**Why:** the proposal wanted alignment as both a pipeline stage (sharpening
Whisper's timings) *and* the source of ground truth. That is circular — the
timing metric would partly measure the aligner against itself. Segment
rendering removes the circularity and drops a model, an install, and a
failure mode.

**Cost we accept:** `say` gives each segment utterance-final prosody, so there
are unnatural seams at segment boundaries. A pause at a word boundary is a cue
real speech would not provide, so our ASR numbers are optimistic. Declared in
limitations; the human-recorded subset measures the gap.

## D2. Silence-trim every segment before concatenating
**Date:** 9 Nov · **Affects:** §5 coverage metric

`say` pads each utterance with silence — measured at 210 ms on a 2.4 s
utterance, about 9%. Left in, every gold span would claim the PII starts
before it is audible, and beep-coverage would be scored against boundaries
that are partly silence.

## D3. Audio I/O is stdlib `wave` + numpy only
**Date:** 9 Nov · **Affects:** reproducibility

No soundfile, no librosa. We only ever handle 16 kHz mono 16-bit PCM, which
`wave` handles natively. Every dependency is a way for the grader's clone to
fail, and this project is on the Demo track.

## D4. Feed faster-whisper raw samples, not a file path
**Date:** 9 Nov · **Affects:** reproducibility

faster-whisper 1.2.1 decodes paths through PyAV, and PyAV 19 removed a kwarg
it still passes (`TypeError: open() got an unexpected keyword argument
'metadata_errors'`). We already hold float32 16 kHz mono — exactly what it
wants — so passing the array routes around the decoder.

**Rejected:** pinning `av` to an old version. Routing around it removes the
dependency from the critical path instead of freezing it.

## D5. Whisper `small`, int8, CPU
**Date:** 9 Nov · **Affects:** §5 speed, §6

Hardware decision, not a quality one. 8 GB M2: `medium` fits but swaps, and we
re-transcribe the corpus many times this week. `beam_size` stays exposed
because §2 claims Whisper's beam search as the Unit 2 content, and sweeping it
is what backs that claim.

## D6. Detectors operate on text offsets; audio mapping is a separate bridge
**Date:** 9 Nov · **Affects:** §4.3, Day 4 experiment

All four comparison levels return `Detection(pii, char_start, char_end, ...)`
over whatever string they were handed. Nothing in a detector knows about
audio. `Transcription.char_to_time` is the only bridge.

**Why it matters:** the error-attribution experiment becomes trivial — run the
same detector on the gold transcript and on the ASR transcript and diff. No
special-casing, and detectors stay unit-testable on plain strings.

## D7. Overwrite samples with a tone; never duck or mix
**Date:** 9 Nov · **Affects:** §4.1, privacy claim

Verified: correlation between original and redacted audio inside a beeped
region is −0.009. Attenuating or layering leaves the original recoverable,
which would make the privacy claim false.

## D8. Redaction margin is a reported parameter, not a tuned constant
**Date:** 9 Nov · **Affects:** §5

Whisper's word timings are good to a few hundred ms, and the errors are not
symmetric in cost: a late start leaks a name's first syllable, an early end
leaks the last digit. Default 150 ms each side. §5 wants the trade-off curve
(coverage of sensitive audio vs. ordinary speech destroyed), not one number.

---

## Open findings (Day 1, n=1 dialogue — not yet a result)

**F1. Whisper normalises spoken digits to numerals.** "nine eight seven six
five four three two one zero" transcribed as `9876543210`. This partly
contradicts §1's premise that regex fails on spoken numbers — for Whisper
specifically, the ASR does the normalisation *for* the regex. Needs measuring
across the corpus: the interesting question is whether it is *consistent*.
Unpredictable normalisation is arguably worse than reliable failure.

**F2. Half-normalisation destroys structured formats.** "rose dot mathew at
gmail dot com" → `rose .matthew at gmail .com`, which matches neither the
standard email pattern nor the spoken one. Regex found this on the gold
transcript and lost it on the ASR transcript.

**F3. Name corruption confirmed.** "Mathew" → "Matthew"; "petrol pump" →
"petrol firm". Both are §1 predictions holding up.

---

## Day 2

## D9. Split by template, never by dialogue
**Date:** 9 Nov · **Affects:** §4.2, every number we report

Dialogues generated from one template share their phrasing; only slot fills
differ. Splitting at the dialogue level would put the same sentences in both
halves, so anything keyed on phrasing — a tuned threshold, a rule, a model —
would be scored on material it had effectively already seen.

Whole templates are assigned to a split: 15 templates (60 dialogues) dev,
23 templates (92 dialogues) test, **zero template overlap**, stratified by
scenario so both halves see a similar mix of call types.

We train nothing, but we *tune* the detection threshold and redaction margin
on dev and report on test. Same leakage risk, same fix.

## D10. How a number is read aloud is an experimental variable
**Date:** 9 Nov · **Affects:** §4.2, §5, finding F1

Day 1 showed Whisper silently normalising spoken digits into numerals, which
made the regex succeed where §1 predicted failure. So speaking style is now
the independent variable, not an incidental detail. Every numeric identifier
is generated with a known canonical value and rendered in one of four
realistic styles — `digit_by_digit`, `with_doubles` ("double seven"),
`paired` ("ninety eight, seventy six"), `grouped` ("nine eight seven, six
five four") — balanced across the corpus at 24 instances each.

`GoldSpan` now carries `canonical` and `style` alongside the spoken text, so
we can ask "did Whisper recover the right digits?" separately from "how were
they spoken?". Without the canonical value a correctly-normalised transcript
would look like a mismatch against the spoken form.

## D11. Only PII slots break a rendering segment
**Date:** 9 Nov · **Affects:** audio naturalness

Every segment boundary is an unnatural pause (D1's declared cost), so we
create as few as possible. Plain slots like `{{city}}` are substituted into
the surrounding literal text and cost no seam. Only PII slots split a
segment, because only those need their own timing.

## D12. Punctuation-only segments are transcribed but not rendered
**Date:** 9 Nov · **Affects:** corpus integrity

`say` *speaks* bare punctuation rather than pausing — measured at 0.96 s for
"." and 1.10 s for "?". The corpus had 288 punctuation-only segments, almost
all of them immediately after a PII span, which would have injected roughly
four minutes of spurious audio exactly where timing accuracy matters most.
They now stay in the reference transcript and render nothing.

## D13. Two render workers, not six
**Date:** 9 Nov · **Affects:** build time

Six concurrent `say` processes plus Python drove the 8 GB M2 into swap: load
average 15 on 8 cores, 2.6 GB swapped, 1.2 M pageouts. Throughput collapsed
about 30× — the first 20 dialogues took 59 s, the next 20 took 32 minutes.
The binding constraint is memory, not CPU, so a smaller pool is strictly
faster on this hardware. Rendering is now resumable, so an interrupted run
costs nothing.

## D14. Transcriptions are cached to disk
**Date:** 9 Nov · **Affects:** iteration speed all week

ASR is the expensive step (~4 s per dialogue) and everything downstream
depends only on its output, so the corpus is transcribed once per
(dialogue, model, beam_size) and reloaded thereafter. Detector changes then
cost nothing to re-evaluate, which is what makes Day 3's four-way comparison
and Day 4's attribution experiment affordable.

## D15. Render-time validation: `say` reports success while writing silence
**Date:** 9 Nov · **Affects:** ground-truth integrity

Five PII segments in the first corpus build came back as 0.017–0.032 s —
exactly the 15 ms trim padding, meaning a transient followed by silence.
`say` had exited 0 in every case. The renders most likely failed during the
memory thrashing in D13.

This is worse than it looks for 5/440. A silent gold span is PII that **no
detector can ever find**, so it is an unfixable miss that depresses every
recall number we report — and not equally across the four comparison levels,
which is exactly the kind of bias that makes a comparison table lie.

`_say` now rejects any segment under 80 ms and retries up to three times
before raising, rather than trusting the exit code. Corpus re-validated:
440/440 spans land on real speech, 440/440 character offsets consistent.

---

## Day 2 (evening) — the normalisation result

## D16. The regex baseline is ASR-aware, because a weak one would flatter us
**Date:** 9 Nov · **Affects:** §4.3, every later comparison

§4.3 says comparing against a weak regex would be unfair *in our favour*.
Measured on the corpus, that is worth **49 points of recall** on numeric
identifiers, so a naive baseline would have inflated every later level by the
same amount. The Level 1 patterns are therefore written against what the ASR
actually emits, not against clean typed text.

**Rejected:** rewriting the transcript to normalise it before matching. The
tolerant pattern matches in the original string, so character offsets stay
valid and no offset remapping is needed.

---

## Findings (measured on 152 dialogues, 440 spans)

**F1 — resolved. Whisper always normalises spoken digits, but fragments the
run.** Of 96 numeric identifiers, **0 were left as words**. §1's premise — that
regex fails because spoken numbers stay as words — is simply false for
Whisper. But the naive opposite reading is also false. Whisper preserves the
speaker's prosodic grouping as *punctuation*:

```
spoken  "nine six zero, five nine five, seven four five, four"
ASR     960 -595 -745 -4
spoken  "ninety seven, forty seven, seventy six, sixty nine, fourteen"
ASR     97, 47, 76, 69, 14
```

The digits are present and correct. A pattern demanding ten *consecutive*
digits cannot see them.

| pattern | recall on numeric PII | false positives |
|---|---|---|
| consecutive digits | 32.3% (31/96) | 0 |
| separator-tolerant | **81.2% (78/96)** | 0 |

The fix costs nothing in precision. Note 81% exceeds the 78% of cases where
the digits were fully correct — **for redaction the transcript does not need
to be right, it needs to be wrong in a shape-preserving way.** A mangled phone
number still gets beeped; a fragmented one does not.

**F2 — emails get a half-normalisation matching neither form.** "dot" becomes
a literal "." glued to the next token while "at" stays a word:
`neha dot goli at gmail dot com` → `neha .goli at gmail .com`. Matches neither
the typed pattern nor the spoken one. Every separator in Level 1 now accepts
all three surface forms.

**F4 — `with_doubles` is the worst speaking style, and the mechanism is not
what it first appeared.** Digit-count recovery by style: grouped 22/24,
paired 22/24, digit_by_digit 19/24, **with_doubles 16/24**.

An early single example suggested "double" survives as a literal word and
breaks the run. Measured across the corpus that happens **once in 24**. The
dominant failure is Whisper *understanding* "double four" and emitting a
single `4` — a silently dropped digit (4/24, delta −1). In 2/24 it mishears
"double" into words entirely ("the second architecture"), an artefact that
appears in **0 of 72** spans in every other style.

This matters for the report because "double seven" is a common way to read a
number aloud in India, and it is the style the ASR handles worst.

**F5 — Level 1 baseline established.** Regex reaches **76% recall (134/176)**
on the four types it can address, with **0 false positives** across 36
dialogues containing none. It cannot address PERSON or LOCATION at all, so
across all 440 spans Level 1 recovers **30%**. That 220-span gap is what
Levels 2–4 exist to close.

---

## Day 2 (late) — human recording kit

## D17. Readers read scripts already in the TTS corpus
**Date:** 9 Nov · **Affects:** the TTS-flattery measurement

Every human recording has a synthetic twin with identical text and identical
PII, so the only variable is the voice source. New dialogues would confound
voice with text and any difference would be unattributable.

This is what turns D1's declared limitation — "TTS seams flatter the ASR, our
numbers are optimistic" — from an assertion in the limitations section into a
measured quantity, on a paired design that is far more powerful at n=42 than
an unpaired one.

## D18. The detector is not involved in producing human ground truth
**Date:** 9 Nov · **Affects:** validity of every human-audio result

PII in the human recordings is located by matching the **known script**
against the transcript, never by running our detector. Script supplies the
labels, Whisper supplies the timings, a human confirms the boundaries.

**Why it matters:** if we beeped the *detector's* output and asked someone
"can you still hear the name?", a yes would be ambiguous between a detector
miss — the result we are trying to measure — and a bad label. We would end up
editing ground truth until our own system looked perfect. Two passes, in
order: pass 1 builds truth, pass 2 is graded against it.

## D19. No margin on proposed ground truth
**Date:** 9 Nov · **Affects:** sensitivity of the verification step

The first version padded each proposed boundary by 120 ms. That makes the
check audio insensitive to exactly the error it exists to catch — pad either
side and a boundary 100 ms wrong still sounds clean, so the listener confirms
a bad label. It also made a given name and the surname after it overlap.

Ground truth is the true extent of the speech. The 150 ms redaction margin is
a separate, reported parameter and lives in `redact.py`.

## D20. Span confidence is scored against the canonical value too
**Date:** 9 Nov · **Affects:** how much manual checking the kit costs

A perfectly transcribed phone number reads `9584917185` where the reader said
"nine five eight four nine one seven one eight five" — zero string overlap
despite being exactly right. Scoring similarity only against the spoken form
flagged 9 of 10 recordings for manual review. Comparing against the canonical
value as well drops that to 1, and the survivor is a genuine failure
(`Joy` → `Ja`, a word-like name mangled by the ASR).

## D21. Reader sheets hyphenate letter-by-letter sequences
**Date:** 9 Nov · **Affects:** PAN data quality

A PAN spelled out can contain `D O T`, which a reader may well say as "dot".
Sheets render those as `N-D-O-T-P-zero-one-nine-eight-X`. This changes only
what the reader sees — the spoken content is identical, so the pair with the
TTS twin holds.

Sheets carry no PII markup at all, for the same reason: marking the sensitive
words would invite careful over-enunciation of precisely the words whose
natural pronunciation we are measuring.
