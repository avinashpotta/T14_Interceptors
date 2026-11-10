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

---

## Day 3 — scoring and the name layer

## D22. Recall is span-level, precision is duration-level
**Date:** 10 Nov · **Affects:** every headline number

Recall asks "was each sensitive item protected?" — a person is either
protected or not, so items are the unit. Precision asks "how much of what we
destroyed needed destroying?" — seconds are the unit, because one enormous
beep is far worse than one small one.

Span-counted precision gets this catastrophically wrong, and our own harness
proved it before any detector ran: a detector beeping the **entire recording**
overlaps every gold span, records zero false positives, and scores a perfect
P=1.00 R=1.00 F2=1.00 while producing audio nobody can use. Under
duration-based precision the same detector scores F2 0.62 with over-redaction
flagged at 100%.

## D23. Bootstrap resamples dialogues, not spans
**Date:** 10 Nov · **Affects:** every confidence interval

Spans inside one dialogue share a speaker, a template and a transcription, so
they are not independent. Resampling spans would give intervals several times
too narrow and manufacture significance that is not there.

## D24. Gazetteers come from outside our own vocabulary
**Date:** 10 Nov · **Affects:** validity of Levels 2 and 4

Built from NLTK's US-Census names (western, 7.5k) and the Indian-Male/Female
name CSVs (indian, 8.7k), never from `data/vocab.json`. Building it from our
own vocab would give Level 2 perfect prior knowledge of every corpus name —
an oracle, not a baseline — and make Level 4 look bad for no real reason.

## D25. Compressed ASR tokens need their span expanded
**Date:** 10 Nov · **Affects:** §5 "did the beep land", PAN and PHONE recall

`M W P H M eight one three three G` is spoken over **2.13 s** and returns as
the single token `MWPHM8133G` timestamped at **0.52 s** — 24% of the real
duration, because the DTW alignment gives a collapsed token one narrow
attention peak. Beeping that range leaves three quarters of the PAN audible.

This is why PAN was *detected* 81% of the time but *fully redacted* only 38%.
`char_to_time` now claims the audio out to the neighbouring undetected words —
whatever the ASR collapsed must live in that gap — capped at 2 s so a long
pause cannot drag the beep across unrelated speech.

| | before | after |
|---|---|---|
| PAN recall | 38% | **81%** |
| PHONE recall | 54% | **68%** |
| over-redaction | 3.7% | 7.5% |

**Detection success is not redaction success.** That distinction is the whole
reason §5 asks for coverage separately, and it would have been invisible
without it.

---

## Finding F6 — a Western name list does not transfer to Indian names

§1 asserts this from the literature. Measured on our corpus:

| name group | western list | indian list |
|---|---|---|
| common Indian given names | **11%** | 80% |
| word-like given names | 58% | 75% |
| surnames | **10%** | 57% |

End-to-end recall on the test split follows directly: GIVEN_NAME 26% → 72%,
SURNAME 7% → 55% simply by swapping the list. The claim is no longer borrowed.

## Finding F7 — ADDRESS and LANDMARK are undetected, and it is not a timing problem

ADDRESS 7%, LANDMARK 7%. Unlike PAN, the ASR word timings cover 97–100% of
those gold spans, so the timing is fine — we simply have no detector for
them. Regex has no address pattern and a name gazetteer contains no streets.
This is what the NER half of Level 4 has to earn its place on.

## D26. Level comparisons use a paired bootstrap, not overlapping intervals
**Date:** 10 Nov · **Affects:** every claim about which level is better

Reading two separately-computed confidence intervals and checking whether
they overlap is not a test of whether the difference is zero. It is strictly
more conservative and routinely reports "no difference" for differences that
are real. Both systems are scored on the identical dialogues, so per-dialogue
noise is shared and cancels when the pair is resampled together.

It changed a conclusion here. From overlapping intervals, `rules + NER`
[0.67-0.76] versus `NER only` [0.62-0.72] looks indistinguishable. Paired, the
difference is **+0.049 [+0.026, +0.077]** — clearly real.

## D27. ORG is excluded from the NER label mapping
**Date:** 10 Nov · **Affects:** over-redaction

`PERSON` maps to our PERSON branch; `GPE`/`LOC`/`FAC` to LOCATION. `ORG` is
deliberately dropped: our dialogues are full of "Nexa Telecom", "CityFirst
Bank" — the company being *called*, which is not the customer's personal
information. Tagging it would beep a brand name in nearly every dialogue for
no privacy gain. `CARDINAL` is dropped too; it fires on "five gigabytes" and
"twelve month" far more than on a flat number.

---

## Finding F8 — the rule layer and the neural model are complementary, not redundant

The ablation, paired on 92 test dialogues:

| comparison | ΔF2 | 95% CI | significant |
|---|---|---|---|
| NER only vs rules only | +0.035 | [−0.003, +0.072] | **no** |
| rules + NER vs NER only | +0.049 | [+0.026, +0.077] | **yes** |
| rules + NER vs rules only | +0.083 | [+0.057, +0.111] | **yes** |
| rules + NER vs regex only | +0.401 | [+0.358, +0.445] | yes |

**Neither component is clearly better than the other alone, but each adds
significantly on top of the other.** Per-type shows why: on GIVEN_NAME the
rule layer gets 78%, the NER 69%, and together 85% — they recover different
names. A carefully-built rule layer is not made obsolete by an off-the-shelf
neural model, and vice versa.

## Finding F9 — landmark addresses are detected by nothing

LANDMARK recall: regex 0%, western list 4%, indian list 7%, rules 7%,
NER 4%, rules+NER **7%**. Across every level, "opposite the Forum Mall petrol
pump" and "behind the Ganesh temple" are essentially invisible — 28 of 268
test spans at near-zero.

§1 argues that Indian addresses "are often given by landmark rather than in a
fixed format". This is that claim, confirmed from the failure side: the
landmark-relative form defeats patterns, gazetteers and an OntoNotes NER
alike. ADDRESS proper reaches 61% once the NER is in; LANDMARK does not move.

## Finding F10 — the NER helps where F6 predicts, and not where it does not

Predicted before running: spaCy is OntoNotes-trained, so it should help on
LOCATION and little on Indian PERSON names.

| type | rules | NER | both |
|---|---|---|---|
| ADDRESS | 7% | **54%** | 61% |
| GIVEN_NAME | 78% | **69%** | 85% |
| SURNAME | 73% | 84% | 89% |

Confirmed on ADDRESS and GIVEN_NAME — the Indian gazetteer beats the Western
NER on Indian given names. SURNAME went the other way (84% vs 73%), which the
prediction did not anticipate.

---

## Day 3 (late) — the error-attribution experiment

## D28. Attribution is scored in character space, with gold spans projected
**Date:** 10 Nov · **Affects:** §5 headline experiment

Both conditions run the identical detector with the identical criterion; the
only difference is whether it is handed the reference transcript or Whisper's.
Gold spans are projected into ASR coordinates by token alignment
(`src/piiaudio/align.py`), so "did we find this item" means the same thing in
both. A span that cannot be projected was destroyed outright and counts as an
ASR-caused miss by construction.

Every item lands in exactly one of four cells. The first version of this
script reported overlapping categories and produced "detector never had it:
101%", which is how the fourth cell — **items the ASR made findable** — was
discovered.

## D29. `require_capital` stays on, because dev says so
**Date:** 10 Nov · **Affects:** method, more than score

Attribution surfaced names Whisper transcribed *perfectly* that we missed
anyway (`Das` → `das.`, `Antony` → `.antony`): the capitalisation filter
rejects them, since Whisper's casing is inconsistent and a leading period
makes a token start lowercase. We were charging our own bug to the ASR.

Relaxing the filter: **dev −0.003 F2, test +0.007 F2 (not significant)**. Dev
is where tuning happens, so it stays on. Had we looked at test first we would
have changed it for a gain that is not real. The effect is tiny either way;
the discipline is the point.

---

## Finding F11 — failures split almost exactly in half

Of the 67 test items the full system fails to redact:

| cause | share | n |
|---|---|---|
| speech recognition destroyed it | **52%** | 35 |
| detector could not find it anyway | **48%** | 32 |

Neither half of the pipeline dominates. Effort spent only on better detection
would address at most half the remaining error, and the same for a better ASR.

## Finding F12 — for structured identifiers, the ASR *helps*

The fourth cell of the attribution table: items found in Whisper's transcript
but **not** in the true one.

| type | lost to ASR | rescued by ASR |
|---|---|---|
| PAN | 0% | **62%** |
| AADHAAR | 15% | **50%** |
| PHONE | 29% | **46%** |
| SURNAME | 18% | 0% |
| EMAIL | 22% | 0% |
| GIVEN_NAME | 9% | 0% |

Spoken identifiers are *harder* to pattern-match than Whisper's normalised
output: `M W P H M eight one three three G` defeats a PAN pattern that
`MWPHM8133G` satisfies. A perfect transcript would not help here — it would
hurt.

**Caveat, stated plainly:** our Level 1 patterns are deliberately ASR-aware
(D16), so they under-fire on spoken text by design. The direction of the
effect is real and mechanistically explained, but its magnitude partly
reflects our pattern coverage, and a spoken-form pattern set would narrow it.
The open-class types (names, addresses, emails) carry no such confound — the
gazetteer and NER are text-agnostic — so the 9–22% ASR cost there is clean.
