# PII Detection and Redaction in Audio Streams

Team Interceptors (Team 14). Detects personally identifiable information in
spoken audio and overwrites it with a tone. Runs entirely locally — no audio
leaves the machine.

**What this project actually contributes is the evaluation.** A working
pipeline is the easy half; knowing *where* such a pipeline loses accuracy is
the hard half, and §5.5 of the report answers it: of the PII we fail to
redact, 50% was destroyed by speech recognition and 50% was never found by the
detector. Neither half dominates.

## Headline results

Held-out test split — 92 dialogues, 268 PII spans, bootstrap intervals over
dialogues.

| approach | recall | precision | **F2** |
|---|---|---|---|
| regex only | 0.280 | 0.866 | 0.324 |
| + Western name list | 0.362 | 0.824 | 0.408 |
| Presidio, off the shelf | 0.478 | 0.671 | 0.507 |
| + Indian name list | 0.563 | 0.797 | 0.599 |
| rule layer, no NER | 0.612 | 0.790 | 0.641 |
| neural NER only | 0.660 | 0.751 | 0.677 |
| **rule layer + NER** | **0.720** | 0.747 | **0.725** |

Recall 0.72 means roughly **one PII item in four still plays in the clear**.
This is a measurement instrument, not a deployable privacy tool, and the
report says so.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_md

.venv/bin/python scripts/demo.py input.wav output.wav       # redact a file
.venv/bin/python tests/test_core.py                          # 19 regression tests
```

First run downloads Whisper `small` (~500 MB). `--level 1` / `2` / `4` selects
the detection stack.

To rebuild everything from scratch:

```bash
.venv/bin/python scripts/build_corpus.py        # 38 templates -> 152 dialogues
.venv/bin/python scripts/render_corpus.py       # render audio + ground truth
.venv/bin/python scripts/transcribe_corpus.py   # transcribe once, cache
.venv/bin/python scripts/evaluate_levels.py     # the comparison table
.venv/bin/python scripts/attribution.py         # where the errors come from
```

**Requirements:** Python 3.11+. Corpus *generation* needs macOS (built-in
`say`); detection and redaction are platform independent.

## How it works

```
audio ──► faster-whisper ──► transcript + word timings
                                   │
                                   ├──► detectors ──► CHARACTER spans
                                   │     regex / gazetteer+rules / NER / Presidio
                                   │
                            char_to_time bridge
                                   │
                                   ▼
                      overwrite those samples with a tone ──► redacted audio
```

Detectors never touch audio — they return character spans over a string. That
keeps them unit-testable, makes all four comparison levels directly
comparable, and turns the error-attribution experiment into a matter of
passing in a different string.

## The ideas worth knowing

**Ground truth is exact by construction.** Dialogues are authored as
alternating ordinary/PII segments and rendered one segment at a time, so the
concatenation offset of each PII segment *is* its true start time. No forced
alignment, and no risk of measuring the aligner against itself.

**Splits are by template, never by dialogue.** Dialogues from one template
share phrasing; splitting at the dialogue level would put near-identical
sentences in both halves.

**Recall is span-level, precision is duration-level.** Span-counted precision
scores a detector that beeps the *entire recording* at a perfect 1.00.

**Comparisons are paired.** Checking whether two independently-computed
confidence intervals overlap is not a test of whether the difference is zero.

## Layout

| path | what |
|---|---|
| `src/piiaudio/taxonomy.py` | PII ontology (§2 KR&R component) |
| `src/piiaudio/tts.py` | corpus renderer + exact ground truth |
| `src/piiaudio/verbalize.py` | identifier generation and speaking styles |
| `src/piiaudio/asr.py` | Whisper wrapper, word timings, text↔time bridge |
| `src/piiaudio/align.py` | reference↔hypothesis token alignment |
| `src/piiaudio/gazetteer.py` | Western / Indian name lists + stoplist |
| `src/piiaudio/detect/` | the four comparison levels |
| `src/piiaudio/evaluate.py` | scoring, bootstrap, paired comparison |
| `src/piiaudio/redact.py` | span → overwritten samples |
| `src/piiaudio/augment.py` | noise, echo, telephony degradation |
| `src/piiaudio/streaming.py` | delay-buffer streaming redaction |
| `scripts/` | corpus build, evaluation, experiments, figures |
| `docs/REPORT.md` | the report |
| `docs/WALKTHROUGH.md` | how it works and why — written for reading cold |
| `DECISIONS.md` | every non-obvious choice, with its rejected alternative |

## Documentation

- **`DECISIONS.md`** — 34 decisions and 18 findings. Each records what was
  rejected and why. Read this before changing anything.
- **`docs/WALKTHROUGH.md`** — mechanism-level explanation plus an interview
  Q&A section.
- **`docs/REPORT.md`** — the submitted report, with figures in
  `docs/figures/`.

## Status

Complete. Corpus, four-level comparison, ablation, error attribution,
degradation study, human-speech validation, streaming analysis, edge cases,
19 regression tests, report and figures.
