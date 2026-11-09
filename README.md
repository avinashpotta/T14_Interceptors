# PII Detection and Redaction in Audio Streams

Team Interceptors (Team 14). Detects personally identifiable information in
spoken audio and overwrites it with a tone. Runs entirely locally — no audio
leaves the machine.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/render_corpus.py        # build audio + ground truth
.venv/bin/python scripts/demo.py data/audio/d001.wav out.wav
```

First run downloads the Whisper `small` model (~500 MB).

**Requirements:** macOS (uses the built-in `say` for corpus generation),
Python 3.11+. The detection and redaction pipeline itself is platform
independent — only dataset generation needs macOS.

## How it works

```
audio ──► faster-whisper ──► transcript + word timings
                                   │
                                   ├──► detectors (regex / namelist / Presidio / NER+rules)
                                   │         └── spans as CHARACTER offsets
                                   │
                            char_to_time bridge
                                   │
                                   ▼
                      overwrite those samples with a tone ──► redacted audio
```

Detectors never touch audio; they return character spans over a string. That
keeps them unit-testable, and it makes the error-attribution experiment a
matter of handing in a different string.

## The dataset trick

Dialogues are authored as alternating ordinary/PII segments and rendered one
segment at a time, so the concatenation offset of each PII segment *is* its
exact ground-truth timing. No forced alignment, and no risk of measuring the
aligner against itself. See `DECISIONS.md` D1.

Each dialogue yields spans in two coordinate systems — character offsets into
the reference transcript (for scoring detectors) and seconds into the audio
(for scoring whether the beep landed).

## Layout

| Path | What |
|---|---|
| `src/piiaudio/taxonomy.py` | PII ontology (§2 KR&R component) |
| `src/piiaudio/tts.py` | Corpus renderer + ground truth |
| `src/piiaudio/asr.py` | Whisper wrapper, word timings, text↔time bridge |
| `src/piiaudio/detect/` | The four comparison levels |
| `src/piiaudio/redact.py` | Span → overwritten samples |
| `data/scripts/` | Dialogue scripts (authored) |
| `data/audio/` | Rendered audio + `.gold.json` (generated, gitignored) |
| `DECISIONS.md` | Why everything is the way it is |

## Status

Day 1 of 6. End-to-end spine works on one dialogue: render → transcribe →
regex-detect → beep. Levels 2–4, scoring, and the corpus are next.
