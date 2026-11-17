# Third-party data and attribution

## Names Corpus (Western name lists)

`data/gazetteer/names_western.zip` and `data/gazetteer/names/` contain the
Names Corpus, Version 1.3 (1994-03-29), Copyright (C) 1991 Mark Kantrowitz,
with additions by Bill Ross.

The authors permit use for any purpose provided credit is given in published
work and the accompanying README is redistributed with the lists. That README
is kept at `data/gazetteer/names/README`. The lists are not in the public domain.

Source: http://www-2.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/

## Other dependencies

Speech recognition uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and OpenAI's Whisper model weights. Named-entity recognition uses
[spaCy](https://spacy.io/) and [Microsoft Presidio](https://github.com/microsoft/presidio)
is used as a comparison baseline. Each is distributed under its own licence.
