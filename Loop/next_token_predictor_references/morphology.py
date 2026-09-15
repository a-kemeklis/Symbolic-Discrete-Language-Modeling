# Dependency-free replacement for lemminflect's dictionary lookups.
#
# lemminflect.getAllLemmas / getAllInflections are pure dictionary lookups
# (no model, no OOV rules), so their tables were frozen once into
# morphology_data.json.gz and are served from here with identical semantics:
#   - lookup is case-insensitive, results mirror the input's caps style
#   - upos="PROPN" is mapped to NOUN with a capitalised key
#   - upos filtering of results works the same way
# Only the standard library is needed.

import gzip
import json
from pathlib import Path

_DATA_PATH = Path(__file__).with_name("morphology_data.json.gz")

DICT_UPOS_TYPES = ["NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV"]

_lemma_lu = None
_infl_lu = None

def _load():
  global _lemma_lu, _infl_lu
  if _lemma_lu is None:
    with gzip.open(_DATA_PATH, "rt", encoding="utf-8") as infile:
      data = json.load(infile)
    _lemma_lu = data["lemma_lu"]
    _infl_lu = data["infl_lu"]

def _caps_style(word):
  if word.isupper():
    return "all_upper"
  if word and word[0].isupper():
    return "first_upper"
  return "lower"

def _apply_caps(word, style):
  if style == "all_upper":
    return word.upper()
  if style == "first_upper":
    return word.capitalize()
  return word.lower()

def _upos_to_tags(upos):
  upos = upos.upper()
  if upos in ("VERB", "AUX"):
    return ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD")
  if upos == "ADJ":
    return ("JJ", "JJR", "JJS")
  if upos == "ADV":
    return ("RB", "RBR", "RBS")
  if upos == "NOUN":
    return ("NN", "NNS", "NNP", "NNPS")
  return ()

def getAllLemmas(word, upos=None):
  # -> { upos: (lemma, ...) }
  if upos is not None and upos not in DICT_UPOS_TYPES:
    return {}
  _load()
  style = _caps_style(word)
  word = word.lower()
  if upos == "PROPN":
    word = word.capitalize()
    upos = "NOUN"
  lemmas = _lemma_lu.get(word, {})
  if upos:
    lemmas = {k: v for k, v in lemmas.items() if k == upos}
  return {k: tuple(_apply_caps(w, style) for w in v) for k, v in lemmas.items()}

def getAllInflections(lemma, upos=None):
  # -> { penn_tag: (form, ...) }
  if upos is not None and upos not in DICT_UPOS_TYPES:
    return {}
  _load()
  style = _caps_style(lemma)
  lemma = lemma.lower()
  if upos == "PROPN":
    lemma = lemma.capitalize()
    upos = "NOUN"
  forms = _infl_lu.get(lemma, {})
  if upos is not None:
    allowed = _upos_to_tags(upos)
    forms = {k: v for k, v in forms.items() if k in allowed}
  return {k: tuple(_apply_caps(w, style) for w in v) for k, v in forms.items()}
