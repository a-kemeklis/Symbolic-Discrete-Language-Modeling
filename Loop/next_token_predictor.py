import copy
from functools import lru_cache

from next_token_predictor_references.animals import animals
from next_token_predictor_references.body_parts import body_parts
from next_token_predictor_references.clothes import clothes
from next_token_predictor_references.colors import colors
from next_token_predictor_references.containers import containers
from next_token_predictor_references.directions import directions
from next_token_predictor_references.emotions_and_polarities import emotions_and_polarities
from next_token_predictor_references.foods import foods
from next_token_predictor_references.human_pronouns import human_pronouns
from next_token_predictor_references.inanimate_pronouns import inanimate_pronouns
from next_token_predictor_references.kin import kin
from next_token_predictor_references.motions import motions
from next_token_predictor_references.names_and_genders import names_and_genders
from next_token_predictor_references.names import names
from next_token_predictor_references.nature import nature
from next_token_predictor_references.numbers import numbers
from next_token_predictor_references.objects import objects
from next_token_predictor_references.people import people
from next_token_predictor_references.percieve import percieve
from next_token_predictor_references.places import places
from next_token_predictor_references.punctuation import punctuation
from next_token_predictor_references.sizes import sizes
from next_token_predictor_references.sounds import sounds
from next_token_predictor_references.speech import speech
from next_token_predictor_references.terminators import terminators
from next_token_predictor_references.thinking import thinking
from next_token_predictor_references.times import times
from next_token_predictor_references.toys import toys
from next_token_predictor_references.vehicles import vehicles
from next_token_predictor_references.weather import weather
from next_token_predictor_references.whitespace import whitespace

from next_token_predictor_references.morphology import getAllLemmas, getAllInflections

# Boolean categories: membership adds the category name to embedding["features"]
BOOLEAN_CATEGORIES = {
  "animals": animals,
  "body_parts": body_parts,
  "clothes": clothes,
  "colors": colors,
  "containers": containers,
  "directions": directions,
  "foods": foods,
  "human_pronouns": human_pronouns,
  "inanimate_pronouns": inanimate_pronouns,
  "kin": kin,
  "motions": motions,
  "names": names,
  "nature": nature,
  "numbers": numbers,
  "objects": objects,
  "people": people,
  "percieve": percieve,
  "places": places,
  "punctuation": punctuation,
  "sizes": sizes,
  "sounds": sounds,
  "speech": speech,
  "terminators": terminators,
  "thinking": thinking,
  "times": times,
  "toys": toys,
  "vehicles": vehicles,
  "weather": weather,
  "whitespace": whitespace,
}

# Valued categories: membership sets a named field to the category's value
VALUED_CATEGORIES = {
  "gender": names_and_genders,
  "polarity": emotions_and_polarities,
}

def _find_in_category(token, lemmas, category):
  # Match on the token itself first, then fall back to any of its base forms
  # so that e.g. "dogs" is found in animals via "dog"
  if token in category:
    return token
  for lemma in lemmas:
    if lemma in category:
      return lemma
  return None

@lru_cache(maxsize=None)
def _build_embedding_cached(token):
  embedding = {
    "raw_token": token,
    "features": set(),     # boolean tags, e.g. "animals", "whitespace", "word"
    "pos": set(),          # universal POS tags from morphology tables, e.g. "VERB"
    "lemmas": set(),       # base forms, e.g. "run" for "ran"
    "inflections": set(),  # Penn tags for this exact form, e.g. "VBD" for "ran"
    "gender": None,        # "m" / "f" from names_and_genders
    "polarity": None,      # "pos" / "neg" from emotions_and_polarities
  }

  if token.isalpha():
    embedding["features"].add("word")

  for universal_part_of_speech, lemmas in getAllLemmas(token).items():
    embedding["pos"].add(universal_part_of_speech)
    for lemma in lemmas:
      embedding["lemmas"].add(lemma)
      for tag, forms in getAllInflections(lemma, upos=universal_part_of_speech).items():
        if token in forms:
          embedding["inflections"].add(tag)

  for name, category in BOOLEAN_CATEGORIES.items():
    if _find_in_category(token, embedding["lemmas"], category) is not None:
      embedding["features"].add(name)

  for field, category in VALUED_CATEGORIES.items():
    match = _find_in_category(token, embedding["lemmas"], category)
    if match is not None:
      embedding[field] = category[match]

  return embedding

def build_embedding(token):
  # Embeddings are pure per-token so they are cached; hand out a copy so
  # downstream steps can mutate freely
  return copy.deepcopy(_build_embedding_cached(token))

def process_token(previous_embeddings, curr_token, verbose=False):
  this_token_embedding = build_embedding(curr_token)

  transformed_embedding = run_mha(previous_embeddings, this_token_embedding)

  predicted_next_embedding = predict_next_embedding(transformed_embedding)

  token_to_return = unembed(predicted_next_embedding)

  previous_embeddings.append(transformed_embedding)

  if verbose:
    print (transformed_embedding)
    print ("  -> predicted next:", predicted_next_embedding)

  return token_to_return


# ---------------------------------------------------------------------------
# run_mha: attention-style heads over previous token embeddings
# ---------------------------------------------------------------------------

def _summary(embedding):
  # A compact view of a selected previous token, so heads do not store whole
  # (recursively nested) embeddings
  return {
    "raw_token": embedding["raw_token"],
    "features": embedding["features"],
    "pos": embedding["pos"],
    "lemmas": embedding["lemmas"],
    "inflections": embedding["inflections"],
    "gender": embedding["gender"],
    "polarity": embedding["polarity"],
  }

def _always(_embedding, _cur=None):
  return True

def _has(summary, feature):
  return summary is not None and feature in summary["features"]

def _is_whitespace(e):
  return "whitespace" in e["features"]

PERSON_CATEGORIES = { "names", "people", "kin" }
# Concrete things a story refers back to; names are handled by latest_person
OBJECT_CATEGORIES = { "animals", "objects", "toys", "foods", "vehicles", "clothes",
                      "containers", "nature", "places" }

def _is_person(embedding):
  return bool(embedding["features"] & PERSON_CATEGORIES)

def _raw(summary):
  return summary["raw_token"] if summary is not None else None

def _same_prev_word(prev, cur):
  # prev is a word that directly followed the same word the current
  # (whitespace) token directly follows
  return "word" in prev["features"] and _has(prev["attn"]["prev_token"], "whitespace") \
    and _raw(prev["attn"]["prev_word"]) is not None \
    and _raw(prev["attn"]["prev_word"]) == _raw(cur["attn"]["prev_word"])

def _is_object(embedding):
  # Only nominal forms: lemma lookup also tags e.g. "could" (can) and "drank" (drink)
  return bool(embedding["features"] & OBJECT_CATEGORIES) \
    and bool(embedding["inflections"] & {"NN", "NNS"})

# Each head fires when `query` is true for the current token, then selects
# from the history (which never includes the current token):
#   offset   - the token `offset` places back
#   latest   - the most recent token for which `key` is true
#   earliest - the first token for which `key` is true
#   average  - all tokens for which `key` is true, aggregated
#   count    - how many tokens `key` is true for
# `key(prev, cur)` scores a previous token against the current one (K.Q), so a
# head can look for previous tokens that relate to the current token.
# `value(selected, distance)` reads what gets written to embedding["attn"][name],
# where distance is 1 for the immediately previous token.
# Heads run in order, so a later head may read attn fields written by earlier
# heads on both the current and previous tokens.
HEADS = [
  { "name": "position", "query": _always, "select": "count", "key": _always },
  { "name": "prev_token", "query": _always, "select": "offset", "offset": 1,
    "value": lambda sel, dist: _summary(sel) },
  { "name": "prev_prev_token", "query": _always, "select": "offset", "offset": 2,
    "value": lambda sel, dist: _summary(sel) },
  { "name": "tokens_since_terminator", "query": _always, "select": "latest",
    "key": lambda prev, cur: "terminators" in prev["features"],
    "value": lambda sel, dist: dist },
  { "name": "quote_count", "query": _always, "select": "count",
    "key": lambda prev, cur: prev["raw_token"] == '"' },
  { "name": "latest_name", "query": _always, "select": "latest",
    "key": lambda prev, cur: "names" in prev["features"],
    "value": lambda sel, dist: _summary(sel) },
  { "name": "latest_animal", "query": _always, "select": "latest",
    "key": lambda prev, cur: "animals" in prev["features"],
    "value": lambda sel, dist: _summary(sel) },
  { "name": "mean_polarity", "query": _always, "select": "average",
    "key": lambda prev, cur: prev["polarity"] is not None,
    "value": lambda sel, dist: 1 if sel["polarity"] == "pos" else -1 },
  # Copy-from-history heads: give whitespace/punctuation tokens direct access
  # to the surrounding words and to the entities currently in focus
  { "name": "prev_word", "query": _always, "select": "latest",
    "key": lambda prev, cur: "word" in prev["features"],
    "value": lambda sel, dist: _summary(sel) },
  # Two-hop copy: the prev_word that the previous word itself recorded
  { "name": "prev_prev_word", "query": _always, "select": "latest",
    "key": lambda prev, cur: "word" in prev["features"],
    "value": lambda sel, dist: sel["attn"]["prev_word"] },
  { "name": "latest_person", "query": _always, "select": "latest",
    "key": lambda prev, cur: _is_person(prev),
    "value": lambda sel, dist: _summary(sel) },
  { "name": "referenced_object", "query": _always, "select": "latest",
    "key": lambda prev, cur: _is_object(prev),
    "value": lambda sel, dist: _summary(sel) },
  # Induction heads: the word that followed the previous occurrence of the
  # word(s) we just read
  { "name": "induction_bigram", "query": _is_whitespace, "select": "latest",
    "key": lambda prev, cur: _same_prev_word(prev, cur)
      and _raw(prev["attn"]["prev_prev_word"]) is not None
      and _raw(prev["attn"]["prev_prev_word"]) == _raw(cur["attn"]["prev_prev_word"]),
    "value": lambda sel, dist: _summary(sel) },
  { "name": "induction_unigram", "query": _is_whitespace, "select": "latest",
    "key": lambda prev, cur: _same_prev_word(prev, cur),
    "value": lambda sel, dist: _summary(sel) },
  # People other than the one most recently mentioned
  { "name": "earliest_person", "query": _always, "select": "earliest",
    "key": lambda prev, cur: _is_person(prev),
    "value": lambda sel, dist: _summary(sel) },
  { "name": "other_person", "query": _always, "select": "latest",
    "key": lambda prev, cur: _is_person(prev) and prev["raw_token"] != _raw(cur["attn"]["latest_person"]),
    "value": lambda sel, dist: _summary(sel) },
  # Sentence state, carried forward token to token (reset at a terminator):
  # how many verbs / nouns the current sentence has had so far (history only)
  { "name": "sentence_verbs", "query": _always, "select": "offset", "offset": 1,
    "value": lambda sel, dist: 0 if "terminators" in sel["features"]
      else (sel["attn"]["sentence_verbs"] or 0) + (1 if "word" in sel["features"] and "VERB" in sel["pos"] and "be" not in sel["lemmas"] else 0) },
  { "name": "sentence_nouns", "query": _always, "select": "offset", "offset": 1,
    "value": lambda sel, dist: 0 if "terminators" in sel["features"]
      else (sel["attn"]["sentence_nouns"] or 0) + (1 if "word" in sel["features"] and sel["inflections"] & {"NN", "NNS"} else 0) },
  # Paragraph state: sentences ended since the last newline
  { "name": "sentences_since_newline", "query": _always, "select": "offset", "offset": 1,
    "value": lambda sel, dist: 0 if sel["raw_token"] == "\n"
      else (sel["attn"]["sentences_since_newline"] or 0) + (1 if "terminators" in sel["features"] else 0) },
  # People mentioned so far in the current sentence
  { "name": "persons_in_sentence", "query": _always, "select": "offset", "offset": 1,
    "value": lambda sel, dist: 0 if "terminators" in sel["features"]
      else (sel["attn"]["persons_in_sentence"] or 0) + (1 if _is_person(sel) else 0) },
  { "name": "persons_in_previous_sentence", "query": _always, "select": "offset", "offset": 1,
    "value": lambda sel, dist: sel["attn"]["persons_in_sentence"] if "terminators" in sel["features"]
      else sel["attn"]["persons_in_previous_sentence"] },
  # Majority vote over every previous continuation of the same word
  { "name": "induction_unigram_vote", "query": _is_whitespace, "select": "average",
    "key": lambda prev, cur: _same_prev_word(prev, cur),
    "value": lambda sel, dist: sel["raw_token"] },
  { "name": "latest_verb", "query": _always, "select": "latest",
    "key": lambda prev, cur: "word" in prev["features"] and "VERB" in prev["pos"] and "be" not in prev["lemmas"],
    "value": lambda sel, dist: _summary(sel) },
]

def _aggregate(values):
  if not values:
    return None
  if all(isinstance(v, bool) for v in values):
    return sum(values) / len(values)
  if all(isinstance(v, (int, float)) for v in values):
    return sum(values) / len(values)
  # Categorical: most common value
  return max(set(values), key=values.count)

def _run_head(previous_embeddings, embedding, head):
  select = head["select"]
  n = len(previous_embeddings)

  if select == "offset":
    index = n - head["offset"]
    if index < 0:
      return None
    return head["value"](previous_embeddings[index], head["offset"])

  if select == "latest":
    for index in range(n - 1, -1, -1):
      if head["key"](previous_embeddings[index], embedding):
        return head["value"](previous_embeddings[index], n - index)
    return None

  if select == "earliest":
    for index in range(0, n):
      if head["key"](previous_embeddings[index], embedding):
        return head["value"](previous_embeddings[index], n - index)
    return None

  if select == "average":
    values = [ head["value"](previous_embeddings[index], n - index)
               for index in range(0, n) if head["key"](previous_embeddings[index], embedding) ]
    return _aggregate(values)

  if select == "count":
    return sum(1 for prev in previous_embeddings if head["key"](prev, embedding))

  raise ValueError(f"Unknown head select mode: {select}")

def run_mha(previous_embeddings, embedding):
  embedding["attn"] = {}
  for head in HEADS:
    if head["query"](embedding):
      embedding["attn"][head["name"]] = _run_head(previous_embeddings, embedding, head)
    else:
      embedding["attn"][head["name"]] = None
  return embedding


# ---------------------------------------------------------------------------
# predict_next_embedding: rules mapping the current embedding to a partial
# target embedding for the next token
# ---------------------------------------------------------------------------

def _empty_target():
  return { "raw_token": None, "features": set(), "pos": set(), "lemmas": set(),
           "inflections": set(), "gender": None, "polarity": None }

def _is(summary, raw_token):
  return summary is not None and summary["raw_token"] == raw_token

def _lemma_is(summary, lemma):
  return summary is not None and lemma in summary["lemmas"]

# Ordered rules: the first one whose condition holds sets the target.
# Each rule is (condition(embedding) -> bool, fill(embedding, target)).
#
# Shares quoted in comments come from continuation_report.txt (8000 train
# stories, see count_continuations.py). Fixed-word tables are tiered around the
# induction heads: a table sits before an induction head only where its share
# beats that head's hit rate in the same context.

def _prev_word_is(e, raw_token):
  return _is(e["attn"]["prev_word"], raw_token)

def _prev_word_in(e, raw_tokens):
  prev_word = e["attn"]["prev_word"]
  return prev_word is not None and prev_word["raw_token"] in raw_tokens

def _after_word(e):
  # Whitespace directly following a word (so prev_word is what we just read,
  # not a word from before some punctuation)
  return _is_whitespace(e) and _has(e["attn"]["prev_token"], "word")

def _person_gender(e):
  latest_person = e["attn"]["latest_person"]
  return latest_person["gender"] if latest_person is not None else None

def _copy_or(e, head, fallback):
  value = e["attn"][head]
  return value["raw_token"] if value is not None else fallback

# --- story opener / closer -------------------------------------------------

def _rule_story_start(e):
  # Leading whitespace before the first word of the story
  prev = e["attn"]["prev_token"]
  return "whitespace" in e["features"] and e["attn"]["position"] <= 1 \
    and (prev is None or _has(prev, "whitespace"))
def _fill_story_start(e, t):
  t["raw_token"] = "once"

def _rule_after_once(e):
  return _is_whitespace(e) and _is(e["attn"]["prev_token"], "once")
def _fill_after_once(e, t):
  t["raw_token"] = "upon"

def _rule_after_upon(e):
  return _is_whitespace(e) and _is(e["attn"]["prev_token"], "upon")
def _fill_after_upon(e, t):
  t["raw_token"] = "a"

def _rule_after_upon_a(e):
  return _is_whitespace(e) and _is(e["attn"]["prev_token"], "a") \
    and _is(e["attn"]["prev_prev_token"], " ") and e["attn"]["position"] <= 6
def _fill_after_upon_a(e, t):
  t["raw_token"] = "time"

# Newline right after a sentence end is most often the end of the story (28%)
def _rule_story_end(e):
  return e["raw_token"] == "\n" and _has(e["attn"]["prev_token"], "terminators")
def _fill_story_end(e, t):
  t["raw_token"] = "CUSTOM_EOS_TOKEN"

# --- punctuation ------------------------------------------------------------

# Word tokens followed by punctuation rather than a space. Ordered; first
# match wins. Conditions read the sentence-state heads (sentence_verbs,
# tokens_since_terminator) and the quote state.
def _in_quote(e):
  return e["attn"]["quote_count"] % 2 == 1

def _sentence_verbs(e):
  return e["attn"]["sentence_verbs"] or 0

def _first_sentence(e):
  return e["attn"]["tokens_since_terminator"] is None

def _long_sentence(e):
  return (e["attn"]["tokens_since_terminator"] or 0) >= 24

ENDS_SENTENCE_CATEGORIES = { "places", "nature", "objects", "names" }

WORD_PUNCTUATION_RULES = [
  (lambda e: _prev_word_is(e, "to"), " "),                                      # 71%: nothing below applies after "to"
  (lambda e: _prev_word_is(e, "named"), "."),                                   # "named lily." 80%
  (lambda e: e["raw_token"] == "end", "."),                                     # "the end." 63%
  (lambda e: e["raw_token"] == "no" and _in_quote(e), ","),                     # 64%
  (lambda e: e["raw_token"] == "let", "'"),                                      # "let's" 74%
  (lambda e: e["raw_token"] == "yes", ","),                                      # 69%
  (lambda e: e["raw_token"] == "said", ","),                                     # 51%
  (lambda e: e["raw_token"] == "on" and _sentence_verbs(e) == 0, ","),           # 79%
  (lambda e: e["raw_token"] == "day" and _prev_word_is(e, "that"), " "),         # "that day " 77%
  (lambda e: e["raw_token"] == "day" and _sentence_verbs(e) >= 2, "."),          # 49%
  (lambda e: e["raw_token"] == "day", ","),                                      # "one day," 61%
  (lambda e: e["raw_token"] == "time" and _sentence_verbs(e) == 0, ","),         # "once upon a time," 59%
  (lambda e: "names" in e["features"] and _in_quote(e), ","),                    # '"hi, lily,' 35%
  (lambda e: "kin" in e["features"] and _in_quote(e), ","),                      # '"mom,' 40%
  (lambda e: e["raw_token"] == "friends" and _sentence_verbs(e) >= 2, "."),      # 58%
  (lambda e: "names" in e["features"] and _sentence_verbs(e) >= 2, "."),         # 71%
  (lambda e: bool(e["features"] & ENDS_SENTENCE_CATEGORIES) and _sentence_verbs(e) >= 2 and _long_sentence(e), "."),  # 52-61%
  (lambda e: "NN" in e["inflections"] and "JJ" not in e["inflections"] and _first_sentence(e) and _sentence_verbs(e) >= 2, "."),  # 61%
]
def _rule_word_punctuation(e):
  return "word" in e["features"] and any(condition(e) for condition, _ in WORD_PUNCTUATION_RULES)
def _fill_word_punctuation(e, t):
  for condition, punctuation in WORD_PUNCTUATION_RULES:
    if condition(e):
      t["raw_token"] = punctuation
      return

# Closing quote directly after a terminator inside dialogue (54%)
def _rule_close_quote(e):
  return "terminators" in e["features"] and e["attn"]["quote_count"] % 2 == 1
def _fill_close_quote(e, t):
  t["raw_token"] = '"'

def _rule_after_terminator(e):
  return "terminators" in e["features"] or e["raw_token"] == ","
def _fill_after_terminator(e, t):
  t["raw_token"] = " "

# Apostrophe: "lily's" (50%), "don't" (35%)
def _rule_apostrophe(e):
  return e["raw_token"] == "'"
def _fill_apostrophe(e, t):
  t["raw_token"] = "s"

# Quote mark: closing -> space (59%), opening -> dialogue usually starts "i" (13%)
def _rule_quote(e):
  return e["raw_token"] == '"'
def _fill_quote(e, t):
  t["raw_token"] = " " if e["attn"]["quote_count"] % 2 == 1 else "i"

# --- whitespace after punctuation -------------------------------------------

# Sentence start: inside dialogue "i" (17%); otherwise a pronoun matching the
# latest person in focus (she 29% / he 27%)
def _rule_sentence_start(e):
  return _is_whitespace(e) and _has(e["attn"]["prev_token"], "terminators")
def _fill_sentence_start(e, t):
  gender = _person_gender(e)
  persons_previous = e["attn"]["persons_in_previous_sentence"] or 0
  if e["attn"]["quote_count"] % 2 == 1:
    t["raw_token"] = "i"
  elif _lemma_is(e["attn"]["prev_word"], "say"):
    t["raw_token"] = '"'                                   # 'said. "' 58%
  elif gender == "f":
    t["raw_token"] = "she"                                 # 20-35%
  elif gender == "m":
    t["raw_token"] = "they" if persons_previous != 1 else "he"   # they 19-30% / he 32%
  elif _prev_word_is(e, "friends"):
    t["raw_token"] = "they"                                # 34%
  else:
    t["raw_token"] = "he"                                  # 15%

# 'said, "' / 'asked, "' (86%)
def _rule_said_comma_quote(e):
  return _is_whitespace(e) and _is(e["attn"]["prev_token"], ",") \
    and (_lemma_is(e["attn"]["prev_word"], "say") or _has(e["attn"]["prev_word"], "speech"))
def _fill_said_comma_quote(e, t):
  t["raw_token"] = '"'

# "once upon a time, there" (88%)
def _rule_time_comma(e):
  return _is_whitespace(e) and _is(e["attn"]["prev_token"], ",") and _prev_word_is(e, "time")
def _fill_time_comma(e, t):
  t["raw_token"] = "there"

# "one day, <person>" (latest_person 25%), else "she" (12%)
def _rule_day_comma(e):
  return _is_whitespace(e) and _is(e["attn"]["prev_token"], ",") and _prev_word_is(e, "day")
def _fill_day_comma(e, t):
  t["raw_token"] = _copy_or(e, "latest_person", "she")

# Inside dialogue, a comma often precedes the person addressed (other_person 12%)
def _rule_comma_in_quote(e):
  return _is_whitespace(e) and _is(e["attn"]["prev_token"], ",") and _in_quote(e) \
    and e["attn"]["other_person"] is not None
def _fill_comma_in_quote(e, t):
  t["raw_token"] = e["attn"]["other_person"]["raw_token"]

# Other whitespace after a comma: "the" after a time word (17%), else "but" (7%)
def _rule_after_comma(e):
  return _is_whitespace(e) and _is(e["attn"]["prev_token"], ",")
def _fill_after_comma(e, t):
  t["raw_token"] = "the" if _has(e["attn"]["prev_word"], "times") else "but"

# After a closing quote: the person being spoken to (other_person 20%),
# else a paragraph break (13%)
def _rule_after_closing_quote(e):
  return _is_whitespace(e) and _is(e["attn"]["prev_token"], '"') and e["attn"]["quote_count"] % 2 == 0
def _fill_after_closing_quote(e, t):
  t["raw_token"] = _copy_or(e, "other_person", "\n")

# --- whitespace after a word: tier 1 (beats bigram induction) ---------------

SUBJECT_PRONOUNS_SINGULAR = { "he", "she", "it", "i" }
SUBJECT_PRONOUNS_PLURAL = { "they", "we", "you" }

# (prev_prev_word, prev_word) -> next word
PAIR_NEXT_WORD_STRONG = {
  ("was", "so"): "happy",      # 34%
  ("was", "very"): "happy",    # 17%
  ("was", "a"): "little",      # 34%
  ("went", "to"): "the",       # 56%
  ("loved", "to"): "play",     # 40%
  ("liked", "to"): "play",     # 29%
  ("and", "they"): "all",      # 14%
  ("it", "was"): "a",          # 20%
  ("he", "was"): "so",         # 17%
  ("she", "was"): "so",        # 17%
}
def _rule_pair_next_word_strong(e):
  return _after_word(e) and (_raw(e["attn"]["prev_prev_word"]), _raw(e["attn"]["prev_word"])) in PAIR_NEXT_WORD_STRONG
def _fill_pair_next_word_strong(e, t):
  t["raw_token"] = PAIR_NEXT_WORD_STRONG[(_raw(e["attn"]["prev_prev_word"]), _raw(e["attn"]["prev_word"]))]

# Pair rules keyed on the previous two words' raw form, category or
# inflection: specs are "raw=x", "cat=x" or "infl=x"
def _matches(summary, spec):
  if summary is None:
    return False
  kind, _, value = spec.partition("=")
  if spec == "any":
    return True
  if kind == "raw":
    return summary["raw_token"] == value
  if kind == "cat":
    return value in summary["features"]
  if kind == "infl":
    tag, _, exclusion = value.partition("&not")
    return tag in summary["inflections"] and not (exclusion and exclusion in summary["inflections"])
  raise ValueError(spec)

def _pair_rule_next(e, table):
  ppw, pw = e["attn"]["prev_prev_word"], e["attn"]["prev_word"]
  for ppw_spec, pw_spec, next_word in table:
    if _matches(pw, pw_spec) and _matches(ppw, ppw_spec):
      return next_word
  return None

# "the <noun> was" when the sentence has exactly one (non-be) verb so far (20-24%)
def _rule_the_noun_was(e):
  prev_word = e["attn"]["prev_word"]
  return _after_word(e) and _is(e["attn"]["prev_prev_word"], "the") and _sentence_verbs(e) == 1 \
    and bool(prev_word["inflections"] & {"NN", "NNS", "VB", "VBP"}) and "JJ" not in prev_word["inflections"]
def _fill_the_noun_was(e, t):
  t["raw_token"] = "was"

PAIR_SPEC_NEXT_WORD_STRONG = [
  ("raw=do", "raw=you", "want"),          # 55%
  ("cat=names", "raw=she", "loved"),      # "...lily. she loved" 44%
  ("raw=so", "infl=RB", "fun"),           # "so much fun" 44%
  ("raw=little", "infl=NN", "was"),       # 27%
  ("cat=sizes", "cat=people", "was"),     # 27%
  ("cat=human_pronouns", "cat=kin", "said"), # 17%
  ("any", "cat=kin", "said"),             # 16%
  ("any", "cat=people", "was"),           # 12%
  ("raw=that", "cat=times", "on"),        # "from that day on" 77%
  ("raw=the", "cat=places", "and"),       # 32%
  ("raw=the", "cat=objects", "and"),      # 30%
  ("raw=saw", "raw=a", "big"),            # 31%
  ("cat=motions", "raw=to", "the"),       # 51%
  ("cat=motions", "cat=directions", "to"),# 26%
  ("cat=motions", "infl=RB", "the"),      # 31%
  ("infl=VBD", "raw=and", "said"),        # "smiled and said" 22%
  ("cat=names", "raw=was", "so"),         # 22%
  ("infl=NN", "raw=was", "so"),           # 16%
  ("cat=percieve", "raw=a", "big"),       # 21%
  ("raw=the", "infl=VB&notJJ", "and"),    # 20% (adjective readings go to induction, 44%)
  ("raw=the", "infl=VBP&notJJ", "and"),   # 20%
  ("raw=the", "cat=animals", "was"),      # 16%
]
def _rule_pair_spec_strong(e):
  return _after_word(e) and _pair_rule_next(e, PAIR_SPEC_NEXT_WORD_STRONG) is not None
def _fill_pair_spec_strong(e, t):
  t["raw_token"] = _pair_rule_next(e, PAIR_SPEC_NEXT_WORD_STRONG)

# prev_word -> next word
NEXT_WORD_STRONG = {
  "play": "with",   # 52%
  "wanted": "to",   # 89%
  "there": "was",   # 83%
  "one": "day",     # 79%
  "liked": "to",    # 73%
  "loved": "to",    # 65%
  "has": "a",       # 49%
  "found": "a",     # 48%
  "time": "there",  # 46%
  "at": "the",      # 46%
  "saw": "a",       # 45%
  "went": "to",     # 42%
  "we": "can",      # 25%
}
def _rule_next_word_strong(e):
  return _after_word(e) and _prev_word_in(e, NEXT_WORD_STRONG)
def _fill_next_word_strong(e, t):
  t["raw_token"] = NEXT_WORD_STRONG[_raw(e["attn"]["prev_word"])]

# "named <name>": the name depends on the gender already in focus
# (f: lily 61%, m: timmy 44%, none: max 11%)
def _rule_after_named(e):
  return _after_word(e) and _prev_word_is(e, "named")
def _fill_after_named(e, t):
  t["raw_token"] = { "f": "lily", "m": "timmy" }.get(_person_gender(e), "max")

def _rule_after_subject_pronoun(e):
  return _after_word(e) and _prev_word_in(e, SUBJECT_PRONOUNS_SINGULAR | SUBJECT_PRONOUNS_PLURAL)
def _fill_after_subject_pronoun(e, t):
  if _prev_word_is(e, "you"):
    t["raw_token"] = "are" if _in_quote(e) else "can"   # 13% / 12%
  elif _prev_word_is(e, "i"):
    t["raw_token"] = "can"                               # 9%
  elif _prev_word_is(e, "it") and _in_quote(e):
    t["raw_token"] = "is"                                # 25%
  elif _prev_word_in(e, SUBJECT_PRONOUNS_SINGULAR):
    t["raw_token"] = "was"
  else:
    t["raw_token"] = "were"

# --- induction: copy what followed the same two words ----------------------

def _rule_induction_bigram(e):
  return _after_word(e) and e["attn"]["induction_bigram"] is not None
def _fill_induction_bigram(e, t):
  t["raw_token"] = e["attn"]["induction_bigram"]["raw_token"]

# --- whitespace after a word: tier 2 (beats unigram induction) --------------

PAIR_NEXT_WORD = {
  ("wanted", "to"): "play",    # 10%
  ("in", "the"): "park",       # 13%
  ("and", "then"): "said",     # 13%
}

# Category fallbacks that beat unigram induction
CATEGORY_NEXT_WORD_MID = [
  ("names", "was"),      # 15%
  ("animals", "named"),  # 17% once "the <animal> was" is taken
]
def _rule_category_next_word_mid(e):
  return _after_word(e) and any(cat in e["attn"]["prev_word"]["features"] for cat, _ in CATEGORY_NEXT_WORD_MID)
def _fill_category_next_word_mid(e, t):
  features = e["attn"]["prev_word"]["features"]
  for category, next_word in CATEGORY_NEXT_WORD_MID:
    if category in features:
      t["raw_token"] = next_word
      return

# Dialogue: "thank you" (speech & in quote -> you 60%)
def _rule_speech_in_quote(e):
  return _after_word(e) and _in_quote(e) and _has(e["attn"]["prev_word"], "speech")
def _fill_speech_in_quote(e, t):
  t["raw_token"] = "you"
def _rule_pair_next_word(e):
  return _after_word(e) and (_raw(e["attn"]["prev_prev_word"]), _raw(e["attn"]["prev_word"])) in PAIR_NEXT_WORD
def _fill_pair_next_word(e, t):
  t["raw_token"] = PAIR_NEXT_WORD[(_raw(e["attn"]["prev_prev_word"]), _raw(e["attn"]["prev_word"]))]

NEXT_WORD = {
  ("wanted", "to"): "play",    # 10%
  ("in", "the"): "park",       # 13%
  ("and", "then"): "said",     # 13%
}

# Category fallbacks that beat unigram induction
CATEGORY_NEXT_WORD_MID = [
  ("names", "was"),      # 15%
  ("animals", "named"),  # 17% once "the <animal> was" is taken
]
def _rule_category_next_word_mid(e):
  return _after_word(e) and any(cat in e["attn"]["prev_word"]["features"] for cat, _ in CATEGORY_NEXT_WORD_MID)
def _fill_category_next_word_mid(e, t):
  features = e["attn"]["prev_word"]["features"]
  for category, next_word in CATEGORY_NEXT_WORD_MID:
    if category in features:
      t["raw_token"] = next_word
      return

# Dialogue: "thank you" (speech & in quote -> you 60%)
def _rule_speech_in_quote(e):
  return _after_word(e) and _in_quote(e) and _has(e["attn"]["prev_word"], "speech")
def _fill_speech_in_quote(e, t):
  t["raw_token"] = "you"
def _rule_pair_next_word(e):
  return _after_word(e) and (_raw(e["attn"]["prev_prev_word"]), _raw(e["attn"]["prev_word"])) in PAIR_NEXT_WORD
def _fill_pair_next_word(e, t):
  t["raw_token"] = PAIR_NEXT_WORD[(_raw(e["attn"]["prev_prev_word"]), _raw(e["attn"]["prev_word"]))]

PAIR_SPEC_NEXT_WORD_MID = [
  ("cat=directions", "raw=to", "the"),   # 31%
  ("cat=directions", "infl=RB", "the"),  # 30%
  ("infl=RB", "raw=to", "the"),          # 21%
  ("infl=NN", "raw=to", "the"),          # 13%
]
def _rule_pair_spec_mid(e):
  return _after_word(e) and _pair_rule_next(e, PAIR_SPEC_NEXT_WORD_MID) is not None
def _fill_pair_spec_mid(e, t):
  t["raw_token"] = _pair_rule_next(e, PAIR_SPEC_NEXT_WORD_MID)

PAIR_SPEC_NEXT_WORD_MID = [
  ("cat=thinking", "raw=to", "play"),    # 13%
  ("cat=thinking", "infl=RB", "play"),   # 11%
  ("cat=directions", "raw=and", "saw"),  # 12%
]
def _rule_pair_spec_mid(e):
  return _after_word(e) and _pair_rule_next(e, PAIR_SPEC_NEXT_WORD_MID) is not None
def _fill_pair_spec_mid(e, t):
  t["raw_token"] = _pair_rule_next(e, PAIR_SPEC_NEXT_WORD_MID)

# "so she" / "but he" / "when she": pronoun by the gender in focus (19-39%)
def _rule_conjunction_pronoun(e):
  return _after_word(e) and _prev_word_in(e, {"so", "but", "when"})
def _fill_conjunction_pronoun(e, t):
  t["raw_token"] = "she" if _person_gender(e) == "f" else "he"

NEXT_WORD = {
  "did": "not",      # 71%
  "who": "was",      # 18%
  "do": "you",       # 36%
  "are": "you",      # 31%
  "have": "a",       # 22%
  "in": "the",       # 58%
  "on": "the",       # 45%
  "back": "to",      # 38%
  "it": "was",       # 33%
  "happy": "and",    # 32%
  "out": "of",       # 31%
  "of": "the",       # 27%
  "up": "and",       # 27%
  "had": "a",        # 26%
  "came": "to",      # 26%
  "all": "the",      # 25%
  "looked": "at",    # 24%
  "was": "a",        # 21%
  "for": "a",        # 19%

  "can": "i",        # 15%
  "will": "be",      # 15%
  "very": "happy",   # 14%
  "is": "a",         # 11%
  "would": "be",     # 11%
  "they": "were",    # 10%
  "day": "on",       # 27%
  "a": "big",        # 10%
  "and": "the",      # 5%, induction after "and" is only 3%
  "to": "the",       # 9%
}
def _rule_next_word(e):
  return _after_word(e) and _prev_word_in(e, NEXT_WORD)
def _fill_next_word(e, t):
  t["raw_token"] = NEXT_WORD[_raw(e["attn"]["prev_word"])]

# "with her" / "with his" by the gender in focus (32% / 28%)
def _rule_after_with(e):
  return _after_word(e) and _prev_word_is(e, "with")
def _fill_after_with(e, t):
  t["raw_token"] = { "f": "her", "m": "his" }.get(_person_gender(e), "the")

# "the <latest animal>" once an animal has been introduced (21%)
def _rule_the_animal(e):
  return _after_word(e) and _prev_word_is(e, "the") and e["attn"]["latest_animal"] is not None
def _fill_the_animal(e, t):
  t["raw_token"] = e["attn"]["latest_animal"]["raw_token"]

# "<person> and <other person>"
def _rule_and_other_person(e):
  return _after_word(e) and _prev_word_is(e, "and") and _is_person(e["attn"]["prev_prev_word"] or {"features": set()}) \
    and e["attn"]["other_person"] is not None
def _fill_and_other_person(e, t):
  t["raw_token"] = e["attn"]["other_person"]["raw_token"]

# "ran to", "went to", ... (motions: to 19%)
def _rule_after_motion(e):
  prev_word = e["attn"]["prev_word"]
  return _after_word(e) and _has(prev_word, "motions") and "VBD" in prev_word["inflections"]
def _fill_after_motion(e, t):
  t["raw_token"] = "to"

# --- induction: copy what followed the same word ----------------------------

def _rule_induction_unigram(e):
  return _after_word(e) and e["attn"]["induction_unigram"] is not None
def _fill_induction_unigram(e, t):
  t["raw_token"] = e["attn"]["induction_unigram_vote"]   # vote 16% vs latest 15%

# --- whitespace after a word: tier 3 (fallbacks) ----------------------------

# "his/her/their mom" (14-16%). Copying other_person here scores worse than the
# fixed word once induction has taken the repeated cases.
POSSESSIVE_PERSON = { "his", "her", "their" }
def _rule_possessive_person(e):
  return _after_word(e) and _prev_word_in(e, POSSESSIVE_PERSON)
def _fill_possessive_person(e, t):
  t["raw_token"] = "mom"

# "that she" / "that he" (12-23%)
def _rule_after_that(e):
  return _after_word(e) and _prev_word_is(e, "that")
def _fill_after_that(e, t):
  t["raw_token"] = "she" if _person_gender(e) == "f" else "he"

PAIR_SPEC_NEXT_WORD_WEAK = [
  ("infl=JJ", "infl=NNS", "and"),        # 26%
  ("infl=JJ", "infl=VB", "and"),         # 25%
  ("infl=JJ", "infl=VBP", "and"),        # 25%
  ("infl=JJ", "infl=NN", "and"),         # 23%
  ("infl=JJ", "infl=JJ", "and"),         # 22%
  ("any", "cat=thinking", "to"),         # 22%
  ("cat=directions", "raw=to", "the"),   # 31%
  ("cat=directions", "infl=RB", "the"),  # 30%
  ("infl=RB", "raw=to", "the"),          # 21%
  ("infl=VBD", "infl=VBD", "and"),   # 32%
  ("infl=VBD", "infl=RB", "and"),    # 27%
  ("cat=kin", "infl=VBD", "and"),    # 26%
  ("cat=people", "infl=VBD", "and"), # 24%
  ("infl=NN", "infl=RB", "the"),     # 22%
  ("infl=VB", "infl=RB", "the"),     # 22%
  ("infl=NN", "infl=VBD", "and"),    # 19%
  ("raw=to", "infl=VB", "the"),      # 15%
  ("raw=to", "infl=VBP", "the"),     # 15%
  ("raw=to", "infl=NN", "the"),      # 14%
  ("infl=RB", "infl=VB", "the"),     # 13%
  ("infl=RB", "infl=VBP", "the"),    # 12%
]
def _rule_pair_spec_weak(e):
  return _after_word(e) and _pair_rule_next(e, PAIR_SPEC_NEXT_WORD_WEAK) is not None
def _fill_pair_spec_weak(e, t):
  t["raw_token"] = _pair_rule_next(e, PAIR_SPEC_NEXT_WORD_WEAK)

# Refer back to the object in focus after a back-referencing determiner (14%)
BACK_REFERENCE_DETERMINERS = { "the", "its", "my", "your", "this" }
def _rule_referenced_object(e):
  return _after_word(e) and e["attn"]["referenced_object"] is not None \
    and _prev_word_in(e, BACK_REFERENCE_DETERMINERS)
def _fill_referenced_object(e, t):
  t["raw_token"] = e["attn"]["referenced_object"]["raw_token"]

# "little girl" / "little boy" by gender in focus (57% / 28%)
def _rule_after_little(e):
  return _after_word(e) and _prev_word_is(e, "little")
def _fill_after_little(e, t):
  t["raw_token"] = "boy" if _person_gender(e) == "m" else "girl"

NEXT_WORD_WEAK = {
  "asked": "her",     # 29%
  "gave": "her",      # 25%
  "made": "a",        # 26%
  "put": "the",       # 24%
  "took": "the",      # 22%
  "this": "is",       # 22%

  "an": "idea",       # 21%
  "be": "careful",    # 14%

  "that": "day",      # 12%
  "could": "not",     # 11%
  "felt": "so",       # 10%
  "i": "can",         # 9%
  "some": "of",       # 7%
  "too": "late",      # 8%
  "and": "said",      # 6%
}
def _rule_next_word_weak(e):
  return _after_word(e) and _prev_word_in(e, NEXT_WORD_WEAK)
def _fill_next_word_weak(e, t):
  t["raw_token"] = NEXT_WORD_WEAK[_raw(e["attn"]["prev_word"])]

# Fallback by the previous word's category (first match wins)
CATEGORY_NEXT_WORD = [
  ("thinking", "to"),            # 38%
  ("sounds", "and"),             # 34%
  ("places", "and"),             # 24% in the residue
  ("body_parts", "and"),         # 31%
  ("objects", "and"),            # 29%
  ("nature", "and"),             # 28%
  ("foods", "and"),              # 24%
  ("sizes", "and"),              # 12% once "little" is taken
  ("toys", "and"),               # 21%
  ("directions", "the"),         # 31% in the residue
  ("percieve", "a"),             # 20%
  ("inanimate_pronouns", "was"), # 20%
  ("motions", "to"),             # 19%
  ("colors", "and"),             # 17%
  ("names", "was"),              # 15%
  ("weather", "was"),            # 14%
  ("kin", "said"),               # 13%
  ("animals", "was"),            # 12%
  ("speech", "her"),             # 11%
  ("people", "named"),           # 9%
  ("human_pronouns", "and"),     # 8%
]
def _rule_category_next_word(e):
  return _after_word(e) and any(cat in e["attn"]["prev_word"]["features"] for cat, _ in CATEGORY_NEXT_WORD)
def _fill_category_next_word(e, t):
  features = e["attn"]["prev_word"]["features"]
  for category, next_word in CATEGORY_NEXT_WORD:
    if category in features:
      t["raw_token"] = next_word
      return

# Fallback by the previous word's inflection (first match wins)
INFLECTION_NEXT_WORD = [
  ("VBN", "the"),  # 12%
  ("VBD", "to"),   # 12% (first verb: "and" 16%, handled in the fill)
  ("NN", "and"),   # 12%
  ("NNS", "and"),  # 13%
  ("JJ", "and"),   # 10%
  ("VB", "and"),   # 11%
  ("VBP", "and"),  # 11%
  ("VBG", "in"),   # 7%
  ("RB", "the"),   # 12%
]
def _rule_inflection_next_word(e):
  return _after_word(e) and any(tag in e["attn"]["prev_word"]["inflections"] for tag, _ in INFLECTION_NEXT_WORD)
def _fill_inflection_next_word(e, t):
  inflections = e["attn"]["prev_word"]["inflections"]
  for tag, next_word in INFLECTION_NEXT_WORD:
    if tag in inflections:
      if tag == "VBD" and _sentence_verbs(e) == 1:
        next_word = "and"
      t["raw_token"] = next_word
      return

# Last resort after a noun-like word: "other" (11%)
def _rule_after_noun_default(e):
  return _after_word(e) and "NOUN" in e["attn"]["prev_word"]["pos"]
def _fill_after_noun_default(e, t):
  t["raw_token"] = "other"

# Whitespace with nothing better: the most common word
def _rule_whitespace_default(e):
  return _is_whitespace(e)
def _fill_whitespace_default(e, t):
  t["raw_token"] = "the"

def _rule_word(e):
  return "word" in e["features"]
def _fill_word(e, t):
  t["raw_token"] = " "

RULES = [
  # opener / closer
  (_rule_story_start, _fill_story_start),
  (_rule_after_once, _fill_after_once),
  (_rule_after_upon, _fill_after_upon),
  (_rule_after_upon_a, _fill_after_upon_a),
  (_rule_story_end, _fill_story_end),
  # punctuation
  (_rule_word_punctuation, _fill_word_punctuation),
  (_rule_close_quote, _fill_close_quote),
  (_rule_after_terminator, _fill_after_terminator),
  (_rule_apostrophe, _fill_apostrophe),
  (_rule_quote, _fill_quote),
  # whitespace after punctuation
  (_rule_sentence_start, _fill_sentence_start),
  (_rule_said_comma_quote, _fill_said_comma_quote),
  (_rule_time_comma, _fill_time_comma),
  (_rule_day_comma, _fill_day_comma),
  (_rule_comma_in_quote, _fill_comma_in_quote),
  (_rule_after_comma, _fill_after_comma),
  (_rule_after_closing_quote, _fill_after_closing_quote),
  # whitespace after a word: tier 1
  (_rule_pair_next_word_strong, _fill_pair_next_word_strong),
  (_rule_the_noun_was, _fill_the_noun_was),
  (_rule_pair_spec_strong, _fill_pair_spec_strong),
  (_rule_next_word_strong, _fill_next_word_strong),
  (_rule_after_named, _fill_after_named),
  (_rule_induction_bigram, _fill_induction_bigram),
  (_rule_after_subject_pronoun, _fill_after_subject_pronoun),
  # tier 2
  (_rule_pair_next_word, _fill_pair_next_word),
  (_rule_pair_spec_mid, _fill_pair_spec_mid),
  (_rule_conjunction_pronoun, _fill_conjunction_pronoun),
  (_rule_speech_in_quote, _fill_speech_in_quote),
  (_rule_next_word, _fill_next_word),
  (_rule_category_next_word_mid, _fill_category_next_word_mid),
  (_rule_after_with, _fill_after_with),
  (_rule_the_animal, _fill_the_animal),
  (_rule_and_other_person, _fill_and_other_person),
  (_rule_after_motion, _fill_after_motion),
  (_rule_induction_unigram, _fill_induction_unigram),
  # tier 3
  (_rule_possessive_person, _fill_possessive_person),
  (_rule_after_that, _fill_after_that),
  (_rule_referenced_object, _fill_referenced_object),
  (_rule_after_little, _fill_after_little),
  (_rule_next_word_weak, _fill_next_word_weak),
  (_rule_pair_spec_weak, _fill_pair_spec_weak),
  (_rule_category_next_word, _fill_category_next_word),
  (_rule_inflection_next_word, _fill_inflection_next_word),
  (_rule_after_noun_default, _fill_after_noun_default),
  (_rule_whitespace_default, _fill_whitespace_default),
  (_rule_word, _fill_word),
]

def predict_next_embedding(embedding):
  target = _empty_target()
  for condition, fill in RULES:
    if condition(embedding):
      fill(embedding, target)
      break
  return target


# ---------------------------------------------------------------------------
# unembed: partial target embedding -> a concrete vocabulary token
# ---------------------------------------------------------------------------

import pickle
import random
from pathlib import Path

VOCAB_INDEX_PATH = Path("./vocab_index.pkl")
FALLBACK_TOKEN = " "

_unembed_random = random.Random(0)
_vocab_index = None

def _index_key(field, value):
  return (field, value)

def _build_vocab_index():
  # Inverse index: (field, value) -> set of vocab tokens having that value.
  # Built once over the whole vocabulary and pickled since it needs a
  # morphology-table pass over every token.
  from data_loader import unique_tokens
  index = {}
  for token in unique_tokens:
    if token == "CUSTOM_EOS_TOKEN":
      continue
    embedding = _build_embedding_cached(token)
    for field in ("features", "pos", "lemmas", "inflections"):
      for value in embedding[field]:
        index.setdefault(_index_key(field, value), set()).add(token)
    for field in ("gender", "polarity"):
      if embedding[field] is not None:
        index.setdefault(_index_key(field, embedding[field]), set()).add(token)
  return index

def _get_vocab_index():
  global _vocab_index
  if _vocab_index is None:
    if VOCAB_INDEX_PATH.exists():
      with open(VOCAB_INDEX_PATH, "rb") as infile:
        _vocab_index = pickle.load(infile)
    else:
      print ("Building vocab index (one time)...")
      _vocab_index = _build_vocab_index()
      with open(VOCAB_INDEX_PATH, "wb") as outfile:
        pickle.dump(_vocab_index, outfile)
  return _vocab_index

def _target_constraints(target):
  constraints = []
  for field in ("features", "pos", "lemmas", "inflections"):
    for value in target[field]:
      constraints.append(_index_key(field, value))
  for field in ("gender", "polarity"):
    if target[field] is not None:
      constraints.append(_index_key(field, target[field]))
  return constraints

def unembed(target):
  if target["raw_token"] is not None:
    return target["raw_token"]

  index = _get_vocab_index()
  candidate_sets = [ index.get(constraint, set()) for constraint in _target_constraints(target) ]
  if not candidate_sets:
    return FALLBACK_TOKEN

  # Most specific (smallest) constraint first; drop the least specific ones
  # until the intersection is non-empty
  candidate_sets.sort(key=len)
  while candidate_sets:
    candidates = set.intersection(*candidate_sets)
    if candidates:
      return _unembed_random.choice(sorted(candidates))
    candidate_sets.pop()
  return FALLBACK_TOKEN
