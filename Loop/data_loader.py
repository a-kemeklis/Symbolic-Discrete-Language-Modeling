import pickle

from pathlib import Path

UNIQUE_TOKENS_PATH = Path(__file__).resolve().parent / "unique_tokens.pkl"

unique_tokens = None
with open(UNIQUE_TOKENS_PATH, "rb") as infile:
  unique_tokens = pickle.load(infile)

unique_tokens.sort(key=len, reverse=True)

EOS_TOKEN_ID = unique_tokens.index("CUSTOM_EOS_TOKEN")

tokens_trie_root = {}
max_tokens_length = None
for token_index in range(0, len(unique_tokens)):
  token = unique_tokens[token_index]
  curr_node = tokens_trie_root
  for char in token:
    if char not in curr_node:
      curr_node[char] = {}
    curr_node = curr_node[char]
  curr_node['FINISH'] = token_index
  if not max_tokens_length or len(token) > max_tokens_length:
    max_tokens_length = len(token)
assert max_tokens_length != None

def load_input_fast(input_line):
  lower_input_line = input_line.lower()
  tokens = []
  int_tokens = []
  curr_start_index = 0
  while curr_start_index < len(input_line):
    #print ("Check starting at:", curr_start_index)
    match_token = None
    match_int_token = None
    curr_node = tokens_trie_root
    for curr_traverse_index in range(curr_start_index, min(curr_start_index + max_tokens_length, len(input_line))):
      curr_char = lower_input_line[curr_traverse_index]
      if curr_char in curr_node:
        curr_node = curr_node[curr_char]
      else:
        break
      if 'FINISH' in curr_node:
        match_token = lower_input_line[curr_start_index:curr_traverse_index + 1]
        match_int_token = curr_node['FINISH']

    assert match_int_token != None and match_token != None, lower_input_line[curr_start_index:curr_start_index + max_tokens_length]
    tokens.append(match_token)
    int_tokens.append(match_int_token)
    #print ("Found token match:", match_token)
    curr_start_index += len(match_token)

  tokens.append("CUSTOM_EOS_TOKEN")
  int_tokens.append(EOS_TOKEN_ID)

  return tokens, int_tokens
