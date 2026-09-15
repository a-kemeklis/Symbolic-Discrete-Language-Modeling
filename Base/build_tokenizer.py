import copy
import pickle
import re
from tqdm import tqdm

unique_tokens = []

punctuation_tokens = [ " ", "-", ".", "'", '"', "!", "?", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "=", "+", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ":", ";", ",", "/", "\\", "{", "}", "|", "`", "~", "…", "\t", "\n", "[", "]", "<", ">" ]

delimiters = copy.deepcopy(punctuation_tokens)
split_pattern = "|".join(map(re.escape, delimiters))

unique_tokens += [ "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " " ]
unique_tokens += punctuation_tokens
unique_tokens.append("CUSTOM_EOS_TOKEN")

#with open("./tokenizer_log_tmp", "w") as outfile:
for line in tqdm(open("../Data/TinyStories-train.txt", "r").readlines()):
  line = line.encode("ascii", "ignore").decode("ascii").replace("—", "-")
  for part in re.split(split_pattern, line):
    token = "".join([char for char in part.strip().lower() if char.isalpha()])
    if token and token not in unique_tokens:
      unique_tokens.append(token)
      #outfile.write(token + "\n")
print (unique_tokens)

print ("Found", len(unique_tokens), "tokens, pickling the unique_tokens list")
with open("unique_tokens.pkl", "wb") as savefile:
  pickle.dump(unique_tokens, savefile)
