from tqdm import tqdm

from data_loader import load_input_fast

def print_tokenized_rows(infile):
  infile_lines = open(infile, "r").readlines()

  line_num = 0
  
  current_story = ""
  for line in tqdm(infile_lines):
    line = line.encode("ascii", "ignore").decode("ascii").replace("—", "-")
  
    if line == "<|endoftext|>\n":
      #print (current_story)
      tokens, _ = load_input_fast(current_story)
      print ("|".join(tokens))

      current_story = ""
    else:
      current_story += line

    line_num += 1
    if line_num == 500_000:
      break

print_tokenized_rows("../Data/TinyStories-train.txt")
