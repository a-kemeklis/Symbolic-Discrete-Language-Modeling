from data_loader import load_input_fast
from story_loader import get_stories
from next_token_predictor import process_token

def run_on_input_data(input_data_file, print_each_story_performance=False):
  num_predicted_tokens = 0
  total_num_tokens = 0
  
  story_num = 0
  for story in get_stories(input_data_file):
    #if story_num % 100 == 0:
    #  print (f"Evaluating story #{story_num} with current accuracy {num_predicted_tokens/max(total_num_tokens, 1):0.8f}")
  
    tokens, _ = load_input_fast(story)
  
    story_num_predicted_tokens = 0
    embeddings_cache = []
    for token_index in range(0, len(tokens) - 1):
      token = tokens[token_index]
      predicted_token = process_token(embeddings_cache, token)
      predicted_correctly = False
      next_token = tokens[token_index + 1]
      if predicted_token == next_token:
        predicted_correctly = True
        story_num_predicted_tokens += 1
      if print_each_story_performance:
        print (f"#{token_index} ({'CORRECT' if predicted_correctly else 'INCORRECT'}): {predicted_token}, {next_token}")
    total_num_tokens += len(tokens)
    num_predicted_tokens += story_num_predicted_tokens
    if print_each_story_performance:
      print (f"This story has {len(tokens)} tokens with {story_num_predicted_tokens} predicted")
  
    story_num += 1

  #print (f"Total accuracy: {num_predicted_tokens/total_num_tokens:0.8f}")
  print (f"{num_predicted_tokens/total_num_tokens:0.8f}")
