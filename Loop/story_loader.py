def get_stories(story_file):
  with open(story_file, "r") as infile:
    current_story = ""
    for line in infile:
      line = line.encode("ascii", "ignore").decode("ascii").replace("—", "-")
    
      if line == "<|endoftext|>\n":
        #print (current_story)
        yield current_story

        current_story = ""
      else:
        current_story += line
