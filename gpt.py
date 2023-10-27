import os
import openai
import pickle

openai.api_key = os.getenv("OPENAI_API_KEY")
c = openai.Completion.create(
  model="gpt-3.5-turbo-instruct",
  prompt="The size of an elephant is",
  max_tokens=7,
  logprobs=5,
  temperature=0
)

# Save the object to a file
with open('my_object2.pkl', 'wb') as file:
    pickle.dump(c, file)

PROMPT = """Do the two entity descriptions match? Answer with 'Yes' if they do and 'No' if they do not.
Entity 1: {}
Entity 2: {}
"""