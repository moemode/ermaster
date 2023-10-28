import os
import openai
import pickle
from pathlib import Path


PROMPT = """Do the two entity descriptions match?
Entity 1: '{e0}'
Entity 2: '{e1}'
Answer with 'Yes' if they do and 'No' if they do not.
"""

import json
from pathlib import Path


def prompts(pattern: str, file: Path):
    try:
        with open(file, "r") as json_file:
            data = json.load(json_file)
            for element in data:
                formatted_prompt = pattern.format(**element)
                print(formatted_prompt)
    except FileNotFoundError:
        print(f"File not found: {file}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Example usage:
file_path = Path("data/0_beer20.json")
prompts(PROMPT, file_path)


"""
openai.api_key = os.getenv("OPENAI_API_KEY")
c = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt="The size of an elephant is",
    max_tokens=7,
    logprobs=5,
    temperature=0,
)
"""
