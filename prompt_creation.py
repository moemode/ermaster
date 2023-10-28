import json
from pathlib import Path
from typing import List

PROMPT = """Do the two entity descriptions match?
Entity 1: '{e0}'
Entity 2: '{e1}'
Answer with 'Yes' if they do and 'No' if they do not.
"""


def prompts_targets(file: Path, pattern=PROMPT) -> List[str]:
    p = []
    t = []
    try:
        with open(file, "r") as json_file:
            data = json.load(json_file)
            for element in data:
                formatted_prompt = pattern.format(**element)
                print(formatted_prompt)
                p.append(formatted_prompt)
                t.append(element["t"])
        return p, t
    except FileNotFoundError:
        print(f"File not found: {file}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
