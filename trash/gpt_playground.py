import os
from timeit import default_timer as timer
import openai
from erllm.utils import retry_with_exponential_backoff


PRICE_PER_1K_TOKENS_PROMPT = 0.002
PRICE_PER_1K_TOKENS_COMPLETE = 0.002


def completions_with_backoff(**kwargs):
    start = timer()
    c = openai.Completion.create(**kwargs)
    return timer() - start, c


completions_with_backoff = retry_with_exponential_backoff(
    completions_with_backoff,
    initial_delay=60,
    exponential_base=1.1,
    max_retries=3,
    jitter=False,
)


model = "gpt-3.5-turbo-instruct"
model_params = dict(model=model, max_tokens=10, logprobs=5, temperature=0, seed=0)
openai.api_key = os.getenv("OAIST_KEY")
print(openai.api_key)
# print(completions_with_backoff(prompt="Hello, my name is", **model_params))
p = """Do the two entity descriptions refer to the same real-world entity? Answer with 'Yes' if they do and 'No' if they do not.
Entity 1: 'micromat diskstudio mac ) micromat 99.0'
Entity 2: 'micromat inc. diskstudio nan 43.36'
"""
genp = """Do the two entity descriptions refer to the same real-world entity? Answer with 'Yes' if they do and 'No' if they do not.
Entity 1: '{e0}'
Entity 2: '{e1}'"""

genp2 = """Do the two entity descriptions refer to the same real-world entity? Answer with ####Yes#### if they do and ####No#### if they do not.
Entity 1: '{e0}'
Entity 2: '{e1}'
Answer: ####"""

genp4 = """Do the two entity descriptions refer to the same real-world entity? Answer with ####Yes if they do and ####No if they do not.
Entity 1: '{e0}'
Entity 2: '{e1}'
####"""
