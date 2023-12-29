from typing import Dict, Iterable
import openai
import os
from erllm.llm_matcher.prompts import Prompt
from erllm.utils import retry_with_exponential_backoff, timed

openai.api_key = os.getenv("OAIST_KEY")


def get_chat_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=None,
) -> openai.ChatCompletion:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools
    completion = openai.ChatCompletion.create(**params)
    return completion


timed_chat_completion = timed(get_chat_completion)
retry_timed_chat_completion = retry_with_exponential_backoff(
    timed_chat_completion,
    initial_delay=60,
    exponential_base=1.1,
    max_retries=3,
    jitter=False,
)


def get_chat_completions(prompts: list[Prompt], model_params: Dict) -> Iterable[dict]:
    """
    Generate OpenAI completions for a list of prompts.
    Makes one call to the OpenAI API for each prompt.

    This function takes a list of prompts, requests OpenAI completions for each prompt,
    and yields a dictionary containing relevant information for each completion.

    Parameters:
    prompts (list[Prompt]): A list of Prompt objects containing prompt information.
    model_params (Dict): Model parameters to be passed to `completions_with_backoff`.

    Yields:
    dict: A dictionary containing information about the completion for each prompt,
          including IDs, prompt text, truth, completion details, time spent, and token counts.
    """
    for p in prompts:
        # create messages array from prompt_string
        messages = [{"role": "user", "content": p.prompt_string}]
        time_spent, r = retry_timed_chat_completion(messages=messages, **model_params)
        yield {
            "id0": p.id0,
            "id1": p.id1,
            "p": p.prompt_string,
            "t": p.truth,
            "c": r.choices[0],
            "d": time_spent,
            "i": r["usage"]["prompt_tokens"],
            "o": r["usage"]["completion_tokens"],
        }


if __name__ == "__main__":
    CLASSIFICATION_PROMPT = """You will be given a headline of a news article.
    Classify the article into one of the following categories: Technology, Politics, Sports, and Art.
    Return only the name of the category, and nothing else.
    MAKE SURE your output is one of the four categories stated.
    Article headline: {headline}"""
    headlines = [
        "Tech Giant Unveils Latest Smartphone Model with Advanced Photo-Editing Features.",
        "Local Mayor Launches Initiative to Enhance Urban Public Transport.",
        "Tennis Champion Showcases Hidden Talents in Symphony Orchestra Debut",
    ]

    for headline in headlines:
        print(f"\nHeadline: {headline}")
        API_RESPONSE = get_chat_completion(
            [
                {
                    "role": "user",
                    "content": CLASSIFICATION_PROMPT.format(headline=headline),
                }
            ],
            model="gpt-4",
        )
        print(f"Category: {API_RESPONSE.choices[0].message.content}\n")
