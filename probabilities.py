from regex import D
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import numpy as np
import torch
from typing import Tuple, Optional

def compute_all_transition_scores(
    model,
    scores: Tuple[torch.Tensor],
    normalize_logits: bool = False,
) -> torch.Tensor:
    """
    Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was
    used). This is a convenient method to quicky obtain the scores of the selected tokens at generation time.

    Parameters:
        sequences (`torch.LongTensor`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or
            shorter if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)`):
            Transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens Tuple of
            `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with
            each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
            generate-time.
        normalize_logits (`bool`, *optional*, defaults to `False`):
            Whether to normalize the logits (which, for legacy reasons, may be unnormalized).

    Return:
        `torch.Tensor`: A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)` containing
            the transition scores (logits)

    Examples:

    ```python
    >>> from transformers import GPT2Tokenizer, AutoModelForCausalLM
    >>> import numpy as np

    >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer.pad_token_id = tokenizer.eos_token_id
    >>> inputs = tokenizer(["Today is"], return_tensors="pt")

    >>> # Example 1: Print the scores for each token generated with Greedy Search
    >>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
    >>> transition_scores = model.compute_transition_scores(
    ...     outputs.sequences, outputs.scores, normalize_logits=True
    ... )
    >>> # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
    >>> # encoder-decoder models, like BART or T5.
    >>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    >>> generated_tokens = outputs.sequences[:, input_length:]
    >>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
    ...     # | token | token string | logits | probability
    ...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
    |   262 |  the     | -1.414 | 24.33%
    |  1110 |  day     | -2.609 | 7.36%
    |   618 |  when    | -2.010 | 13.40%
    |   356 |  we      | -1.859 | 15.58%
    |   460 |  can     | -2.508 | 8.14%

    >>> # Example 2: Reconstruct the sequence scores from Beam Search
    >>> outputs = model.generate(
    ...     **inputs,
    ...     max_new_tokens=5,
    ...     num_beams=4,
    ...     num_return_sequences=4,
    ...     return_dict_in_generate=True,
    ...     output_scores=True,
    ... )
    >>> transition_scores = model.compute_transition_scores(
    ...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
    ... )
    >>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
    >>> # Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
    >>> # use case, you might want to recompute it with `normalize_logits=True`.
    >>> output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
    >>> length_penalty = model.generation_config.length_penalty
    >>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
    >>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
    True
    ```"""
    # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
    # seq_len - input_length
    scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)
    # 3. Optionally normalize the logits (across the vocab dimension)
    if normalize_logits:
        scores = scores.reshape(-1, model.config.vocab_size, scores.shape[-1])
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        #scores = scores.reshape(-1, scores.shape[-1])
    return scores

def my_probs(scores):
    probs = torch.stack(scores, dim=1).softmax(-1)
    return probs

def top_k(probabilities, k, tokenizer):
    topk_values, topk_indices = torch.topk(probabilities, k=k, dim=2, largest=True)
    print(topk_values)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
prompt = """Product 1: ' Ubiquiti UniFi Protect Video Security  '
Product 2: ' Ubiquiti UniFi Video G3 Infrared Range Extender GBP 70.80'
Do the two product descriptions match? Answer with 'Yes' if they do and 'No' if they do not.
The answer is
"""
inputs = tokenizer([prompt], return_tensors="pt")

# Example 1: Print the scores for each token generated with Greedy Search
outputs = model.generate(**inputs, max_new_tokens=2, return_dict_in_generate=True, output_scores=True)
transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
# input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
# encoder-decoder models, like BART or T5.
input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | logits | probability
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

probs = compute_all_transition_scores(model, outputs.scores, True)
tk = top_k(my_probs(outputs.scores), 5, tokenizer)

# Example 2: Reconstruct the sequence scores from Beam Search
outputs = model.generate(
    **inputs,
    max_new_tokens=1,
    num_beams=4,
    num_return_sequences=1,
    return_dict_in_generate=True,
    output_scores=True,
)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
)
# If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
# Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
# use case, you might want to recompute it with `normalize_logits=True`.
output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
length_penalty = model.generation_config.length_penalty
reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
print(np.allclose(outputs.sequences_scores, reconstructed_scores))