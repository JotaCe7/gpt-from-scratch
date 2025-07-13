"""
This module provides utility functions for text generation.

It includes implementations for different decoding strategies, such as
greedy decoding, to generate new text sequences from a trained
language model.
"""

"""
This module provides utility functions for text generation.

It includes implementations for various decoding strategies, from simple
greedy decoding to more advanced methods like temperature scaling and
top-k filtering, to generate new text sequences from a trained
language model.
"""

import torch
import torch.nn as nn


def generate_text_simple(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """
    Generates new tokens autoregressively using greedy decoding.

    Args:
        model (torch.nn.Module): The GPT model instance.
        idx (torch.Tensor): The starting context of token IDs, of shape (batch, n_tokens).
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The maximum context length supported by the model.

    Returns:
        torch.Tensor: The input context plus the newly generated tokens.
    """
    for _ in range(max_new_tokens):
        # Crop current context if exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the model's predictions
        with torch.no_grad():
            logits = model(idx_cond) # batch, n_tokens, vocab_size
        
        # Focus on the logits for the very last token
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)
        
        # Get the token ID with the highest probability
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # Append the new token ID
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate(model: nn.Module, idx: torch.Tensor,
             max_new_tokens: int, context_size: int,
            temperature: float = 0.0,
            top_k: int | None = None,
            eos_id: int|None = None) -> torch.Tensor:
    """
    Generates new token sequences autoregressively with advanced sampling.

    Args:
        model (nn.Module): The GPT model instance.
        idx (torch.Tensor): The starting context of token IDs, of shape
            (batch_size, n_tokens).
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The maximum context length supported by the model.
        temperature (float): Controls randomness. A value of 0.0 means greedy
            decoding. Higher values increase randomness. Defaults to 0.0.
        top_k (int | None): If set, restricts sampling to the k most likely
            tokens. Defaults to None.
        eos_id (int | None): If set, generation stops when this token ID is
            produced. Defaults to None.

    Returns:
        torch.Tensor: The input context plus the newly generated tokens.
    """
     # Autoregressive generation loop
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # Get the model's predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on the logits for the very last token in the sequence
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device), logits)

        # Use probabilistic sampling with temperature
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Use greedy decoding if temperature is 0
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop if the end-of-sequence token is generated
        if idx_next == eos_id:
            break

        # Append the new token ID to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx