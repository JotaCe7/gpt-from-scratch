"""
This module provides utility functions for text generation.

It includes implementations for different decoding strategies, such as
greedy decoding, to generate new text sequences from a trained
language model.
"""

import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
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