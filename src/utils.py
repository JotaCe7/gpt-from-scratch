"""
This module provides a collection of general-purpose helper functions
used throughout the `gpt-from-scratch` project, including data loading
and token conversion utilities.
"""

import os
import urllib.request

import torch

from .tokenizing import Tokenizer


def load_data(file_path: str, url: str = None) -> str:
    """
    Downloads a text file if it doesn't exist locally, then reads and return its content.

    Args:
        file_path (str): The local path where the file is stored or will be saved
        url (str): The URL of the text file to download if it doesn't exist locally.

    Returns:
        str: The content of the text file as string.
    """
    if not os.path.exists(file_path):
        print(f"Downloading data from {url}...")
        with urllib.request.urlopen(url) as resposne:
            text_data = resposne.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        print("Download complete.")
    else:
        print(f"File '{file_path}' already exists. Loading fomr disk...")
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        print("Load complete.")
    
    return text_data


def text_to_token_ids(text: str, tokenizer: Tokenizer) -> torch.Tensor:
    """
    Converts a string of text into a batch of token IDs.

    Args:
        text (str): The input text string.
        tokenizer (Tokenizer): The tokenizer instance.

    Returns:
        torch.Tensor: A 2D tensor of shape (1, num_tokens).
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Tokenizer) -> str:
    """
    Converts a batch of token IDs back into a string of text.

    Args:
        token_ids (torch.Tensor): A 2D tensor of shape (1, num_tokens).
        tokenizer (Tokenizer): The tokenizer instance.

    Returns:
        str: The decoded text string.
    """
    flat = token_ids.squeeze(0)  # Remove batch dimension
    return tokenizer.decode(flat.tolist())