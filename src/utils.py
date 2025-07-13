"""
This module provides a collection of general-purpose helper functions
used throughout the `gpt-from-scratch` project, including data loading
and token conversion utilities.
"""

import os
import urllib
import urllib.request
from typing import Any

import numpy as np
import torch
import tensorflow as tf
from tqdm import tqdm

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


def download_file(
        url: str,
        destination: str,
        backup_url: str | None = None
) -> None:
    """
    Downloads a file with a progress bar, handling potential errors and
    offering a backup URL.

    Args:
        url (str): The primary URL to download the file from.
        destination (str): The local path to save the file.
        backup_url (str | None, optional): A backup URL to try if the
            primary URL fails. Defaults to None.
    """
    def _attempt_download(download_url: str) -> bool:
        """Nested function to handle the actual download logic."""
        with urllib.request.urlopen(download_url) as response:
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True

            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(download_url)
            with tqdm(
                total=file_size, unit="iB", unit_scale=True,
                desc=progress_bar_description
            ) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # If both attempts fail
        error_message = f"Failed to download from both primary URL ({url})"
        if backup_url:
            error_message += f" and backup URL: {backup_url}"
        error_message += "\nCheck your internet connection or the file availability.\n" + \
                "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        print(error_message)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: dict[str, Any]):
    """
    Loads GPT-2 parameters from a TensorFlow checkpoint into a nested
    Python dictionary.

    Args:
        ckpt_path (str): The path to the TensorFlow checkpoint file.
        settings (dict[str, Any]): The model's configuration dictionary,
            used to structure the output dictionary.

    Returns:
        dict[str, Any]: A nested dictionary containing the model's weights.
    """
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params