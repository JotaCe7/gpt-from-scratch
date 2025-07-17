"""
This module defines the abstract protocols for tokenization tasks.

It provides a standard `Tokenizer` interface to ensure compatibility
between different tokenizer implementations used throughout the project.
"""

from typing import Protocol, Any


class Tokenizer(Protocol):
    """Defines the interface for a generic tokenizer."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """Encodes a string into a list of token IDs.

        Args:
            text (str): The input text string to be tokenized.
            **kwargs: Additional keyword arguments for compatibility with
                different tokenizer libraries.

        Returns:
            list[int]: A list of integer token IDs representing the text.
        """
        ...

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string.

        Args:
            ids (list[int]): The list of integer token IDs to be decoded.

        Returns:
            str: The reconstructed text string.
        """
        ...