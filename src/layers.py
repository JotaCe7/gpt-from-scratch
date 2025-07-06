"""
This module contains fundamental neural network layers like LayerNorm, GELU, 
and FeedForward, which are used as building blocks for the transformer model.
"""

import math
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """A custom Layer Normalization module."""

    def __init__(self, emb_dim: int) -> None:
        """
        Initializes the LayerNorm module.

        Args:
            emb_dim (int): The dimension of the embedding vector to be normalized.
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies layer normalization to the input tensor along the last dimension.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The normalized tensor, with the same shape as the input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    This is an approximation of GELU used in the original GPT-2 model.
    """
    def __init__(self) -> None:
        """Initializes the GELU activation layer."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the GELU activation function element-wise.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with GELU applied.
        """
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """ A simple feed-forward network for the transformer block."""

    def __init__(self, cfg: dict) -> None:
        """
        Initializes the FeedForward network.

        Args:
            cfg (dict): The model configuration dictionary, which is expected
                        to contain the embedding dimension "emb_dim".
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the feed-forward layers.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor, with the same shame as the input.
        """
        return self.layers(x)
 