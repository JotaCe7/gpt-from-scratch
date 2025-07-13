"""
This module defines the final GPT model architecture by assembling the
individual components from the `layers` and `attention` modules.

It contains the `TransformerBlock`, which is the fundamental repeating
unit of the model, and the top-level `GPTModel` class that brings
everything together from input embeddings to final output logits.
"""

import torch
import torch.nn as nn

from .layers import LayerNorm, FeedForward
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """The complete Transformer Block, which is the core repeating unit
    of the GPT model"""

    def __init__(self, cfg: dict) -> None:
        """
        Initializes the TransformerBlock.
        
        Args:
            cfg (dict): The model configuration dictionary. Expected keys:
                        "emb_dim", "context_length", "n_heads",
                        "dropout_rate", "qkv_bias".
        """
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout_rate=cfg["dropout_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for a single transformer block.
        This uses a pre-layer normalization architecture.

        Args:
            x (torch.Tensor): The input tensor of shape
                              (batch_size, num_tokens, emb_dim).
        
        Returns:
            torch.Tensor: The output tensor of the same shape as
                          the input.
        """
        # Shortcut connection for the attention block
        shortcut = x
        x_norm = self.norm1(x)
        attn_output = self.attn(x_norm)
        x = shortcut + self.drop_shortcut(attn_output)

        # Shortcut connection for the feed-forward block
        shortcut = x
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = shortcut + self.drop_shortcut(ff_output)

        return x
    

class GPTModel(nn.Module):
    """The full GPT model architecture"""

    def __init__(self, cfg: dict) -> None:
        """
        Initializes the GPTModel.

        Args:
            cfg (dict): The model configuration dictionary containing all
                        hyperparameters for the model architecture.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the GPT model.

        Args:
            in_idx (torch.Tensor): The input tensor of token indices, with shape
                                   (batch_size, seq_len).
        
        Returns:
            torch.Tensor: The output logits tensor, with shape
                          (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits