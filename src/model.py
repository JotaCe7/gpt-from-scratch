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
            qkv_bias=cfg["qkv_bias"]
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