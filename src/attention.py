import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    """A self-attention module with a causal nask."""
    def __init__(self, d_in: int, d_out: int, context_length:int,
                 dropout_rate: float, qkv_bias = False) -> None:
        super().__init__()
        """
        Initializes the Causal Self-Attention layer.

        Args:
            d_in (int): Input embedding dimension.
            d_out (int): Output embedding dimension (for query, key, value).
            context_length (int): The maximum sequence length for the causal mask.
            dropout_rate (float): The dropout rate.
            qkv_bias (bool): Whether to include a bias term in the Q, K, V linear layers.
        """

        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.register_buffer('causal_mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for causal self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            torch.Tensor: Output context vector tensor of shape (batch_size, num_tokens, d_out).
        """
        batch_size, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Calculate similarity scores via the QK^T dot product
        attn_scores = queries @ keys.transpose(-2, -1)

        # Crop the mask to the size of the current input and apply it
        causal_mask = self.causal_mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill(causal_mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        return context_vec    


class MultiHeadAttention(nn.Module):
    """An efficient multi-head attention module with weight splitting."""
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout_rate: float, num_heads: int, qkv_bias: bool = False) -> None:
        """
        Initializes the Multi-Head Attention layer.

        Args:
            d_in (int): Input embedding dimension.
            d_out (int): Total output dimension for the module. Must be divisible by num_heads.
            context_length (int): The maximum sequence length for the causal mask.
            dropout_rate (float): The dropout rate.
            num_heads (int): The number of parallel attention heads.
            qkv_bias (bool): Whether to include a bias term in the Q, K, V linear layers.
        """
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.register_buffer("causal_mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            torch.Tensor: Output context vector tensor of shape (batch_size, num_tokens, d_out).
        """
        batch_size, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape for multi-head attention by splitting the d_out dimension
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpose to bring the 'num_heads' dimension forward for batched matrix multiplication
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculate similarity scores via the QK^T dot product
        attn_scores = queries @ keys.transpose(-2, -1)

        # Crop the mask to the size of the current input and apply it
        causal_mask = self.causal_mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(causal_mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads back into a single tensor
        # .contiguous() is required before .view() after a transpose
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec