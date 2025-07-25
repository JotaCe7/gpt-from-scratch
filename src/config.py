GPT_CONFIG_124M = {
    "vocab_size": 50257,        # Vocabulary size
    "context_length": 1024,      # Context length
    "emb_dim": 768,            # Embedding dimension
    "n_heads": 12,               # Number of attention heads
    "n_layers": 12,              # Number of layers
    "dropout_rate": 0.1,         # Dropout rate
    "qkv_bias": False            # Query-Key-Value bias in attention
}

GPT_CONFIG_124M_SMALL = {
    "vocab_size": 50257,        # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,            # Embedding dimension
    "n_heads": 12,               # Number of attention heads
    "n_layers": 12,              # Number of layers
    "dropout_rate": 0.1,         # Dropout rate
    "qkv_bias": False            # Query-Key-Value bias in attention
}