"""
Configuration file for Llama 3 model architecture.

This module contains all hyperparameters and constants used throughout the Llama 3
implementation. These values are based on the Llama 3 paper specifications.
"""

# Model dimension: The hidden size of the transformer model
# This is the size of the embedding vectors and the main dimension used throughout
# Value taken from Llama 3 Table 3
DIM = 4096

# Feed-forward network dimension: The hidden size of the position-wise feed-forward network
# This is larger than DIM to provide more capacity in the FFN layers
# Value taken from Llama 3 Table 3
FFN_DIM = 14336

# Number of transformer layers: The depth of the model
# Each layer contains an attention mechanism and a feed-forward network
# Value taken from Llama 3 Table 3
N_LAYERS = 32

# Number of attention heads: The number of parallel attention mechanisms
# Each head processes a different subspace of the input
# Value taken from Llama 3 Table 3
N_HEADS = 32

# Number of key-value heads: Used for Grouped Query Attention (GQA)
# GQA reduces the number of key-value heads while keeping query heads the same
# This improves inference speed and reduces memory usage
# Llama 3 uses 8 key-value heads with 32 query heads
N_KV_HEADS = 8

# Vocabulary size: The number of unique tokens in the tokenizer
# Llama 3 paper mentions 128K tokens, but the exact number from the tokenizer is 128256
VOCAB_SIZE = 128256

# Normalization epsilon: Small value added to prevent division by zero in RMSNorm
# This prevents numerical instability when normalizing very small values
# Value taken from Llama 3 code ModelArgs
NORM_EPS = 1e-5

# RoPE theta: The base frequency for Rotary Position Embeddings
# Llama 3 increases this to 500,000 (from 10,000 in earlier models) to extend context length
# Higher theta means slower frequency decay, allowing better position encoding for longer sequences
ROPE_THETA = 500000

# Maximum batch size: The maximum number of sequences that can be processed in parallel
# This is a practical limit based on available GPU memory
# If fewer examples are provided, the actual batch size will be smaller
MAX_BATCH_SIZE = 4

# Maximum sequence length: The maximum number of tokens in a single sequence
# This is a practical limit based on available GPU memory and model configuration
# Can be adjusted based on your hardware specifications
MAX_SEQ_LEN = 128

# Key-value head repetition factor: How many times each KV head is repeated
# This is calculated as N_HEADS // N_KV_HEADS to match the number of query heads
# For example, with 32 query heads and 8 KV heads, each KV head is repeated 4 times
N_KV_HEAD_REP = N_HEADS // N_KV_HEADS

# Head dimension: The dimension of each attention head
# This is calculated by dividing the model dimension by the number of heads
# Each head processes a DIM // N_HEADS dimensional subspace
HEAD_DIM = DIM // N_HEADS

