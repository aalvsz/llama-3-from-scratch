"""
Rotary Position Embedding (RoPE) utilities.

RoPE is a positional encoding technique that encodes absolute positional information
with rotation matrices. It allows the model to understand the relative positions of
tokens in a sequence.

This module contains functions for precomputing RoPE frequencies and applying
rotary embeddings to query and key tensors.
"""

import torch
from config import HEAD_DIM, MAX_SEQ_LEN, ROPE_THETA


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the complex frequencies for Rotary Position Embeddings.
    
    This function creates a matrix of complex numbers that encode positional information.
    The frequencies are computed based on the dimension and theta parameter, then
    combined with position indices to create the full frequency matrix.
    
    Args:
        dim: The dimension of each attention head (HEAD_DIM)
        end: The maximum sequence length to precompute (typically MAX_SEQ_LEN * 2)
        theta: The base frequency parameter (ROPE_THETA for Llama 3)
    
    Returns:
        A complex tensor of shape (end, dim // 2) containing the precomputed frequencies
    """
    # Create frequency values: 1 / (theta^(2i/dim)) for i in [0, dim/2)
    # This creates a geometric progression of frequencies
    # The [: (dim // 2)] slice ensures we only take the first half of dimensions
    # Each frequency corresponds to a pair of dimensions in the embedding
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Create position indices: [0, 1, 2, ..., end-1]
    # These represent the positions in the sequence
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # Compute outer product: each position gets multiplied by each frequency
    # This creates a matrix where each row corresponds to a position and each column to a frequency
    # Shape: (end, dim // 2)
    freqs = torch.outer(t, freqs)
    
    # Convert to complex numbers: e^(i * freqs) = cos(freqs) + i * sin(freqs)
    # torch.polar creates complex numbers from magnitude and angle
    # We use magnitude=1 (torch.ones_like) and angle=freqs
    # This creates the rotation matrices for RoPE
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    
    # Return the complex frequency matrix
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting with input tensor.
    
    This function reshapes the frequency tensor so it can be broadcasted across
    the batch and head dimensions of the input tensor. The frequency tensor
    needs to match the sequence length and head dimension, but should broadcast
    across batch size and number of heads.
    
    Args:
        freqs_cis: The frequency tensor of shape (seq_len, head_dim // 2)
        x: The input tensor to broadcast with, typically of shape (batch, seq_len, heads, head_dim)
    
    Returns:
        Reshaped frequency tensor that can be broadcasted with x
    """
    # Get the number of dimensions in the input tensor
    ndim = x.ndim
    
    # Assert that the input has at least 2 dimensions (batch and sequence)
    assert 0 <= 1 < ndim
    
    # Assert that the frequency tensor matches the sequence length and head dimension
    # freqs_cis should be (seq_len, head_dim // 2)
    # x should be (batch, seq_len, ..., head_dim)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # Create a shape that broadcasts correctly:
    # - Keep the sequence dimension (index 1) as is
    # - Keep the last dimension (head_dim) as is
    # - Set all other dimensions to 1 for broadcasting
    # This allows the frequency tensor to broadcast across batch and head dimensions
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # Reshape the frequency tensor to the computed shape
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.
    
    This function applies the precomputed rotary embeddings to the query and key
    tensors. The rotation is applied in the complex plane, which allows the model
    to encode relative positional information.
    
    Args:
        xq: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        freqs_cis: Precomputed frequency tensor of shape (seq_len, head_dim // 2)
    
    Returns:
        Tuple of (rotated_query, rotated_key) with the same shapes as inputs
    """
    # Reshape query tensor to complex numbers
    # The head_dim is split into pairs: [d0, d1] -> (d0 + i*d1)
    # This converts the real tensor to a complex representation
    # Shape: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, num_heads, head_dim // 2) complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    
    # Reshape key tensor to complex numbers (same process as query)
    # Shape: (batch, seq_len, num_kv_heads, head_dim) -> (batch, seq_len, num_kv_heads, head_dim // 2) complex
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape frequency tensor for broadcasting
    # This ensures the frequency tensor can be multiplied with the query/key tensors
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotation: multiply complex numbers (rotation in complex plane)
    # This rotates each pair of dimensions by an angle proportional to the position
    # The rotation encodes the positional information
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    
    # Apply rotation to keys (same process)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # Convert back to the original dtype (e.g., bfloat16) to match input precision
    return xq_out.type_as(xq), xk_out.type_as(xk)

