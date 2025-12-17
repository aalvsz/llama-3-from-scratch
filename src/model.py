"""
Llama 3 model architecture implementation.

This module contains all the neural network components of the Llama 3 transformer model:
- RMSNorm: Root Mean Square Layer Normalization
- FeedForward: Position-wise feed-forward network with SwiGLU activation
- Attention: Grouped Query Attention (GQA) with KV cache
- TransformerBlock: A single transformer layer combining attention and FFN
- Transformer: The complete model architecture
"""

import torch
from torch import nn
from torch.nn import functional as F

from .config import (
    DIM,
    FFN_DIM,
    N_LAYERS,
    N_HEADS,
    N_KV_HEADS,
    HEAD_DIM,
    VOCAB_SIZE,
    NORM_EPS,
    MAX_BATCH_SIZE,
    MAX_SEQ_LEN,
    N_KV_HEAD_REP,
    ROPE_THETA,
)
from .rope import precompute_freqs_cis, apply_rotary_emb


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is a variant of layer normalization that normalizes by the root mean square
    of the input, rather than the mean and variance. It's more efficient and has been
    shown to work well in transformer models.
    
    Llama 3 uses RMSNorm for all normalization operations.
    """
    
    def __init__(self, dim: int, norm_eps: float):
        """
        Initialize RMSNorm layer.
        
        Args:
            dim: The dimension of the input tensor (DIM)
            norm_eps: Small epsilon value to prevent division by zero (NORM_EPS)
        """
        # Call parent class constructor
        super().__init__()
        
        # Store the epsilon value for numerical stability
        self.norm_eps = norm_eps
        
        # Create a learnable weight parameter
        # This is initialized to ones, allowing the model to learn the scale
        # Shape: (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the RMS normalization.
        
        This normalizes the input by dividing by the root mean square, which is
        computed as sqrt(mean(x^2) + eps).
        
        Args:
            x: Input tensor of shape (..., dim)
        
        Returns:
            Normalized tensor of the same shape
        """
        # Compute mean of squared values along the last dimension
        # x.pow(2) squares each element
        # .mean(-1, keepdim=True) computes mean along last dimension, keeping dimensions
        # This gives us the mean squared value for each position
        # Add epsilon to prevent division by zero
        # torch.rsqrt computes 1/sqrt, which is more efficient than 1/sqrt
        # Multiply by x to normalize: x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RMSNorm layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        
        Returns:
            Normalized and scaled tensor of the same shape
        """
        # Normalize the input
        # Convert to float32 for numerical stability during normalization
        # Then convert back to original dtype (e.g., bfloat16)
        out = self._norm(x.float()).type_as(x)
        
        # Multiply by learnable weight parameter
        # This allows the model to learn the appropriate scale after normalization
        # Output shape: (batch, seq_len, dim)
        return out * self.weight


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation.
    
    The FFN applies two linear transformations with a SwiGLU activation function
    in between. SwiGLU (Swish-Gated Linear Unit) is a variant of GLU that uses
    the Swish activation function.
    
    Formula: FFN(x) = W2 * (SwiGLU(W1(x), W3(x)))
    where SwiGLU(a, b) = Swish(a) * b = (a * sigmoid(a)) * b
    """
    
    def __init__(self):
        """Initialize the feed-forward network layers."""
        # Call parent class constructor
        super().__init__()

        # First linear transformation: projects from DIM to FFN_DIM
        # bias=False: No bias term, as it's typically not used in transformer FFNs
        # This reduces parameters and can improve training stability
        self.w1 = nn.Linear(DIM, FFN_DIM, bias=False)
        
        # Third linear transformation: also projects from DIM to FFN_DIM
        # This is the "gate" projection in the SwiGLU activation
        # The gate controls which parts of the input are passed through
        self.w3 = nn.Linear(DIM, FFN_DIM, bias=False)
        
        # Second linear transformation: projects back from FFN_DIM to DIM
        # This is the output projection that brings the dimension back to DIM
        self.w2 = nn.Linear(FFN_DIM, DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, DIM)
        
        Returns:
            Output tensor of shape (batch, seq_len, DIM)
        """
        # Apply SwiGLU activation: Swish(W1(x)) * W3(x)
        # F.silu is the Swish activation: x * sigmoid(x)
        # This is element-wise multiplied with W3(x) to create the gating mechanism
        # Then W2 projects back to DIM
        # Output shape: (batch, seq_len, DIM)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """
    Grouped Query Attention (GQA) with KV cache for efficient inference.
    
    GQA uses fewer key-value heads than query heads, reducing memory and computation
    during inference. The KV cache stores previously computed keys and values to
    avoid recomputation during autoregressive generation.
    """
    
    def __init__(self):
        """Initialize the attention mechanism with query, key, value, and output projections."""
        # Call parent class constructor
        super().__init__()
        
        # Query projection: projects from DIM to (N_HEADS * HEAD_DIM)
        # This creates N_HEADS query vectors, each of dimension HEAD_DIM
        self.wq = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        
        # Key projection: projects from DIM to (N_KV_HEADS * HEAD_DIM)
        # Note: N_KV_HEADS < N_HEADS for GQA (e.g., 8 vs 32)
        self.wk = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        
        # Value projection: projects from DIM to (N_KV_HEADS * HEAD_DIM)
        # Same number of heads as keys
        self.wv = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        
        # Output projection: projects from (N_HEADS * HEAD_DIM) back to DIM
        # This combines all attention heads back into a single vector
        self.wo = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)

        # KV cache tensors will be lazily initialized on first forward pass
        # This avoids allocating large tensors during model initialization
        # Shape: (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM)
        self.cache_k = None
        self.cache_v = None

    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the attention mechanism.
        
        Args:
            x: Input tensor of shape (batch, seq_len, DIM)
            start_pos: Starting position in the sequence (for KV cache)
            freqs_cis: Precomputed RoPE frequencies
            mask: Attention mask to prevent attending to future tokens (causal mask)
        
        Returns:
            Output tensor of shape (batch, seq_len, DIM)
        """
        # Extract batch size and sequence length from input
        bsz, seqlen, _ = x.shape
        
        # Project input to queries, keys, and values
        # queries: (batch, seq_len, N_HEADS * HEAD_DIM)
        # keys, values: (batch, seq_len, N_KV_HEADS * HEAD_DIM)
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to separate heads
        # queries: (batch, seq_len, N_HEADS, HEAD_DIM)
        # keys, values: (batch, seq_len, N_KV_HEADS, HEAD_DIM)
        queries = queries.view(bsz, seqlen, N_HEADS, HEAD_DIM)
        keys = keys.view(bsz, seqlen, N_KV_HEADS, HEAD_DIM)
        values = values.view(bsz, seqlen, N_KV_HEADS, HEAD_DIM)

        # Apply Rotary Position Embeddings to queries and keys
        # This encodes positional information into the attention mechanism
        queries, keys = apply_rotary_emb(queries, keys, freqs_cis=freqs_cis)

        # Lazy initialization of KV cache tensors on first forward pass
        # This avoids allocating large tensors during model initialization
        if self.cache_k is None:
            self.cache_k = torch.zeros(
                (
                    MAX_BATCH_SIZE,
                    MAX_SEQ_LEN,
                    N_KV_HEADS,
                    HEAD_DIM,
                ),
                dtype=queries.dtype,
                device=queries.device,
            )
        if self.cache_v is None:
            self.cache_v = torch.zeros(
                (
                    MAX_BATCH_SIZE,
                    MAX_SEQ_LEN,
                    N_KV_HEADS,
                    HEAD_DIM,
                ),
                dtype=queries.dtype,
                device=queries.device,
            )
        
        # Ensure cache is on the same device as queries (for CPU->GPU migration)
        if self.cache_k.device != queries.device:
            self.cache_k = self.cache_k.to(queries.device)
        if self.cache_v.device != queries.device:
            self.cache_v = self.cache_v.to(queries.device)

        # Update KV cache with new keys and values
        # Store the computed keys/values at positions [start_pos : start_pos + seqlen]
        # This allows us to reuse them in subsequent forward passes
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = keys
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = values

        # Retrieve all keys and values up to current position
        # This includes both cached values and newly computed ones
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # Repeat KV heads to match the number of query heads (GQA -> MHA)
        # Each KV head is repeated N_KV_HEAD_REP times
        # This allows each query head to attend to the same KV heads
        # Shape: (batch, seq_len, N_KV_HEADS, HEAD_DIM) -> (batch, seq_len, N_HEADS, HEAD_DIM)
        keys = torch.repeat_interleave(
            keys, dim=2, repeats=N_KV_HEAD_REP
        )
        values = torch.repeat_interleave(
            values, dim=2, repeats=N_KV_HEAD_REP
        )

        # Reshape for scaled_dot_product_attention
        # PyTorch's attention expects (batch, num_heads, seq_len, head_dim)
        # Transpose to move head dimension before sequence dimension
        queries = queries.transpose(1, 2)  # (batch, N_HEADS, seq_len, HEAD_DIM)
        keys = keys.transpose(1, 2)         # (batch, N_HEADS, seq_len, HEAD_DIM)
        values = values.transpose(1, 2)     # (batch, N_HEADS, seq_len, HEAD_DIM)

        # Apply scaled dot-product attention
        # This computes: softmax(QK^T / sqrt(head_dim)) * V
        # The mask prevents attending to future tokens (causal attention)
        # Output shape: (batch, N_HEADS, seq_len, HEAD_DIM)
        out = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask,
        )
        
        # Reshape back to (batch, seq_len, N_HEADS, HEAD_DIM)
        # Transpose to move sequence dimension back before head dimension
        out = out.transpose(1, 2).contiguous()
        
        # Flatten heads: (batch, seq_len, N_HEADS, HEAD_DIM) -> (batch, seq_len, N_HEADS * HEAD_DIM)
        # The -1 in view() automatically computes N_HEADS * HEAD_DIM = DIM (it puts everything in a single dimension)
        out = out.view(bsz, seqlen, -1)
        
        # Apply output projection to combine all heads
        # Output shape: (batch, seq_len, DIM)
        return self.wo(out)


class TransformerBlock(nn.Module):
    """
    A single transformer block combining attention and feed-forward layers.
    
    Each block applies:
    1. Pre-norm attention with residual connection
    2. Pre-norm feed-forward with residual connection
    
    Pre-norm means normalization is applied before the sub-layer, not after.
    """
    
    def __init__(self):
        """Initialize the transformer block with attention, FFN, and normalization layers."""
        # Call parent class constructor
        super().__init__()
        
        # Multi-head attention mechanism
        self.attention = Attention()
        
        # Position-wise feed-forward network
        self.feed_forward = FeedForward()
        
        # Normalization layer before attention (pre-norm architecture)
        self.attention_norm = RMSNorm(DIM, NORM_EPS)
        
        # Normalization layer before feed-forward (pre-norm architecture)
        self.ffn_norm = RMSNorm(DIM, NORM_EPS)

    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, DIM)
            start_pos: Starting position for KV cache
            freqs_cis: Precomputed RoPE frequencies
            mask: Attention mask
        
        Returns:
            Output tensor of shape (batch, seq_len, DIM)
        """
        # Pre-norm attention with residual connection
        # Normalize input, apply attention, then add residual connection
        # This is the "pre-norm" architecture used in Llama
        # Output shape: (batch, seq_len, DIM)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        
        # Pre-norm feed-forward with residual connection
        # Normalize hidden state, apply FFN, then add residual connection
        # Output shape: (batch, seq_len, DIM)
        out = h + self.feed_forward(self.ffn_norm(h))
        
        # Return the output
        return out


class Transformer(nn.Module):
    """
    Complete Llama 3 transformer model.
    
    This is the main model class that combines:
    - Token embeddings
    - N_LAYERS transformer blocks
    - Final normalization and output projection
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize the complete transformer model.
        
        Args:
            device: Device to initialize the model on. Defaults to CPU to avoid OOM during initialization.
        """
        # Call parent class constructor
        super().__init__()
        
        # Default to CPU to avoid GPU memory allocation during initialization
        # This prevents OOM errors when initializing large models
        if device is None:
            device = torch.device("cpu")
        
        # Token embedding layer: maps token IDs to dense vectors
        # Input: token IDs (integers), Output: embedding vectors of size DIM
        # Vocabulary size: VOCAB_SIZE, embedding dimension: DIM
        # Initialize on CPU first, then move to device to avoid GPU allocation during init
        self.tok_embeddings = nn.Embedding(VOCAB_SIZE, DIM)
        self.tok_embeddings = self.tok_embeddings.to(device)
        
        # Create a list of transformer blocks
        # Each block is identical and processes the sequence independently
        self.layers = torch.nn.ModuleList()
        
        # Add N_LAYERS transformer blocks
        # Initialize on CPU first, then move to avoid GPU allocation during init
        for i in range(N_LAYERS):
            block = TransformerBlock()
            self.layers.append(block.to(device))
            if (i + 1) % 8 == 0:
                # Print progress every 8 layers to track initialization
                pass  # Can add print here if needed for debugging

        # Final normalization layer before output projection
        self.norm = RMSNorm(DIM, NORM_EPS)
        self.norm = self.norm.to(device)
        
        # Output projection: maps hidden states to vocabulary logits
        # This projects from DIM to VOCAB_SIZE to predict the next token
        # This is a large layer (4096 x 128256 ≈ 525M parameters, ~2GB in float32)
        # Initialize on CPU first, then move to device
        # bias=False: No bias term in the output layer
        self.output = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        # Move to device - this is a large transfer but necessary
        self.output = self.output.to(device)

        # Precompute RoPE frequencies for all possible positions
        # We compute for MAX_SEQ_LEN * 2 to have some buffer
        # These frequencies are used to encode positional information
        self.freqs_cis = precompute_freqs_cis(
            HEAD_DIM,           # Dimension per attention head
            MAX_SEQ_LEN * 2,    # Maximum sequence length (with buffer)
            ROPE_THETA,         # Base frequency parameter
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        Forward pass through the complete transformer model.
        
        Args:
            tokens: Token IDs of shape (batch, seq_len)
            start_pos: Starting position in the sequence (for KV cache)
        
        Returns:
            Logits over vocabulary of shape (batch, seq_len, VOCAB_SIZE)
        """
        # Extract batch size and sequence length (unpack tuple)
        _bsz, seqlen = tokens.shape
        
        # Convert token IDs to embedding vectors
        # Input: (batch, seq_len) of integers
        # Output: (batch, seq_len, DIM) of float vectors
        h = self.tok_embeddings(tokens)
        
        # Extract frequencies for the current sequence positions
        # We only need frequencies for positions [start_pos : start_pos + seqlen]
        # Use a local copy to avoid modifying the class attribute when moving between GPUs
        # Move to the device of tokens (will be moved to layer device later if needed)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(tokens.device)

        # Initialize attention mask to None
        # When seqlen=1 (autoregressive generation with KV cache), no mask is needed
        mask = None
        
        # Create causal mask for the first forward pass (when seqlen > 1)
        # During the first pass, we process multiple tokens at once, so we need a mask
        # to prevent the model from attending to future tokens
        if seqlen > 1:
            # Create a matrix filled with -inf
            # Shape: (seq_len, seq_len)
            # -inf ensures that masked positions contribute nothing to attention
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            # Create upper triangular mask (causal mask)
            # torch.triu with diagonal=1 sets all positions above the diagonal to -inf
            # This prevents attending to future tokens
            # Example for seq_len=4:
            # [[0, -inf, -inf, -inf],
            #  [0, 0, -inf, -inf],
            #  [0, 0, 0, -inf],
            #  [0, 0, 0, 0]]
            mask = torch.triu(mask, diagonal=1).to(tokens.device)

        # Pass through all transformer layers
        # Each layer processes the hidden states and passes them to the next
        # If using model parallelism, move tensors between GPUs as needed
        for i, layer in enumerate(self.layers):
            # Get the device of the layer (first parameter's device)
            try:
                layer_device = next(layer.parameters()).device
            except StopIteration:
                # Fallback if layer has no parameters (shouldn't happen)
                layer_device = h.device
            
            # Move input to the layer's device if using model parallelism
            if h.device != layer_device:
                h = h.to(layer_device)
            # Also move freqs_cis and mask if needed (always check, as device may have changed)
            if freqs_cis.device != layer_device:
                freqs_cis = freqs_cis.to(layer_device)
            if mask is not None and mask.device != layer_device:
                mask = mask.to(layer_device)
            
            # Apply transformer block: attention + FFN with residual connections
            # Output shape: (batch, seq_len, DIM)
            h = layer(h, start_pos, freqs_cis, mask)
        
        # Apply final normalization
        # Output shape: (batch, seq_len, DIM)
        h = self.norm(h)
        
        # Project to vocabulary logits
        # Convert to float32 for numerical stability in softmax
        # Output shape: (batch, seq_len, VOCAB_SIZE)
        out = self.output(h).float()
        
        # Return logits over vocabulary
        return out

