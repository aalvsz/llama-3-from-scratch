"""
Inference and text generation for Llama 3.

This module contains:
- Llama: Main class for loading models and generating text
- sample_top_p: Top-p (nucleus) sampling function

Copyright (c) Meta Platforms, Inc. and affiliates.
This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from config import MAX_BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE
from model import Transformer
from tokenizer import Tokenizer, ChatFormat, Dialog, Message


class CompletionPrediction(TypedDict, total=False):
    """
    Typed dictionary for text completion predictions.
    
    Attributes:
        generation: The generated text completion
        tokens: List of individual token strings (optional)
        logprobs: List of log probabilities for each token (optional)
    """
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    """
    Typed dictionary for chat completion predictions.
    
    Attributes:
        generation: The generated message with role and content
        tokens: List of individual token strings (optional)
        logprobs: List of log probabilities for each token (optional)
    """
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class Llama:
    """
    Main class for loading and using Llama 3 models.
    
    This class handles:
    - Loading model checkpoints and tokenizers
    - Text completion (autoregressive generation)
    - Chat completion (conversational generation)
    """
    
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by loading a model checkpoint.
        
        Args:
            ckpt_dir: Path to directory containing checkpoint files
            tokenizer_path: Path to the tokenizer model file
            max_seq_len: Maximum sequence length for inference
            max_batch_size: Maximum batch size for inference
            model_parallel_size: Number of model parallel processes (optional)
            seed: Random seed for reproducibility
        
        Returns:
            Llama instance with loaded model and tokenizer
        
        Raises:
            AssertionError: If checkpoint directory or tokenizer file doesn't exist
        """
        # Validate maximum sequence length (Llama 3 supports up to 8192)
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        
        # Verify checkpoint directory exists
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        
        # Verify tokenizer file exists
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        # Get local rank for distributed training (defaults to 0 for single GPU)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set the CUDA device to use
        torch.cuda.set_device(local_rank)

        # Set random seed for reproducibility
        # Seed must be the same in all processes for distributed training
        torch.manual_seed(seed)

        # Suppress stdout for non-main processes in distributed training
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        # Start timing the loading process
        start_time = time.time()
        
        # Find all checkpoint files in the directory
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        
        # Verify that at least one checkpoint file exists
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        
        # Load the main checkpoint file
        # consolidated.00.pth contains the model weights
        # map_location="cpu" loads to CPU first (will be moved to GPU later)
        checkpoint = torch.load(ckpt_dir + "consolidated.00.pth", map_location="cpu")
        
        # Load model parameters from params.json
        # This file contains hyperparameters and model configuration
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
            
        # Initialize the tokenizer
        tokenizer = Tokenizer(model_path=tokenizer_path)
        
        # Verify that the tokenizer vocabulary size matches our configuration
        assert VOCAB_SIZE == tokenizer.n_words
        
        # Set default tensor type based on GPU capabilities
        # BFloat16 is preferred for modern GPUs (better performance)
        # Half precision (FP16) is used as fallback for older GPUs
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
        # Initialize the transformer model
        model = Transformer()
        
        # Print the total number of parameters in the model
        print(f"PARAMETERS: {sum(p.numel() for p in model.parameters())}")
        
        # Load the checkpoint weights into the model
        # strict=False allows loading even if some keys don't match (for flexibility)
        model.load_state_dict(checkpoint, strict=False)
        
        # Print the time taken to load the model
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        # Return a Llama instance with the loaded model and tokenizer
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        """
        Initialize a Llama instance.
        
        Args:
            model: The loaded Transformer model
            tokenizer: The Tokenizer instance
        """
        # Store the model for inference
        self.model = model
        
        # Store the tokenizer for encoding/decoding
        self.tokenizer = tokenizer
        
        # Initialize the chat formatter for conversational interactions
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences from tokenized prompts.
        
        This is the core generation function that performs autoregressive generation
        using the transformer model with KV caching for efficiency.
        
        Args:
            prompt_tokens: List of tokenized prompts (each prompt is a list of token IDs)
            max_gen_len: Maximum length of generated text (in tokens)
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
            top_p: Top-p (nucleus) sampling parameter (0.9 = use top 90% probability mass)
            logprobs: Whether to compute and return token log probabilities
            echo: Whether to include prompt tokens in the output
        
        Returns:
            Tuple of (generated_token_sequences, optional_logprob_sequences)
        """
        # Get batch size (number of prompts)
        bsz = len(prompt_tokens)
        
        # Verify batch size doesn't exceed maximum
        assert bsz <= MAX_BATCH_SIZE, (bsz, MAX_BATCH_SIZE)

        # Find minimum and maximum prompt lengths
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        
        # Verify maximum prompt length doesn't exceed model's maximum
        assert max_prompt_len <= MAX_SEQ_LEN
        
        # Calculate total sequence length (prompt + generation)
        # Cap at MAX_SEQ_LEN to prevent exceeding model limits
        total_len = min(MAX_SEQ_LEN, max_gen_len + max_prompt_len)

        # Get padding token ID (not used, but needed for tensor initialization)
        pad_id = self.tokenizer.pad_id
        
        # Create tensor to store all tokens (prompt + generated)
        # Initialize with padding tokens
        # Shape: (batch_size, total_len)
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        
        # Fill in the prompt tokens
        for k, t in enumerate(prompt_tokens):
            # Copy prompt tokens into the tensor
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        
        # Initialize log probabilities tensor if requested
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        # Track previous position for KV cache
        prev_pos = 0
        
        # Track which sequences have reached end-of-sequence
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        
        # Create mask to identify which positions contain prompt tokens (not padding)
        input_text_mask = tokens != pad_id
        
        # If all prompts are already at maximum length, compute logits once
        if min_prompt_len == total_len:
            # Forward pass through the model
            logits = self.model.forward(tokens, prev_pos)
            
            # Compute log probabilities if requested
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),  # (batch, vocab, seq) for cross_entropy
                target=tokens,                  # Target tokens
                reduction="none",               # Don't reduce, keep per-token losses
                ignore_index=pad_id,            # Ignore padding tokens
            )

        # Convert stop tokens to tensor for efficient checking
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        # Autoregressive generation loop
        # Generate one token at a time until max length or all sequences stop
        for cur_pos in range(min_prompt_len, total_len):
            # Forward pass: process tokens from prev_pos to cur_pos
            # This uses KV cache to avoid recomputing previous tokens
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            
            # Sample next token from the logits
            if temperature > 0:
                # Apply temperature scaling to logits
                # Higher temperature = flatter distribution = more randomness
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                
                # Use top-p (nucleus) sampling
                # This samples from the smallest set of tokens whose cumulative probability >= top_p
                next_token = sample_top_p(probs, top_p)
            else:
                # Greedy decoding: take the token with highest probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # Reshape to 1D tensor
            next_token = next_token.reshape(-1)
            
            # Only replace token if we're past the prompt (not in prompt region)
            # This ensures we don't overwrite the original prompt
            next_token = torch.where(
                input_text_mask[:, cur_pos],  # If this position is in the prompt
                tokens[:, cur_pos],           # Keep the original prompt token
                next_token                    # Otherwise use the generated token
            )
            
            # Store the next token
            tokens[:, cur_pos] = next_token
            
            # Compute log probabilities if requested
            if logprobs:
                # Compute cross-entropy loss for the newly generated tokens
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            
            # Check if any sequences have reached a stop token
            # Only check for sequences that are past the prompt (not in input_text_mask)
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            
            # Update previous position for next iteration
            prev_pos = cur_pos
            
            # If all sequences have stopped, break early
            if all(eos_reached):
                break

        # Convert log probabilities to Python list if requested
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        
        # Process output tokens
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # Determine start position based on echo flag
            # If echo=True, include prompt; if echo=False, only include generated text
            start = 0 if echo else len(prompt_tokens[i])
            
            # Extract tokens up to max generation length
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            
            # Extract corresponding log probabilities if requested
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            
            # Truncate at first stop token if found
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    # Find index of stop token
                    eos_idx = toks.index(stop_token)
                    # Truncate tokens and logprobs at stop token
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    # Stop token not found, continue
                    pass
            
            # Add to output lists
            out_tokens.append(toks)
            out_logprobs.append(probs)
        
        # Return generated tokens and optional log probabilities
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of text prompts.
        
        This is a high-level interface that handles tokenization and decoding.
        
        Args:
            prompts: List of text prompts to complete
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_gen_len: Maximum generation length (defaults to MAX_SEQ_LEN - 1)
            logprobs: Whether to return token log probabilities
            echo: Whether to include prompt in output
        
        Returns:
            List of completion predictions with generated text
        """
        # Set default max generation length if not provided
        if max_gen_len is None:
            max_gen_len = MAX_SEQ_LEN - 1
        
        # Tokenize all prompts
        # bos=True: Add beginning-of-sequence token
        # eos=False: Don't add end-of-sequence token (model will generate it)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        
        # Generate tokens using the core generation function
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        
        # Format results with optional log probabilities
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),  # Decode tokens to text
                    "tokens": [self.tokenizer.decode([x]) for x in t],  # Individual token strings
                    "logprobs": logprobs_i,  # Log probabilities for each token
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        
        # Return simple completions without log probabilities
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for conversational dialogs.
        
        This handles the special chat formatting required for multi-turn conversations.
        
        Args:
            dialogs: List of conversational dialogs (each dialog is a list of messages)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_gen_len: Maximum generation length (defaults to MAX_SEQ_LEN - 1)
            logprobs: Whether to return token log probabilities
        
        Returns:
            List of chat predictions with assistant responses
        """
        # Set default max generation length if not provided
        if max_gen_len is None:
            max_gen_len = MAX_SEQ_LEN - 1

        # Encode dialogs using the chat formatter
        # This adds special tokens for roles and message boundaries
        prompt_tokens = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ]
        
        # Generate tokens using the core generation function
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        
        # Format results with optional log probabilities
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",  # All generated responses are from assistant
                        "content": self.tokenizer.decode(t),  # Decoded text content
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],  # Individual tokens
                    "logprobs": logprobs_i,  # Log probabilities
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        
        # Return simple chat completions without log probabilities
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                },
            }
            for t in generation_tokens
        ]


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.
    
    Top-p sampling selects the smallest set of tokens whose cumulative probability
    mass exceeds the threshold p. The distribution is then renormalized and sampled from.
    
    This provides a balance between diversity and quality:
    - High p (e.g., 0.9): More diverse outputs, includes lower-probability tokens
    - Low p (e.g., 0.5): More focused outputs, only high-probability tokens
    
    Args:
        probs: Probability distribution tensor of shape (batch, vocab_size)
        p: Probability threshold (typically 0.9)
    
    Returns:
        Sampled token indices of shape (batch, 1)
    """
    # Sort probabilities in descending order
    # probs_sort: sorted probabilities
    # probs_idx: original indices of sorted probabilities
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # Compute cumulative sum of sorted probabilities
    # This gives us the cumulative probability mass
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # Create mask for tokens to keep
    # We keep tokens where cumulative probability - current probability <= p
    # This selects the smallest set whose cumulative probability > p
    mask = probs_sum - probs_sort > p
    
    # Set probabilities of masked tokens to 0
    # These tokens will not be sampled
    probs_sort[mask] = 0.0
    
    # Renormalize the distribution
    # Divide by sum to ensure probabilities sum to 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # Sample one token from the renormalized distribution
    # torch.multinomial samples according to the probability distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    
    # Map sampled indices back to original vocabulary indices
    # torch.gather retrieves the original token IDs
    next_token = torch.gather(probs_idx, -1, next_token)
    
    # Return the sampled token indices
    return next_token

