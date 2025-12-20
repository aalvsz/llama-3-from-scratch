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

from .config import MAX_BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE, N_LAYERS
from .model import Transformer
from .tokenizer import Tokenizer, ChatFormat, Dialog, Message


def _hf_to_meta_key(key: str) -> str:
    if key.startswith("model."):
        key = key[len("model.") :]
    key = key.replace("self_attn.q_proj", "attention.wq")
    key = key.replace("self_attn.k_proj", "attention.wk")
    key = key.replace("self_attn.v_proj", "attention.wv")
    key = key.replace("self_attn.o_proj", "attention.wo")
    key = key.replace("mlp.gate_proj", "feed_forward.w1")
    key = key.replace("mlp.up_proj", "feed_forward.w3")
    key = key.replace("mlp.down_proj", "feed_forward.w2")
    key = key.replace("input_layernorm", "attention_norm")
    key = key.replace("post_attention_layernorm", "ffn_norm")
    key = key.replace("embed_tokens", "tok_embeddings")
    key = key.replace("lm_head", "output")
    return key


def _load_safetensors_shards(model: Transformer, shards: List[Path]) -> None:
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(
            "safetensors is required to load model-*.safetensors files. "
            "Install it with `pip install safetensors`."
        ) from e

    state_dict = model.state_dict()
    loaded = 0
    skipped = 0

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                new_key = _hf_to_meta_key(key)
                if new_key not in state_dict:
                    skipped += 1
                    continue
                tensor = f.get_tensor(key)
                if tensor.shape != state_dict[new_key].shape:
                    raise ValueError(
                        f"Shape mismatch for {new_key}: "
                        f"checkpoint {tuple(tensor.shape)} vs model {tuple(state_dict[new_key].shape)}"
                    )
                state_dict[new_key].copy_(tensor)
                loaded += 1

    if loaded == 0:
        raise ValueError("No matching tensors were loaded from safetensors shards.")

    if skipped > 0:
        print(f"Note: skipped {skipped} unmatched tensors while loading safetensors.")

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
        
        # Set the CUDA device to use (only if CUDA is available)
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
            except Exception as e:
                print(f"Warning: Could not set CUDA device {local_rank}: {e}")
                local_rank = 0
        # Note: MPS (Metal Performance Shaders) for Apple Silicon doesn't need explicit device setting

        # Set random seed for reproducibility
        # Seed must be the same in all processes for distributed training
        torch.manual_seed(seed)

        # Suppress stdout for non-main processes in distributed training
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        # Start timing the loading process
        start_time = time.time()
        
        # Identify available checkpoint files
        consolidated_path = Path(ckpt_dir) / "consolidated.00.pth"
        safetensors_files = sorted(Path(ckpt_dir).glob("model-*.safetensors"))
        
        # Verify that at least one checkpoint format exists
        assert consolidated_path.exists() or safetensors_files, (
            f"no checkpoint files found in {ckpt_dir} "
            f"(expected consolidated.00.pth or model-*.safetensors)"
        )
        
        # Load model parameters from params.json (optional for inference)
        params_path = Path(ckpt_dir) / "params.json"
        if params_path.exists():
            with open(params_path, "r") as f:
                _ = json.loads(f.read())
        else:
            print(f"Warning: params.json not found in {ckpt_dir}. Continuing without it.")
            
        # Initialize the tokenizer
        tokenizer = Tokenizer(model_path=tokenizer_path)
        
        # Verify that the tokenizer vocabulary size matches our configuration
        assert VOCAB_SIZE == tokenizer.n_words
        
        # Clear GPU memory before model initialization to prevent OOM errors
        # This is especially important when running in notebooks where cells may be re-executed
        if torch.cuda.is_available():
            # Report current memory usage for all GPUs
            num_gpus = torch.cuda.device_count()
            for gpu_id in range(num_gpus):
                allocated = torch.cuda.memory_allocated(gpu_id)
                reserved = torch.cuda.memory_reserved(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                free = total - allocated
                
                if allocated > 0:
                    print(f"GPU {gpu_id} memory before cleanup: {allocated / 1024**3:.2f} GB allocated, "
                          f"{reserved / 1024**3:.2f} GB reserved, {free / 1024**3:.2f} GB free")
            
            # Clear cache on all GPUs and run garbage collection
            for gpu_id in range(num_gpus):
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # Report memory after cleanup
            for gpu_id in range(num_gpus):
                allocated_after = torch.cuda.memory_allocated(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                free_after = total - allocated_after
                if allocated_after > 0:
                    print(f"GPU {gpu_id} memory after cleanup: {allocated_after / 1024**3:.2f} GB allocated, "
                          f"{free_after / 1024**3:.2f} GB free")
            print()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS (Metal Performance Shaders) for Apple Silicon
            import gc
            gc.collect()
            print("MPS (Metal) available - memory will be managed automatically")
            print()
        
        # Initialize the transformer model on CPU to avoid OOM during initialization
        # Use float16 on CPU when CUDA is available to reduce host RAM footprint
        print("Initializing model on CPU...")
        cpu_device = torch.device("cpu")
        init_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(init_dtype)
            model = Transformer(device=cpu_device)
            torch.set_default_dtype(prev_dtype)
            print(f"✓ Model initialized on CPU ({init_dtype})")
        except Exception as e:
            torch.set_default_dtype(prev_dtype)
            print(f"❌ Error initializing model: {e}")
            import traceback
            traceback.print_exc()
            raise

        # For multi-GPU setups, move the model to GPUs before loading weights
        # This mirrors the behavior of initializing directly on GPU to reduce CPU RAM usage
        model_on_gpu = False
        if torch.cuda.is_available():
            if model_parallel_size is None and torch.cuda.device_count() >= 2:
                model_parallel_size = 2
                print("Auto-selected model_parallel_size=2 for multi-GPU inference")

            if model_parallel_size is not None and model_parallel_size > 1:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                num_gpus = min(model_parallel_size, torch.cuda.device_count())
                device0 = torch.device("cuda:0")
                print(f"Placing model on {num_gpus} GPUs before loading weights...")

                model.tok_embeddings = model.tok_embeddings.to(device=device0, dtype=dtype)
                model.norm = model.norm.to(device=device0, dtype=dtype)
                model.output = model.output.to(device=device0, dtype=dtype)

                layers_per_gpu = (N_LAYERS + num_gpus - 1) // num_gpus
                for i, layer in enumerate(model.layers):
                    gpu_id = min(i // layers_per_gpu, num_gpus - 1)
                    layer_device = torch.device(f"cuda:{gpu_id}")
                    model.layers[i] = layer.to(device=layer_device, dtype=dtype)
                model_on_gpu = True
        
        # Print the total number of parameters in the model
        try:
            param_count = sum(p.numel() for p in model.parameters())
            print(f"PARAMETERS: {param_count:,}")
        except Exception as e:
            print(f"Warning: Could not count parameters: {e}")
        
        # Load the checkpoint weights into the model (on CPU)
        # strict=False allows loading even if some keys don't match (for flexibility)
        print("Loading checkpoint weights...")
        try:
            if consolidated_path.exists():
                # consolidated.00.pth contains the model weights
                # map_location="cpu" loads to CPU first (will be moved to GPU later)
                checkpoint = torch.load(str(consolidated_path), map_location="cpu")
                model.load_state_dict(checkpoint, strict=False)
                print("✓ Checkpoint loaded from consolidated.00.pth")
            else:
                _load_safetensors_shards(model, safetensors_files)
                print("✓ Checkpoint loaded from safetensors shards")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Move model to GPU/MPS and convert to appropriate dtype
        # This is more memory-efficient than initializing directly on GPU
        if torch.cuda.is_available():
            if model_on_gpu:
                print("Model already placed on GPU(s) before loading weights")
                device = next(model.tok_embeddings.parameters()).device
                dtype = next(model.tok_embeddings.parameters()).dtype
            else:
                # Determine device and dtype
                device = torch.device(f"cuda:{local_rank}")
                # Use bfloat16 for modern GPUs, float16 as fallback
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

                # If model parallelism is requested, split layers across GPUs
                if model_parallel_size is not None and model_parallel_size > 1:
                    num_gpus = min(model_parallel_size, torch.cuda.device_count())
                    if num_gpus < 2:
                        print(f"Warning: model_parallel_size={model_parallel_size} requested but only {torch.cuda.device_count()} GPU(s) available. Using single GPU.")
                        model = model.to(device=device, dtype=dtype)
                        print(f"Model moved to GPU {local_rank} with dtype {dtype}")
                    else:
                        print(f"Using model parallelism across {num_gpus} GPUs")
                        
                        # Move embedding and output to first GPU
                        try:
                            model.tok_embeddings = model.tok_embeddings.to(device=device, dtype=dtype)
                            model.norm = model.norm.to(device=device, dtype=dtype)
                            model.output = model.output.to(device=device, dtype=dtype)
                            
                            # Distribute layers across GPUs
                            layers_per_gpu = (N_LAYERS + num_gpus - 1) // num_gpus
                            for i, layer in enumerate(model.layers):
                                gpu_id = min(i // layers_per_gpu, num_gpus - 1)
                                layer_device = torch.device(f"cuda:{gpu_id}")
                                model.layers[i] = layer.to(device=layer_device, dtype=dtype)
                                if i == 0 or (i + 1) % layers_per_gpu == 0 or i == N_LAYERS - 1:
                                    end_layer = min(i + layers_per_gpu - 1, N_LAYERS - 1)
                                    print(f"  Layers {i}-{end_layer} on GPU {gpu_id}")
                        except RuntimeError as e:
                            print(f"Error during model parallelism setup: {e}")
                            print("Falling back to single GPU mode...")
                            # Fallback to single GPU
                            model = model.to(device=device, dtype=dtype)
                            print(f"Model moved to GPU {local_rank} with dtype {dtype}")
                else:
                    # Single GPU: move entire model
                    try:
                        model = model.to(device=device, dtype=dtype)
                        print(f"Model moved to GPU {local_rank} with dtype {dtype}")
                    except RuntimeError as e:
                        print(f"Error moving model to GPU: {e}")
                        print("This might be an out-of-memory error. Try:")
                        print("  1. Reducing max_seq_len or max_batch_size")
                        print("  2. Using model_parallel_size=2 to split across GPUs")
                        raise
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS (Metal Performance Shaders) for Apple Silicon (M1/M2/M3)
            device = torch.device("mps")
            # MPS supports float32 and float16, but bfloat16 support varies by macOS version
            # Use float16 for better performance on Apple Silicon
            dtype = torch.float16
            print(f"Using MPS (Metal) device on Apple Silicon")
            
            # MPS doesn't support model parallelism across multiple GPUs
            if model_parallel_size is not None and model_parallel_size > 1:
                print(f"Warning: model_parallel_size={model_parallel_size} requested but MPS doesn't support multi-GPU parallelism. Using single device.")
            
            try:
                model = model.to(device=device, dtype=dtype)
                print(f"Model moved to MPS device with dtype {dtype}")
            except RuntimeError as e:
                print(f"Error moving model to MPS: {e}")
                print("This might be an out-of-memory error. Try:")
                print("  1. Reducing max_seq_len or max_batch_size")
                print("  2. Using CPU instead (slower but more memory available)")
                raise
        else:
            # Fallback to CPU
            device = torch.device("cpu")
            dtype = torch.float32  # CPU typically uses float32
            print("No GPU/MPS available - using CPU (this will be slow)")
            model = model.to(device=device, dtype=dtype)
            print(f"Model moved to CPU with dtype {dtype}")
        
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
        
        # Get the device of the model's embedding layer (first GPU in model parallelism)
        model_device = next(self.model.tok_embeddings.parameters()).device
        
        # Create tensor to store all tokens (prompt + generated)
        # Initialize with padding tokens
        # Shape: (batch_size, total_len)
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=model_device)
        
        # Fill in the prompt tokens
        for k, t in enumerate(prompt_tokens):
            # Copy prompt tokens into the tensor
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=model_device)
        
        # Initialize log probabilities tensor if requested
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        # Track previous position for KV cache
        prev_pos = 0
        
        # Track which sequences have reached end-of-sequence
        eos_reached = torch.tensor([False] * bsz, device=model_device)
        
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
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device=model_device)

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
