"""
Convert Hugging Face Llama 3 model to Meta checkpoint format.

This script converts a Hugging Face Llama 3 model to the format expected by
this implementation (consolidated.00.pth + params.json).

Usage:
    python convert_hf_to_meta.py --hf_model_dir /path/to/hf/model --output_dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def convert_hf_to_meta(hf_model_dir: str, output_dir: str):
    """
    Convert Hugging Face Llama 3 model to Meta checkpoint format.
    
    Args:
        hf_model_dir: Path to Hugging Face model directory
        output_dir: Path to output directory for converted files
    """
    hf_model_dir = Path(hf_model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Hugging Face model from {hf_model_dir}...")
    
    # Load Hugging Face model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = LlamaTokenizer.from_pretrained(hf_model_dir)
    
    # Get model config
    config = model.config
    
    # Create params.json in Meta format
    params = {
        "dim": config.hidden_size,
        "n_layers": config.num_hidden_layers,
        "n_heads": config.num_attention_heads,
        "n_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "vocab_size": config.vocab_size,
        "multiple_of": getattr(config, "multiple_of", 256),
        "ffn_dim_multiplier": getattr(config, "ffn_dim_multiplier", None),
        "norm_eps": config.rms_norm_eps,
        "rope_theta": getattr(config, "rope_theta", 500000.0),
    }
    
    # Save params.json
    params_path = output_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved params.json to {params_path}")
    
    # Convert state dict to Meta format
    # Meta format uses different key names
    state_dict = model.state_dict()
    meta_state_dict = {}
    
    # Mapping from HF keys to Meta keys
    for key, value in state_dict.items():
        # Remove 'model.' prefix if present
        new_key = key.replace("model.", "")
        
        # Convert layer names
        # HF: layers.0.self_attn.q_proj.weight
        # Meta: layers.0.attention.wq.weight
        new_key = new_key.replace("self_attn.q_proj", "attention.wq")
        new_key = new_key.replace("self_attn.k_proj", "attention.wk")
        new_key = new_key.replace("self_attn.v_proj", "attention.wv")
        new_key = new_key.replace("self_attn.o_proj", "attention.wo")
        
        # Convert feedforward names
        new_key = new_key.replace("mlp.gate_proj", "feed_forward.w1")
        new_key = new_key.replace("mlp.up_proj", "feed_forward.w3")
        new_key = new_key.replace("mlp.down_proj", "feed_forward.w2")
        
        # Convert normalization names
        new_key = new_key.replace("input_layernorm", "attention_norm")
        new_key = new_key.replace("post_attention_layernorm", "ffn_norm")
        
        # Embeddings
        new_key = new_key.replace("embed_tokens", "tok_embeddings")
        
        # Output layer
        new_key = new_key.replace("lm_head", "output")
        
        meta_state_dict[new_key] = value
    
    # Save consolidated checkpoint
    checkpoint_path = output_dir / "consolidated.00.pth"
    torch.save(meta_state_dict, checkpoint_path)
    print(f"Saved consolidated.00.pth to {checkpoint_path}")
    
    # Copy tokenizer.model if it exists
    tokenizer_model_path = hf_model_dir / "tokenizer.model"
    if tokenizer_model_path.exists():
        import shutil
        output_tokenizer_path = output_dir / "tokenizer.model"
        shutil.copy2(tokenizer_model_path, output_tokenizer_path)
        print(f"Copied tokenizer.model to {output_tokenizer_path}")
    else:
        print("Warning: tokenizer.model not found in Hugging Face model directory")
        print("You may need to download it separately from Meta's original release")
    
    print("\nConversion complete!")
    print(f"\nTo use the converted model, update your code:")
    print(f"  ckpt_dir = '{output_dir}/'")
    print(f"  tokenizer_path = '{output_dir / 'tokenizer.model'}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF Llama 3 to Meta format")
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Path to Hugging Face model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for converted files",
    )
    
    args = parser.parse_args()
    convert_hf_to_meta(args.hf_model_dir, args.output_dir)

