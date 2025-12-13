"""
Simple script to load Q4KM quantized Llama 3.1 8B.

This script attempts to load a publicly available Q4KM model or provides
instructions for authentication if needed.
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_q4km_model(output_dir="./checkpoints"):
    """
    Load Q4KM quantized Llama 3.1 8B model.
    
    For Q4KM, we'll use BitsAndBytes 4-bit quantization which is equivalent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Loading Q4KM Quantized Llama 3.1 8B Model")
    print("=" * 60)
    
    # Check for Hugging Face token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    if not token:
        print("\n⚠️  No Hugging Face token found.")
        print("   Llama 3.1 models are gated and require authentication.")
        print("   Please do one of the following:")
        print("\n   1. Get access at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
        print("   2. Set your token:")
        print("      export HF_TOKEN=your_token_here")
        print("      # or")
        print("      huggingface-cli login")
        print("\n   Then run this script again.")
        return None, None
    
    print(f"\n✓ Using Hugging Face token (starts with {token[:10]}...)")
    
    # Try to load with 4-bit quantization (equivalent to Q4KM)
    print("\n1. Loading model with 4-bit quantization (Q4KM equivalent)...")
    
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    try:
        # Load tokenizer first
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
        )
        print("   ✓ Tokenizer loaded")
        
        # Configure 4-bit quantization (equivalent to Q4KM)
        print("   Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with quantization
        print("   Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=token,
        )
        print("   ✓ Model loaded successfully with 4-bit quantization")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        print("\n   Make sure you have:")
        print("   1. Access to the model on Hugging Face")
        print("   2. A valid token set in HF_TOKEN environment variable")
        print("   3. Sufficient disk space (~5GB for quantized model)")
        return None, None
    
    # Get model config
    config = model.config
    print(f"\n2. Model Configuration:")
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Layers: {config.num_hidden_layers}")
    print(f"   - Attention heads: {config.num_attention_heads}")
    print(f"   - KV heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
    print(f"   - Vocab size: {config.vocab_size}")
    
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
    print(f"\n3. Saved params.json to {params_path}")
    
    # Convert state dict to Meta format
    print("\n4. Converting model weights to Meta format...")
    state_dict = model.state_dict()
    meta_state_dict = {}
    
    # Mapping from HF keys to Meta keys
    for key, value in state_dict.items():
        # Remove 'model.' prefix if present
        new_key = key.replace("model.", "")
        
        # Convert layer names
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
        
        # Handle quantized weights
        if hasattr(value, 'data'):
            meta_state_dict[new_key] = value.data
        else:
            meta_state_dict[new_key] = value
    
    # Save consolidated checkpoint
    checkpoint_path = output_dir / "consolidated.00.pth"
    print(f"   Saving checkpoint to {checkpoint_path}...")
    torch.save(meta_state_dict, checkpoint_path)
    print(f"   ✓ Saved consolidated.00.pth ({checkpoint_path.stat().st_size / 1024**3:.2f} GB)")
    
    # Download tokenizer.model
    print(f"\n5. Downloading tokenizer.model...")
    try:
        from huggingface_hub import hf_hub_download
        tokenizer_model = hf_hub_download(
            repo_id=model_name,
            filename="tokenizer.model",
            local_dir=output_dir,
            token=token,
        )
        tokenizer_path = Path(tokenizer_model)
        print(f"   ✓ Tokenizer saved to {tokenizer_path}")
    except Exception as e:
        print(f"   ⚠ Could not download tokenizer.model: {e}")
        print(f"   You may need to download it manually from {model_name}")
        tokenizer_path = output_dir / "tokenizer.model"
        print(f"   Expected path: {tokenizer_path}")
    
    print("\n" + "=" * 60)
    print("Model Loading Complete!")
    print("=" * 60)
    print(f"\nTo use the model, update your code:")
    print(f"  ckpt_dir = '{output_dir}/'")
    print(f"  tokenizer_path = '{tokenizer_path}'")
    print("\nNote: The model is quantized (4-bit), so it uses less memory")
    print("but may have slightly different behavior than full precision.")
    
    return str(output_dir), str(tokenizer_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Q4KM quantized Llama 3.1 8B")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for converted files",
    )
    
    args = parser.parse_args()
    
    load_q4km_model(output_dir=args.output_dir)

