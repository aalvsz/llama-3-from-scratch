"""
Convert downloaded Hugging Face model to Meta format.

This script converts the already-downloaded model files to the format
expected by the inference code.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Please install: pip install safetensors")
    sys.exit(1)


def convert_model(hf_model_dir, output_dir):
    """
    Convert Hugging Face model to Meta format.
    
    Args:
        hf_model_dir: Path to downloaded HF model directory
        output_dir: Output directory for converted files
    """
    hf_model_dir = Path(hf_model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Converting Hugging Face Model to Meta Format")
    print("=" * 60)
    
    # Load config
    config_path = hf_model_dir / "config.json"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\n1. Model Configuration:")
    print(f"   - Hidden size: {config['hidden_size']}")
    print(f"   - Layers: {config['num_hidden_layers']}")
    print(f"   - Attention heads: {config['num_attention_heads']}")
    print(f"   - KV heads: {config.get('num_key_value_heads', config['num_attention_heads'])}")
    print(f"   - Vocab size: {config['vocab_size']}")
    
    # Create params.json
    params = {
        "dim": config["hidden_size"],
        "n_layers": config["num_hidden_layers"],
        "n_heads": config["num_attention_heads"],
        "n_kv_heads": config.get("num_key_value_heads", config["num_attention_heads"]),
        "vocab_size": config["vocab_size"],
        "multiple_of": config.get("multiple_of", 256),
        "ffn_dim_multiplier": config.get("ffn_dim_multiplier", None),
        "norm_eps": config["rms_norm_eps"],
        "rope_theta": config.get("rope_theta", 500000.0),
    }
    
    params_path = output_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n2. Saved params.json to {params_path}")
    
    # Load model weights from safetensors
    print("\n3. Loading model weights from safetensors...")
    
    # Find all safetensors files
    safetensors_files = sorted(hf_model_dir.glob("model-*.safetensors"))
    if not safetensors_files:
        print(f"❌ No safetensors files found in {hf_model_dir}")
        return False
    
    print(f"   Found {len(safetensors_files)} safetensors file(s)")
    
    # Load all weights
    state_dict = {}
    for st_file in safetensors_files:
        print(f"   Loading {st_file.name}...")
        tensors = load_file(str(st_file))
        state_dict.update(tensors)
    
    print(f"   ✓ Loaded {len(state_dict)} weight tensors")
    
    # Convert to Meta format
    print("\n4. Converting to Meta format...")
    meta_state_dict = {}
    
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
        
        meta_state_dict[new_key] = value
    
    print(f"   ✓ Converted {len(meta_state_dict)} weight tensors")
    
    # Save consolidated checkpoint
    checkpoint_path = output_dir / "consolidated.00.pth"
    print(f"\n5. Saving consolidated checkpoint...")
    torch.save(meta_state_dict, checkpoint_path)
    size_gb = checkpoint_path.stat().st_size / 1024**3
    print(f"   ✓ Saved consolidated.00.pth ({size_gb:.2f} GB)")
    
    # Copy tokenizer.model
    print(f"\n6. Copying tokenizer.model...")
    tokenizer_sources = [
        hf_model_dir / "original" / "tokenizer.model",
        hf_model_dir / "tokenizer.model",
    ]
    
    tokenizer_path = None
    for src in tokenizer_sources:
        if src.exists():
            import shutil
            dst = output_dir / "tokenizer.model"
            shutil.copy2(src, dst)
            tokenizer_path = dst
            print(f"   ✓ Copied tokenizer.model to {dst}")
            break
    
    if not tokenizer_path:
        print(f"   ⚠️  tokenizer.model not found in expected locations")
        print(f"   Please copy it manually to {output_dir / 'tokenizer.model'}")
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {params_path}")
    print(f"  - {checkpoint_path}")
    if tokenizer_path:
        print(f"  - {tokenizer_path}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert downloaded HF model to Meta format")
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        default="./checkpoints/hf_model",
        help="Path to downloaded HF model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for converted files",
    )
    
    args = parser.parse_args()
    
    success = convert_model(args.hf_model_dir, args.output_dir)
    sys.exit(0 if success else 1)

