"""
Test script to verify model loading works correctly.

This script tests:
1. Model loading with Llama.build()
2. Basic model initialization
3. Parameter count verification
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from main project
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from inference import Llama
from config import DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE


def test_model_loading(ckpt_dir="./checkpoints", tokenizer_path="./checkpoints/tokenizer.model"):
    """
    Test that the model loads correctly.
    
    Args:
        ckpt_dir: Directory containing consolidated.00.pth and params.json
        tokenizer_path: Path to tokenizer.model file
    """
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    # Check if checkpoint directory exists
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        print(f"❌ Checkpoint directory not found: {ckpt_dir}")
        return False
    
    # Check for required files
    consolidated_path = ckpt_dir / "consolidated.00.pth"
    params_path = ckpt_dir / "params.json"
    tokenizer_file = Path(tokenizer_path)
    
    missing_files = []
    if not consolidated_path.exists():
        missing_files.append(f"consolidated.00.pth")
    if not params_path.exists():
        missing_files.append(f"params.json")
    if not tokenizer_file.exists():
        missing_files.append(f"tokenizer.model")
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print(f"✓ Found checkpoint directory: {ckpt_dir}")
    print(f"✓ Found consolidated.00.pth: {consolidated_path.stat().st_size / 1024**3:.2f} GB")
    print(f"✓ Found params.json")
    print(f"✓ Found tokenizer.model")
    
    # Try to load the model
    print("\nLoading model with Llama.build()...")
    try:
        generator = Llama.build(
            ckpt_dir=str(ckpt_dir) + "/",
            tokenizer_path=tokenizer_path,
            max_seq_len=128,
            max_batch_size=4,
        )
        print("✅ Model loaded successfully!")
        
        # Verify model parameters
        total_params = sum(p.numel() for p in generator.model.parameters())
        print(f"\nModel Statistics:")
        print(f"  - Total parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
        print(f"  - Expected: ~8B parameters")
        
        # Check memory usage (approximate)
        import torch
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  - GPU memory allocated: {memory_allocated:.2f} GB")
            if memory_allocated < 6:
                print(f"  ✅ Memory usage looks correct for quantized model (~5GB)")
            else:
                print(f"  ⚠️  Memory usage higher than expected for quantized model")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./checkpoints/tokenizer.model",
        help="Path to tokenizer.model",
    )
    
    args = parser.parse_args()
    
    success = test_model_loading(args.ckpt_dir, args.tokenizer_path)
    sys.exit(0 if success else 1)

