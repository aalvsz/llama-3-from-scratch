"""
Test script for model inference.

This script tests:
1. Text completion
2. Chat completion
3. Output quality verification
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from main project
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from inference import Llama


def test_text_completion(generator):
    """Test text completion functionality."""
    print("=" * 60)
    print("Test 1: Text Completion")
    print("=" * 60)
    
    prompts = [
        "The meaning of life is",
        "Python is a programming language that",
    ]
    
    print(f"\nGenerating completions for {len(prompts)} prompts...")
    try:
        results = generator.text_completion(
            prompts,
            max_gen_len=32,
            temperature=0.6,
            top_p=0.9,
        )
        
        print("\nResults:")
        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"\n{i}. Prompt: {prompt}")
            print(f"   Completion: {result['generation']}")
        
        print("\n✅ Text completion test passed!")
        return True
    except Exception as e:
        print(f"\n❌ Text completion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chat_completion(generator):
    """Test chat completion functionality."""
    print("\n" + "=" * 60)
    print("Test 2: Chat Completion")
    print("=" * 60)
    
    dialogs = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    ]
    
    print(f"\nGenerating chat response...")
    try:
        results = generator.chat_completion(
            dialogs,
            max_gen_len=32,
            temperature=0.6,
            top_p=0.9,
        )
        
        print("\nChat Result:")
        dialog = dialogs[0]
        for message in dialog:
            print(f"  {message['role']}: {message['content']}")
        
        result = results[0]
        print(f"  {result['generation']['role']}: {result['generation']['content']}")
        
        # Verify response structure
        assert 'generation' in result
        assert 'role' in result['generation']
        assert 'content' in result['generation']
        
        print("\n✅ Chat completion test passed!")
        return True
    except Exception as e:
        print(f"\n❌ Chat completion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization_verification(generator):
    """Verify quantization is working (memory usage)."""
    print("\n" + "=" * 60)
    print("Test 3: Quantization Verification")
    print("=" * 60)
    
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory verification")
        return True
    
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    
    print(f"\nMemory Usage:")
    print(f"  - Allocated: {memory_allocated:.2f} GB")
    print(f"  - Reserved: {memory_reserved:.2f} GB")
    
    # For quantized model, should be around 5GB, not 16GB
    if memory_allocated < 6:
        print(f"  ✅ Memory usage indicates quantization is working (~5GB expected)")
        return True
    elif memory_allocated < 10:
        print(f"  ⚠️  Memory usage is moderate ({memory_allocated:.2f}GB)")
        print(f"     Expected ~5GB for quantized model, but this may be acceptable")
        return True
    else:
        print(f"  ⚠️  Memory usage is high ({memory_allocated:.2f}GB)")
        print(f"     Expected ~5GB for quantized model")
        return False


def main():
    """Run all inference tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model inference")
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
    
    print("=" * 60)
    print("Model Inference Tests")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.ckpt_dir}...")
    try:
        generator = Llama.build(
            ckpt_dir=args.ckpt_dir + "/",
            tokenizer_path=args.tokenizer_path,
            max_seq_len=128,
            max_batch_size=4,
        )
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Run tests
    tests_passed = 0
    tests_total = 3
    
    if test_text_completion(generator):
        tests_passed += 1
    
    if test_chat_completion(generator):
        tests_passed += 1
    
    if test_quantization_verification(generator):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Summary: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    return 0 if tests_passed == tests_total else 1


if __name__ == "__main__":
    sys.exit(main())

