"""
Example usage of the Llama 3 model for text completion.

This script demonstrates how to:
1. Load a pre-trained Llama 3 model
2. Generate text completions from prompts
3. Use chat completion for conversational interactions
"""

# For use as a module
try:
    from .inference import Llama
except ImportError:
    # For use as a standalone script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.inference import Llama

# Configuration for model loading
# Update these paths to point to your model checkpoint and tokenizer
ckpt_dir = "/kaggle/input/llama3/"  # Directory containing consolidated.00.pth and params.json
tokenizer_path = "/kaggle/input/llama3/tokenizer.model"  # Path to tokenizer.model file

# Generation parameters
temperature = 0.6  # Sampling temperature (0.0 = deterministic, higher = more random)
top_p = 0.9       # Top-p (nucleus) sampling parameter
max_seq_len = 128 # Maximum sequence length for the model
max_gen_len = 64  # Maximum number of tokens to generate
max_batch_size = 4  # Maximum batch size for processing multiple prompts

# Build the Llama model instance
# This loads the model weights and tokenizer
print("Loading model...")
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    model_parallel_size=1  # Set to 1 for single GPU, increase for multi-GPU
)
print("Model loaded successfully!\n")

# Example 1: Text completion
# Generate continuations for text prompts
print("=" * 50)
print("Example 1: Text Completion")
print("=" * 50)

prompts = [
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
]

# Generate completions for all prompts
results = generator.text_completion(
    prompts,
    max_gen_len=max_gen_len,
    temperature=temperature,
    top_p=top_p,
)

# Print results
for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Completion: {result['generation']}")
    print("\n" + "=" * 50 + "\n")

# Example 2: Chat completion
# Generate responses in a conversational format
print("=" * 50)
print("Example 2: Chat Completion")
print("=" * 50)

# Define a conversation dialog
# Each message has a role (system, user, or assistant) and content
dialogs = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
]

# Generate assistant response
chat_results = generator.chat_completion(
    dialogs,
    max_gen_len=max_gen_len,
    temperature=temperature,
    top_p=top_p,
)

# Print chat results
for dialog, result in zip(dialogs, chat_results):
    print("Dialog:")
    for message in dialog:
        print(f"  {message['role']}: {message['content']}")
    print(f"\nAssistant: {result['generation']['content']}")
    print("\n" + "=" * 50 + "\n")

# Example 3: Multi-turn conversation
print("=" * 50)
print("Example 3: Multi-turn Conversation")
print("=" * 50)

# Start a conversation
conversation = [
    {"role": "user", "content": "Explain quantum computing in simple terms."},
]

# Generate first response
response1 = generator.chat_completion(
    [conversation],
    max_gen_len=max_gen_len,
    temperature=temperature,
    top_p=top_p,
)[0]

# Add assistant response to conversation
conversation.append(response1["generation"])

# Add follow-up question
conversation.append({"role": "user", "content": "How does it differ from classical computing?"})

# Generate second response
response2 = generator.chat_completion(
    [conversation],
    max_gen_len=max_gen_len,
    temperature=temperature,
    top_p=top_p,
)[0]

# Print the full conversation
print("Full conversation:")
for message in conversation:
    print(f"  {message['role']}: {message['content']}")
print(f"  {response2['generation']['role']}: {response2['generation']['content']}")

