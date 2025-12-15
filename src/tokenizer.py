"""
Tokenizer and chat formatting for Llama 3.

This module contains:
- Tokenizer: Converts text to token IDs and vice versa using Tiktoken
- ChatFormat: Formats conversational dialogs into token sequences

Copyright (c) Meta Platforms, Inc. and affiliates.
This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
"""

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe

# Set up logger for debugging and information messages
logger = getLogger(__name__)

# Define the role types that can appear in a chat message
# "system": System-level instructions or context
# "user": User messages
# "assistant": Assistant responses
Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """
    Typed dictionary representing a single chat message.
    
    Attributes:
        role: The role of the message sender (system, user, or assistant)
        content: The text content of the message
    """
    role: Role
    content: str


# A dialog is a sequence of messages forming a conversation
Dialog = Sequence[Message]


class Tokenizer:
    """
    Tokenizer for encoding and decoding text using the Tiktoken tokenizer.
    
    This tokenizer uses Byte Pair Encoding (BPE) to convert text into token IDs
    and vice versa. It handles special tokens used for chat formatting.
    """

    # Dictionary mapping special token strings to their token IDs
    special_tokens: Dict[str, int]

    # Number of reserved special tokens in the vocabulary
    # These are tokens reserved for special purposes (e.g., chat formatting)
    num_reserved_special_tokens = 256

    # Regular expression pattern for tokenization
    # This pattern defines how text is split into tokens
    # It handles contractions, words, numbers, punctuation, and whitespace
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initialize the Tokenizer with a Tiktoken model file.
        
        Args:
            model_path: Path to the Tiktoken BPE model file (tokenizer.model)
        
        Raises:
            AssertionError: If the model file doesn't exist
        """
        # Verify that the model file exists
        assert os.path.isfile(model_path), model_path

        # Load the BPE (Byte Pair Encoding) mergeable ranks from the model file
        # This contains the vocabulary and merge rules for tokenization
        mergeable_ranks = load_tiktoken_bpe(model_path)
        
        # Get the number of base tokens (before adding special tokens)
        num_base_tokens = len(mergeable_ranks)
        
        # Define the list of special tokens used in Llama 3
        # These tokens have special meanings in the chat format
        special_tokens = [
            "<|begin_of_text|>",      # Start of text marker
            "<|end_of_text|>",        # End of text marker
            "<|reserved_special_token_0|>",  # Reserved tokens for future use
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",   # Start of message header (role)
            "<|end_header_id|>",      # End of message header
            "<|reserved_special_token_4|>",
            "<|eot_id|>",             # End of turn (end of message)
        ] + [
            # Generate remaining reserved tokens
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        
        # Create a dictionary mapping special token strings to their IDs
        # Token IDs start after the base tokens (num_base_tokens + index)
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        
        # Create the Tiktoken encoding object
        # This is the main tokenizer that performs encoding/decoding
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,      # Name of the encoding (from filename)
            pat_str=self.pat_str,            # Tokenization pattern
            mergeable_ranks=mergeable_ranks,  # BPE mergeable ranks
            special_tokens=self.special_tokens,  # Special token mappings
        )
        
        # Log that the tokenizer was loaded successfully
        logger.info(f"Reloaded tiktoken model from {model_path}")

        # Get the total vocabulary size (base tokens + special tokens)
        self.n_words: int = self.model.n_vocab
        
        # Get the ID of the beginning-of-sequence token
        # This token is prepended to sequences
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        
        # Get the ID of the end-of-sequence token
        # This token is appended to sequences
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        
        # Padding token ID (not used in this implementation, set to -1)
        self.pad_id: int = -1
        
        # Set of stop tokens that signal the end of generation
        # When the model generates one of these tokens, generation stops
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],  # End of text
            self.special_tokens["<|eot_id|>"],       # End of turn
        }
        
        # Log vocabulary information
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encode a string into a list of token IDs.
        
        Args:
            s: The input string to be encoded
            bos: Whether to prepend the beginning-of-sequence token
            eos: Whether to append the end-of-sequence token
            allowed_special: Special tokens that are allowed in the string
            disallowed_special: Special tokens that should raise an error if found
        
        Returns:
            List of token IDs representing the encoded string
        
        Note:
            By default, disallowed_special=() encodes special tokens as natural text.
            Setting allowed_special="all" treats all special tokens as special tokens.
        """
        # Verify that input is a string
        assert type(s) is str

        # Maximum number of characters that Tiktoken can handle in one call
        # Beyond this, we may get a pyo3_runtime.PanicException
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # Maximum consecutive whitespace or non-whitespace characters
        # This is a workaround for a Tiktoken issue with very long sequences
        # See: https://github.com/openai/tiktoken/issues/195
        MAX_NO_WHITESPACES_CHARS = 25_000

        # Split the string into manageable chunks
        # First, split by TIKTOKEN_MAX_ENCODE_CHARS
        # Then, further split by MAX_NO_WHITESPACES_CHARS to handle edge cases
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)  # Split into 400k char chunks
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        
        # Initialize list to store token IDs
        t: List[int] = []
        
        # Encode each substring and extend the token list
        for substr in substrs:
            # Encode the substring using Tiktoken
            # This converts the text to a list of token IDs
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,      # Which special tokens are allowed
                    disallowed_special=disallowed_special,  # Which special tokens raise errors
                )
            )
        
        # Prepend beginning-of-sequence token if requested
        if bos:
            t.insert(0, self.bos_id)
        
        # Append end-of-sequence token if requested
        if eos:
            t.append(self.eos_id)
        
        # Return the list of token IDs
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decode a list of token IDs into a string.
        
        Args:
            t: List of token IDs to be decoded
        
        Returns:
            The decoded string
        """
        # Cast to List[int] for type safety
        # Tiktoken's decode method accepts any sequence, but we cast for clarity
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Split a string so that each substring contains no more than max_consecutive_slice_len
        consecutive whitespaces or consecutive non-whitespaces.
        
        This is a workaround for Tiktoken's handling of very long sequences of
        whitespace or non-whitespace characters.
        
        Args:
            s: The string to split
            max_consecutive_slice_len: Maximum length of consecutive whitespace/non-whitespace
        
        Yields:
            Substrings of the input string
        """
        # Initialize tracking variables
        current_slice_len = 0  # Length of current consecutive sequence
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False  # Is current sequence whitespace?
        slice_start = 0  # Start index of current slice

        # Iterate through each character in the string
        for i in range(len(s)):
            # Check if current character is whitespace
            is_now_space = s[i].isspace()

            # If we've switched from whitespace to non-whitespace (or vice versa)
            # XOR (^) returns True if the states are different
            if current_slice_is_space ^ is_now_space:
                # Reset the slice length and update the state
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                # Same type of character, increment length
                current_slice_len += 1
                
                # If we've exceeded the maximum length, yield the current slice
                if current_slice_len > max_consecutive_slice_len:
                    # Yield substring from slice_start to current position
                    yield s[slice_start:i]
                    # Start a new slice at current position
                    slice_start = i
                    current_slice_len = 1
        
        # Yield the remaining substring
        yield s[slice_start:]


class ChatFormat:
    """
    Formats conversational dialogs into token sequences for the model.
    
    This class handles the special formatting required for chat-based interactions,
    including role headers and message boundaries.
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize the chat formatter.
        
        Args:
            tokenizer: The tokenizer instance to use for encoding
        """
        # Store the tokenizer for encoding messages
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        """
        Encode a message header (role) into token IDs.
        
        The header format is: <|start_header_id|>role<|end_header_id|>\n\n
        
        Args:
            message: The message containing the role to encode
        
        Returns:
            List of token IDs for the header
        """
        # Initialize list to store token IDs
        tokens = []
        
        # Add the start-of-header token
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        
        # Encode the role (e.g., "system", "user", "assistant")
        # bos=False: Don't add BOS token (we'll add it at the dialog level)
        # eos=False: Don't add EOS token (we'll add it at the message level)
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        
        # Add the end-of-header token
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        
        # Encode two newlines (separator between header and content)
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        
        # Return the header tokens
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        """
        Encode a complete message (header + content) into token IDs.
        
        The format is: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        
        Args:
            message: The message to encode
        
        Returns:
            List of token IDs for the complete message
        """
        # Encode the header (role)
        tokens = self.encode_header(message)
        
        # Encode the message content
        # .strip() removes leading/trailing whitespace
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        
        # Add the end-of-turn token to signal the end of this message
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        
        # Return the complete message tokens
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        """
        Encode a complete dialog (conversation) into token IDs.
        
        The format is:
        <|begin_of_text|>
        <|start_header_id|>role1<|end_header_id|>\n\ncontent1<|eot_id|>
        <|start_header_id|>role2<|end_header_id|>\n\ncontent2<|eot_id|>
        ...
        <|start_header_id|>assistant<|end_header_id|>\n\n
        
        The final assistant header is added to prompt the model to generate a response.
        
        Args:
            dialog: The conversation dialog (sequence of messages)
        
        Returns:
            List of token IDs for the complete dialog prompt
        """
        # Initialize list to store token IDs
        tokens = []
        
        # Add the beginning-of-text token to start the sequence
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        
        # Encode each message in the dialog
        for message in dialog:
            # Encode the complete message (header + content + EOT)
            tokens.extend(self.encode_message(message))
        
        # Add the start of an assistant message header
        # This prompts the model to generate a response
        # The content is empty because the model will generate it
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        
        # Return the complete dialog prompt tokens
        return tokens

