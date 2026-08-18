"""Regex pre-tokenization with a GPT-4/cl100k-style split pattern.

Split on word boundaries before BPE so merges don't cross words.
"""

import regex


# Adapted from minbpe's GPT4_SPLIT_PATTERN. This is only the split pattern;
# this project trains its own vocabulary and does not reproduce GPT-4 tokens.
# - common English contractions (case insensitive)
# - words (with an optional leading non-letter/non-number character)
# - numbers (1-3 digits at a time)
# - non-whitespace punctuation sequences
# - whitespace that includes newlines
# - remaining whitespace
GPT4_SPLIT_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def pre_tokenize(text, pattern=None):
    """Split text into chunks that BPE will process independently."""
    if pattern is None:
        pattern = GPT4_SPLIT_PATTERN
    return pattern.findall(text)


if __name__ == "__main__":
    test = "Hello, world! Don't you think it's great? 123 numbers."
    chunks = pre_tokenize(test)
    print(f"Input: '{test}'")
    print(f"Chunks ({len(chunks)}):")
    for c in chunks:
        print(f"  '{c}'")
