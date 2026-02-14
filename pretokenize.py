"""Regex pre-tokenization (GPT-2 style).

Split on word boundaries before BPE so merges don't cross words.
"""

import regex


# GPT-2's pre-tokenization pattern:
# - contractions: 's, 't, 're, 've, 'm, 'll, 'd (case insensitive)
# - words (with optional leading space)
# - numbers (1-3 digits at a time)
# - non-whitespace punctuation sequences
# - whitespace that includes newlines
# - remaining whitespace
GPT2_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


def pre_tokenize(text, pattern=None):
    """Split text into chunks that BPE will process independently."""
    if pattern is None:
        pattern = GPT2_PATTERN
    return pattern.findall(text)


if __name__ == "__main__":
    test = "Hello, world! Don't you think it's great? 123 numbers."
    chunks = pre_tokenize(test)
    print(f"Input: '{test}'")
    print(f"Chunks ({len(chunks)}):")
    for c in chunks:
        print(f"  '{c}'")
