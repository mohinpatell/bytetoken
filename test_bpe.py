"""Focused tests for the byte-level BPE tokenizer."""

import os
import tempfile

from bpe import BPETokenizer
from pretokenize import pre_tokenize


def test_roundtrip():
    """Every encode->decode should give back the original text."""
    tok = BPETokenizer()
    text = "hello world! hello hello world. the cat sat on the mat."
    tok.train(text, vocab_size=280)

    test_cases = [
        "hello world",
        "the cat",
        "hello hello hello",
        "unknown text that wasn't in training",
        "numbers 12345 and symbols @#$",
        "",  # empty string
    ]

    for t in test_cases:
        encoded = tok.encode(t)
        decoded = tok.decode(encoded)
        assert decoded == t, f"roundtrip failed: '{t}' -> {encoded} -> '{decoded}'"
    print(f"roundtrip: all {len(test_cases)} cases passed")


def test_pretokenization_golden():
    """Keep the split pattern's observable boundaries stable."""
    text = "Hello, world! Don't STOP. 12345\n\nNext line."
    expected = [
        "Hello", ",", " world", "!", " Don", "'t", " STOP", ".", " ",
        "123", "45", "\n\n", "Next", " line", ".",
    ]
    chunks = pre_tokenize(text)
    assert chunks == expected, f"unexpected chunks: {chunks}"
    assert "".join(chunks) == text
    print("pretokenization golden: passed")


def test_unicode_roundtrip():
    """Round-trip multi-byte code points, emoji sequences, and newlines."""
    training_text = ("naïve café 漢字 👩🏽‍💻\r\n" * 8) + "plain ASCII"
    tok = BPETokenizer()
    tok.train(training_text, vocab_size=280)

    samples = [
        "naïve café",
        "漢字 and emoji 👩🏽‍💻",
        "unseen: Ελληνικά 🚀\n",
    ]
    for text in samples:
        assert tok.decode(tok.encode(text)) == text
    print("unicode roundtrip: passed")


def test_merge_order():
    """Encoding must replay learned merges by rank, including nested merges."""
    tok = BPETokenizer()
    tok.vocab = {i: bytes([i]) for i in range(256)}
    tok.merges = {
        (ord("a"), ord("b")): 256,
        (256, ord("c")): 257,
    }
    tok.vocab[256] = b"ab"
    tok.vocab[257] = b"abc"

    assert tok.encode("abcab") == [257, 256]
    assert tok.decode([257, 256]) == "abcab"
    print("merge order: passed")


def test_save_load():
    """Persist merge rank, vocabulary reconstruction, and special-token data."""
    tok = BPETokenizer()
    tok.train("low lower lowest low lower lowest " * 8, vocab_size=275)
    tok.add_special_token("<|endoftext|>")
    sample = "lowest lower unknown"
    expected_ids = tok.encode(sample)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tokenizer.model")
        tok.save(path)
        loaded = BPETokenizer()
        loaded.load(path)

    assert list(loaded.merges.items()) == list(tok.merges.items())
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.encode(sample) == expected_ids
    assert loaded.decode(expected_ids) == sample
    print("save/load: passed")


def run_optional_tiktoken_comparison():
    """Print an end-to-end reference comparison when tiktoken is installed."""
    try:
        import tiktoken
    except ModuleNotFoundError:
        print("tiktoken comparison: skipped (optional dependency not installed)")
        return

    samples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Don't you think it's a beautiful day?",
        "Numbers: 42, 3.14, 1000000",
    ]
    training_text = "\n".join(samples * 32)
    tok = BPETokenizer()
    tok.train(training_text, vocab_size=300)
    enc = tiktoken.get_encoding("cl100k_base")

    print("\nreference token counts (different vocabularies and training data):")
    print(f"{'text':>45} {'ours':>6} {'cl100k':>9}")
    print("-" * 64)

    for s in samples:
        our_tokens = len(tok.encode(s))
        tik_tokens = len(enc.encode(s))
        label = s[:42] + "..." if len(s) > 42 else s
        print(f"{label:>45} {our_tokens:>6} {tik_tokens:>9}")


if __name__ == "__main__":
    tests = [
        test_roundtrip,
        test_pretokenization_golden,
        test_unicode_roundtrip,
        test_merge_order,
        test_save_load,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")
    run_optional_tiktoken_comparison()
