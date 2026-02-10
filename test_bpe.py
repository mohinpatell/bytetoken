"""Test our BPE against tiktoken (GPT-2's tokenizer)."""

import tiktoken
from bpe import BPETokenizer
from pretokenize import pre_tokenize


def test_roundtrip():
    """Every encode->decode should give back the original text."""
    tok = BPETokenizer()

    # train on a small corpus
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


def test_pretokenization():
    """Verify pre-tokenization splits correctly."""
    chunks = pre_tokenize("Hello, world! Don't stop.")
    # "Don't" should split into "Don" + "'t"
    assert "'t" in chunks, f"contraction not split: {chunks}"
    print(f"pretokenization: contractions handled correctly")


def test_compression():
    """Compare compression at different vocab sizes."""
    import os
    path = os.path.join(os.path.dirname(__file__), "data", "input.txt")
    if not os.path.exists(path):
        print("skipping compression test (no data)")
        return

    with open(path) as f:
        text = f.read()

    raw_bytes = len(text.encode("utf-8"))
    print(f"\ncompression analysis on {raw_bytes:,} bytes of shakespeare:")
    print(f"{'vocab_size':>12} {'tokens':>10} {'bytes/token':>12} {'ratio':>8}")
    print("-" * 46)

    for vs in [300, 500, 1000, 2000, 5000]:
        tok = BPETokenizer()
        tok.train(text, vocab_size=vs)
        n_tokens = len(tok.encode(text))
        bpt = raw_bytes / n_tokens
        ratio = raw_bytes / n_tokens
        print(f"{vs:>12} {n_tokens:>10,} {bpt:>12.2f} {ratio:>7.2f}x")


def test_vs_tiktoken():
    """Compare our tokenizer against tiktoken on the same text."""
    enc = tiktoken.get_encoding("gpt2")

    samples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Don't you think it's a beautiful day?",
        "Numbers: 42, 3.14, 1000000",
    ]

    print(f"\nours vs tiktoken (gpt2):")
    print(f"{'text':>45} {'ours':>6} {'tiktoken':>9}")
    print("-" * 64)

    for s in samples:
        our_tokens = len(pre_tokenize(s))  # just compare pre-tokenization chunks
        tik_tokens = len(enc.encode(s))
        label = s[:42] + "..." if len(s) > 42 else s
        print(f"{label:>45} {our_tokens:>6} {tik_tokens:>9}")

    print("\n(token counts differ because vocab sizes are different,")
    print(" but pre-tokenization splits should be similar)")


if __name__ == "__main__":
    test_roundtrip()
    test_pretokenization()
    test_compression()
    test_vs_tiktoken()
