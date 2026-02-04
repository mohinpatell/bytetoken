"""Train a BPE tokenizer on Shakespeare."""

import os
import urllib.request
from bpe import BPETokenizer


def get_shakespeare():
    path = os.path.join(os.path.dirname(__file__), "data", "input.txt")
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading to {path}...")
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        return f.read()


if __name__ == "__main__":
    text = get_shakespeare()
    print(f"training on {len(text):,} characters")

    tok = BPETokenizer()
    tok.train(text, vocab_size=500, verbose=True)

    # test roundtrip
    sample = "ROMEO:\nO, she doth teach the torches to burn bright!"
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"\n'{sample}'")
    print(f"-> {len(encoded)} tokens (was {len(sample.encode('utf-8'))} bytes)")
    print(f"-> decoded: '{decoded}'")
    assert decoded == sample, "roundtrip failed!"

    tok.save("shakespeare.model")
    print(f"\nsaved to shakespeare.model")
