"""Byte-pair encoding tokenizer with regex pre-tokenization."""

import json
from pretokenize import pre_tokenize


class BPETokenizer:

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}   # int -> bytes
        self.special_tokens = {}  # str -> int

    def _get_pair_counts(self, token_lists):
        """Count adjacent pairs across all token sequences."""
        counts = {}
        for tokens in token_lists:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, token_lists, pair, new_id):
        """Replace all occurrences of pair with new_id."""
        result = []
        for tokens in token_lists:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            result.append(new_tokens)
        return result

    def train(self, text, vocab_size=500, verbose=False):
        """Train BPE on text. Learns merges until vocab reaches vocab_size."""
        assert vocab_size > 256, "vocab_size must be > 256 (base byte vocab)"

        # pre-tokenize: split on word boundaries so merges don't cross words
        chunks = pre_tokenize(text)

        # convert each chunk to bytes
        token_lists = [list(chunk.encode("utf-8")) for chunk in chunks]

        # base vocab: one token per byte value
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        num_merges = vocab_size - 256
        for i in range(num_merges):
            counts = self._get_pair_counts(token_lists)
            if not counts:
                break

            best_pair = max(counts, key=counts.get)
            new_id = 256 + i

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            token_lists = self._merge_pair(token_lists, best_pair, new_id)

            if verbose and (i + 1) % 50 == 0:
                print(f"merge {i+1}/{num_merges}: {best_pair} -> {new_id} "
                      f"('{self.vocab[new_id].decode('utf-8', errors='replace')}')")

        if verbose:
            print(f"trained {len(self.merges)} merges, vocab size: {len(self.vocab)}")

    def _encode_chunk(self, chunk_bytes):
        """Encode a single pre-tokenized chunk."""
        tokens = list(chunk_bytes)

        # important: apply merges in the order they were learned during training,
        # NOT by frequency in the current text. got bitten by this — if you
        # merge greedily by frequency, you get different tokenizations than training.
        for pair, new_id in self.merges.items():
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = new_id
                    del tokens[i + 1]
                else:
                    i += 1
        return tokens

    def encode(self, text):
        """Encode text to token ids."""
        chunks = pre_tokenize(text)
        tokens = []
        for chunk in chunks:
            tokens.extend(self._encode_chunk(chunk.encode("utf-8")))
        return tokens

    def decode(self, tokens):
        """Decode token ids back to text."""
        raw = b"".join(self.vocab[t] for t in tokens)
        return raw.decode("utf-8", errors="replace")

    def add_special_token(self, token_str):
        """Register a special token (like <|endoftext|>)."""
        new_id = len(self.vocab)
        self.special_tokens[token_str] = new_id
        self.vocab[new_id] = token_str.encode("utf-8")
        return new_id

    def save(self, path):
        data = {
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair_str, new_id in data["merges"].items():
            a, b = pair_str.split(",")
            pair = (int(a), int(b))
            self.merges[pair] = int(new_id)
            self.vocab[int(new_id)] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.special_tokens = data.get("special_tokens", {})
        for tok_str, tok_id in self.special_tokens.items():
            self.vocab[tok_id] = tok_str.encode("utf-8")
