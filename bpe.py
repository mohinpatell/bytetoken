"""Byte-pair encoding tokenizer."""

import json


class BPETokenizer:

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}   # int -> bytes

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

        # start with raw bytes
        raw_bytes = text.encode("utf-8")
        tokens = list(raw_bytes)
        token_lists = [tokens]

        # base vocab: one token per byte value
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        num_merges = vocab_size - 256
        for i in range(num_merges):
            counts = self._get_pair_counts(token_lists)
            if not counts:
                break

            # most frequent pair
            best_pair = max(counts, key=counts.get)
            new_id = 256 + i

            # record the merge
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # apply merge
            token_lists = self._merge_pair(token_lists, best_pair, new_id)

            if verbose and (i + 1) % 50 == 0:
                print(f"merge {i+1}/{num_merges}: {best_pair} -> {new_id} "
                      f"('{self.vocab[new_id].decode('utf-8', errors='replace')}')")

        if verbose:
            print(f"trained {len(self.merges)} merges, vocab size: {len(self.vocab)}")

    def encode(self, text):
        """Encode text to token ids using learned merges."""
        tokens = list(text.encode("utf-8"))

        # apply merges in the order they were learned
        for pair, new_id in self.merges.items():
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = new_id
                    del tokens[i + 1]
                else:
                    i += 1
        return tokens

    def decode(self, tokens):
        """Decode token ids back to text."""
        raw = b"".join(self.vocab[t] for t in tokens)
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        """Save merges and vocab to file."""
        data = {
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        """Load merges from file, rebuild vocab."""
        with open(path) as f:
            data = json.load(f)
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair_str, new_id in data["merges"].items():
            a, b = pair_str.split(",")
            pair = (int(a), int(b))
            self.merges[pair] = int(new_id)
            self.vocab[int(new_id)] = self.vocab[pair[0]] + self.vocab[pair[1]]
