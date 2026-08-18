# bytetoken

A byte-level BPE tokenizer with a GPT-4/cl100k-style regex pre-tokenization pass. It trains its own merge vocabulary, encodes, decodes, saves, and loads in a few small Python files.

After [tinylm](https://github.com/mohinpatell/tinylm), where I trained a transformer on a 65-character vocabulary, I wanted to understand a common modern input pipeline: regex segmentation followed by byte-level BPE. Once you've implemented the pipeline, the tokenizer stops feeling like a black box.

## How it works

```
text
  |
  v  cl100k-style split regex (separate letters, numbers, punctuation, whitespace)
  v
chunks
  |
  v  utf-8 bytes per chunk
  v
[ list[int] per chunk ]
  |
  v  greedy merge of the most frequent adjacent pair, repeat until vocab_size
  v
[ merged token ids ]
```

Three steps:

1. **Pre-tokenize** with a split pattern adapted from minbpe's `GPT4_SPLIT_PATTERN`, so merges cannot cross its letter, number, punctuation, and whitespace boundaries. This is not GPT-2's regex, and the split pattern alone does not reproduce GPT-4 token IDs.
2. **Train** by counting adjacent byte pairs across all chunks, merging the most frequent one, and repeating until you hit your vocab budget.
3. **Encode** by replaying the learned merges *in the order they were learned*. Not by frequency in the new text. This is a real footgun (see below).

Usage:

```python
from bpe import BPETokenizer

tok = BPETokenizer()
tok.train(open("data/input.txt").read(), vocab_size=500, verbose=True)
tok.save("shakespeare.model")

ids = tok.encode("Hello, world!")
back = tok.decode(ids)
assert back == "Hello, world!"
```

## What's in here

```
bpe.py          BPETokenizer: train / encode / decode / save / load / special tokens.
pretokenize.py  the GPT-4/cl100k-style split pattern and clause notes.
train.py        train on Shakespeare, save the .model file, roundtrip check.
test_bpe.py     golden boundaries, Unicode, merge-order, save/load, and roundtrip tests.
demo.ipynb      first 30 merges learned, compression vs vocab size, reference comparison.
```

## What BPE actually learns

Trained on Shakespeare with `vocab_size=500` — 244 merges on top of the 256 byte base vocab. The first 15 merges, in order:

```
 1  ' ' + 't'   ->  ' t'
 2  'h' + 'e'   ->  'he'
 3  ' ' + 'a'   ->  ' a'
 4  'o' + 'u'   ->  'ou'
 5  ' ' + 's'   ->  ' s'
 6  ' ' + 'm'   ->  ' m'
 7  'i' + 'n'   ->  'in'
 8  ' ' + 'w'   ->  ' w'
 9  'r' + 'e'   ->  're'
10  'h' + 'a'   ->  'ha'
11  ':' + '\n'  ->  ':\n'
12  'n' + 'd'   ->  'nd'
13  ' t' + 'he' ->  ' the'
14  ' ' + 'b'   ->  ' b'
15  'i' + 's'   ->  'is'
```

Two things to notice. First, leading-space tokens dominate the early merges because the split pattern keeps a leading space with many letter chunks, allowing a merge to capture the space and word together. Second, merge 13 (`' t'` + `'he'` -> `' the'`) is a *recursive* merge: it consumes two earlier merge results. Vocabulary is built bottom up.

## Compression

Measured by training and then encoding the 1,115,394-byte Tiny Shakespeare corpus:

| vocab | tokens to encode the corpus | bytes/token |
|------:|----------------------------:|------------:|
|   256 | 1,115,394                   | 1.00 |
|   300 |   791,019                   | 1.41 |
|   500 |   553,176                   | 2.02 |
|  1000 |   431,132                   | 2.59 |
|  2000 |   357,642                   | 3.12 |

![compression](compression.png)

Diminishing returns show up quickly in this run: increasing the vocabulary from 500 to 1,000 produces 22.1% fewer tokens, and increasing it from 1,000 to 2,000 produces another 17.0% reduction.

## Reference token counts

The tokenizer trained above at `vocab_size=500`, compared end to end on the same strings with `tiktoken.get_encoding("cl100k_base")`:

|                                                | ours | cl100k_base |
|-----------------------------------------------:|-----:|------------:|
| `Hello, world!`                                |    8 |        4 |
| `The quick brown fox jumps over the lazy dog.` |   27 |       10 |
| `Don't you think it's a beautiful day?`        |   19 |       10 |
| `ROMEO:\nO, she doth teach the torches to burn bright!` | 29 | 16 |

This is a reference comparison, not a quality benchmark: the vocabulary sizes and training corpora are radically different. The useful fact is narrower—the table measures both complete tokenizers' output counts, rather than comparing this project's pre-token chunks with another tokenizer's final tokens.

## Things that broke

**Encode applied merges by current frequency, not by training rank.** The naive encoder—"find the most frequent pair in this input, merge it, repeat"—looks symmetric with training but uses the new sample's statistics instead of the learned merge ranks. The fix is to replay `self.merges` in insertion order and apply each learned merge greedily. `test_merge_order` locks down a nested merge whose result depends on that ordering.

**Special-token registration is decode-only.** `add_special_token` registers an id and `decode` reconstructs its string, but `encode("foo<|endoftext|>bar")` treats the special-token text as ordinary input. Full support would require explicitly separating allowed special tokens before normal encoding.

## What's not here

- **Speed.** Training performs a full pair-count and merge pass for every learned token, so runtime grows quickly with both corpus size and vocabulary size. Production tokenizers use substantially more efficient data structures; this implementation is intentionally educational.
- **A SentencePiece-style unigram model.** BPE only.
- **Caching.** No on-disk merge cache for repeated `encode` calls on the same text.
- **A wall-clock benchmark against tiktoken.** Token counts are measured above; runtime is not.

## How to run

```bash
pip install -r requirements.txt   # just `regex`
python train.py                   # downloads shakespeare, trains, saves shakespeare.model
python test_bpe.py                # five focused tests; optional tiktoken comparison skips cleanly

# optional reference comparison
pip install tiktoken
python test_bpe.py
```

## References

- *Neural Machine Translation of Rare Words with Subword Units* — Sennrich, Haddow, Birch, 2016. Introduced BPE subwords for open-vocabulary neural machine translation.
- OpenAI's [tiktoken encoding definitions](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py), which show how GPT-2 and cl100k use different split patterns.
- Karpathy's [minbpe](https://github.com/karpathy/minbpe), the source of the `GPT4_SPLIT_PATTERN` adapted here.
