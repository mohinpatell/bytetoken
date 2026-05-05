# bytetoken

A byte level BPE tokenizer with GPT-2 style regex pre-tokenization. Trains, encodes, decodes, saves, loads. ~250 lines of Python.

After [tinylm](https://github.com/mohinpatell/tinylm), where I trained a transformer on a 65 character vocab, I wanted to know what real models actually do at the input layer. The answer is byte pair encoding with a regex pre-pass, and once you've written it once it stops feeling like magic.

## How it works

```
text
  |
  v  GPT-2 regex (split on word boundaries, keep contractions intact)
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

1. **Pre tokenize** with the GPT-2 regex so merges can't bleed across word boundaries (`'ll`, `n't`, leading spaces, runs of digits, etc).
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
pretokenize.py  the GPT-2 regex, with a comment explaining each clause.
train.py        train on Shakespeare, save the .model file, roundtrip check.
test_bpe.py     roundtrip cases, contraction handling, and a tiktoken sanity comparison.
demo.ipynb      first 30 merges learned, compression vs vocab size, GPT-2 comparison.
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

Two things to notice. First, leading-space tokens dominate the early merges — that's exactly the GPT-2 regex doing its job, splitting `" the"` off as one chunk so the merge captures word + leading-space together. Second, merge 13 (`' t'` + `'he'` -> `' the'`) is a *recursive* merge: it consumes two earlier merge results. This is the part of BPE that surprises people the first time. Vocabulary is built bottom up.

## Compression

Same Shakespeare corpus, ~1.1M characters:

| vocab | tokens to encode the corpus | bytes/token |
|------:|----------------------------:|------------:|
|   256 | 1,115,394                   | 1.00 |
|   300 |   791,019                   | 1.41 |
|   500 |   553,176                   | 2.02 |
|  1000 |   431,132                   | 2.59 |
|  2000 |   357,642                   | 3.12 |

![compression](compression.png)

Diminishing returns kick in fast: doubling the vocab from 500 to 1000 only buys ~28% fewer tokens, and from 1000 to 2000 another ~17%.

## vs tiktoken

Same text, our tokenizer at `vocab=500` against `tiktoken.get_encoding("gpt2")` (50,257 vocab):

|                                                | ours | tiktoken |
|-----------------------------------------------:|-----:|---------:|
| `Hello, world!`                                |    8 |        4 |
| `The quick brown fox jumps over the lazy dog.` |   27 |       10 |
| `Don't you think it's a beautiful day?`        |   19 |       10 |
| `ROMEO:\nO, she doth teach the torches to burn bright!` | 29 | 16 |

We lose by about 2x. That's the right ballpark — GPT-2 has 100x our vocab and is trained on 100x more text — but it's a useful number to look at, because the gap is *smaller* on Shakespeare-flavored text than on modern English. Train BPE on the domain you'll tokenize, and a tiny vocab eats most of the loss.

The pre tokenization should match GPT-2's exactly, since I copied their regex. The merge inventory is what differs.

## Things that broke

**Encode applied merges by current frequency, not by training rank.** This is the canonical BPE bug and I walked right into it. The naive thing — and the thing I wrote first — is "to encode, find the most frequent pair in the input, merge it, repeat." This *feels* symmetric with training. It's wrong: the input text has different statistics from the training corpus, so you'll merge in a different order and the same input encodes to different ids on different runs. The fix is to iterate `self.merges.items()` in insertion order (Python 3.7+ guarantees this) and apply each merge greedily across the token list. The comment in `bpe.py:75` is the reminder I left for myself: *"got bitten by this — if you merge greedily by frequency, you get different tokenizations than training."*

**Special tokens half work.** `add_special_token` registers an id and `decode` produces the right string back, but `encode("foo<|endoftext|>bar")` does not recognize the special string — it just byte-encodes the literal angle-bracket goo into the regular merge stream. To do this properly you'd split the text on registered specials first, encode the surrounding chunks normally, and stitch the special ids in by hand. I left the half-implementation in because it's *almost* what you want — but if you encode a string that contains an `<|endoftext|>` literal you should know what you'll get back is not what you'd want.

## What's not here

- **Speed.** Training is O(num_merges × corpus_size) the way I wrote it: every merge re-scans the whole token list to count pairs. tiktoken-grade implementations use a priority queue and dirty-set tracking to amortize this. On 1MB of Shakespeare this finishes in seconds for vocab 500 and minutes for vocab 5000 — fine for learning, useless for production.
- **A sentencepiece style unigram model.** BPE only.
- **Caching.** No on-disk merge cache for repeated `encode` calls on the same text.
- **A real benchmark against tiktoken's wall-clock.** I know it loses; I haven't measured by how much.

## How to run

```bash
pip install -r requirements.txt   # just `regex`. tiktoken if you want test_bpe.py
python train.py                   # downloads shakespeare, trains, saves shakespeare.model
python test_bpe.py                # roundtrip + tiktoken sanity check
```

## References

- *Neural Machine Translation of Rare Words with Subword Units* — Sennrich, Haddow, Birch, 2016. The original BPE paper.
- *Language Models Are Unsupervised Multitask Learners* — Radford et al., 2019. Source of the regex pre tokenization pattern I copied.
- Karpathy's [minbpe](https://github.com/karpathy/minbpe). Same approach, cleaner code than mine.
