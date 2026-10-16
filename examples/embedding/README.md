# llama.cpp/example/embedding

This example demonstrates generate high-dimensional embedding vector of a given text with llama.cpp.

## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:

### Unix-based systems (Linux, macOS, etc.):

```bash
./llama-embedding -m ./path/to/model --pooling mean --log-disable -p "Hello World!" 2>/dev/null
```

### Windows:

```powershell
llama-embedding.exe -m ./path/to/model --pooling mean --log-disable -p "Hello World!" 2>$null
```

The above command will output space-separated float values.

## extra parameters
### --embd-normalize $integer$
| $integer$ | description         | formula |
|-----------|---------------------|---------|
| $-1$      | none                |
| $0$       | max absolute int16  | $\Large{{32760 * x_i} \over\max \lvert x_i\rvert}$
| $1$       | taxicab             | $\Large{x_i \over\sum \lvert x_i\rvert}$
| $2$       | euclidean (default) | $\Large{x_i \over\sqrt{\sum x_i^2}}$
| $>2$      | p-norm              | $\Large{x_i \over\sqrt[p]{\sum \lvert x_i\rvert^p}}$

### --embd-output-format $'string'$
| $'string'$ | description                  |  |
|------------|------------------------------|--|
| ''         | same as before               | (default)
| 'array'    | single embeddings            | $[[x_1,...,x_n]]$
|            | multiple embeddings          | $[[x_1,...,x_n],[x_1,...,x_n],...,[x_1,...,x_n]]$
| 'json'     | openai style                 |
| 'json+'    | add cosine similarity matrix |
| 'raw'      | plain text output            |

### --embd-separator $"string"$
| $"string"$   | |
|--------------|-|
| "\n"         | (default)
| "<#embSep#>" | for example
| "<#sep#>"    | other example

## examples
### Unix-based systems (Linux, macOS, etc.):

```bash
./llama-embedding -p 'Castle<#sep#>Stronghold<#sep#>Dog<#sep#>Cat' --pooling mean --embd-separator '<#sep#>' --embd-normalize 2  --embd-output-format '' -m './path/to/model.gguf' --n-gpu-layers 99 --log-disable 2>/dev/null
```

### Windows:

```powershell
llama-embedding.exe -p 'Castle<#sep#>Stronghold<#sep#>Dog<#sep#>Cat' --pooling mean --embd-separator '<#sep#>' --embd-normalize 2  --embd-output-format '' -m './path/to/model.gguf' --n-gpu-layers 99 --log-disable 2>/dev/null
```

## BGE-M3: dense, sparse and ColBERT retrieval

[BGE-M3](https://huggingface.co/BAAI/bge-m3) has three retrieval heads on one encoder. `convert_hf_to_gguf.py` picks up its `colbert_linear.pt` and `sparse_linear.pt` automatically, and `--pooling` selects the head:

| `--pooling` | output | same as FlagEmbedding's |
|-------------|--------|-------------------------|
| `cls` (default) | one dense vector per text | `dense_vecs` |
| `sparse` | a weight per token, the highest per token id, without special tokens | `lexical_weights` |
| `none` | one ColBERT vector per token | `colbert_vecs` (drop the first row, which belongs to `[CLS]`) |

```bash
python convert_hf_to_gguf.py ./bge-m3 --outtype f16 --outfile bge-m3-f16.gguf

./llama-embedding -m bge-m3-f16.gguf --pooling sparse --embd-separator '<#sep#>' \
    -p 'What is BGE M3?<#sep#>BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.<#sep#>BM25 is a bag-of-words retrieval function'
```

```
lexical weights 0: '3'=0.2693 'GE'=0.2517 ' M'=0.1698 ' B'=0.1294 ' What'=0.0835 ' is'=0.0813 '?'=0.0407
...

lexical matching score matrix:

What i BGE M3 BM25 i
 0.197  0.195  0.008 What is BG
 0.195  0.599  0.048 BGE M3 is
 0.008  0.048  0.394 BM25 is a
```

The lexical matching score of two texts is the sum of `w_q * w_d` over the tokens they share. With `--embd-output-format json`, each `embedding` is an object of token ids and weights, the format `llama-server` returns from `/embeddings` with `--pooling sparse`.

For ColBERT, the score of a query and a document is the mean, over the query tokens, of the best dot product with any document token. `llama-embedding` normalizes each token vector (`--embd-normalize 2`, the default); `llama-server` returns them unnormalized, so normalize each row first, as FlagEmbedding does.

The model card combines the three scores as `0.4 * dense + 0.2 * sparse + 0.4 * colbert`. Each head needs its own context, so run one `llama-server` per pooling type to serve all three.
