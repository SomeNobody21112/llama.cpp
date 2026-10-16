import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from utils import *

# gguf-py from this repository, used to graft BGE-M3 style heads onto a small BERT model
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "gguf-py"))
import gguf  # noqa: E402

BASE_MODEL_URL = "https://huggingface.co/ggml-org/models/resolve/main/bert-bge-small/ggml-model-f16.gguf"

# repeated words: lexical weights keep the highest weight of each token
PROMPT = "the cat sat on the mat and the cat slept"


class Heads:
    def __init__(self, n_embd: int):
        rng = np.random.default_rng(42)
        self.colbert   = rng.normal(0.0, 0.05, (n_embd, n_embd)).astype(np.float32)
        self.colbert_b = rng.normal(0.0, 0.05, n_embd).astype(np.float32)
        self.sparse    = rng.normal(0.0, 0.05, (1, n_embd)).astype(np.float32)
        self.sparse_b  = np.array([0.01], dtype=np.float32)


@pytest.fixture(scope="module")
def models():
    """bge-small as is, and a copy of it with random ColBERT and sparse heads (the tensors BGE-M3 converts to)"""
    base = download_file(BASE_MODEL_URL, "./tmp/bert-bge-small-f16.gguf")
    with_heads = "./tmp/bert-bge-small-f16-m3-heads.gguf"

    reader = gguf.GGUFReader(base)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    heads = Heads(int(reader.fields[gguf.Keys.LLM.EMBEDDING_LENGTH.format(arch=arch)].contents()))
    token_types = reader.fields[gguf.Keys.Tokenizer.TOKEN_TYPE].contents()
    skipped = {i for i, t in enumerate(token_types) if t in (gguf.TokenType.CONTROL, gguf.TokenType.UNKNOWN)}

    writer = gguf.GGUFWriter(with_heads, arch=arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        sub_type = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), field.types[0], sub_type=sub_type)

    extra = {
        "colbert.weight": heads.colbert,
        "colbert.bias":   heads.colbert_b,
        "sparse.weight":  heads.sparse,
        "sparse.bias":    heads.sparse_b,
    }
    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, data in extra.items():
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    for data in extra.values():
        writer.write_tensor_data(data)
    writer.close()

    return base, with_heads, heads, skipped


def start_server(model_file: str, pooling: str) -> ServerProcess:
    server = ServerPreset.bert_bge_small()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = model_file
    server.pooling = pooling
    server.start()
    return server


def embed(model_file: str, pooling: str) -> tuple[Any, list[int]]:
    """the /embeddings result (token vectors, pooled vectors or lexical weights) and the prompt's tokens"""
    server = start_server(model_file, pooling)
    res = server.make_request("POST", "/embeddings", data={"input": PROMPT})
    assert res.status_code == 200
    tokens = server.make_request("POST", "/tokenize", data={"content": PROMPT, "add_special": True}).body["tokens"]
    server.stop()
    return res.body[0]["embedding"], tokens


def test_colbert_head_projects_token_embeddings(models):
    base, with_heads, heads, _ = models
    hidden, tokens = embed(base, "none")
    colbert, _ = embed(with_heads, "none")

    hidden, colbert = np.array(hidden), np.array(colbert)
    assert colbert.shape == hidden.shape == (len(tokens), heads.colbert.shape[0])
    np.testing.assert_allclose(colbert, hidden @ heads.colbert.T + heads.colbert_b, rtol=1e-3, atol=1e-3)


def test_sparse_head_lexical_weights(models):
    base, with_heads, heads, skipped = models
    hidden, tokens = embed(base, "none")
    weights, _ = embed(with_heads, "sparse")

    # relu(w h + b) per token, then the highest weight of each token id, without special tokens
    token_weights = np.maximum(np.array(hidden) @ heads.sparse[0] + heads.sparse_b[0], 0.0)
    expected = {}
    for tok, w in zip(tokens, token_weights):
        if tok not in skipped and w > 0:
            expected[str(tok)] = max(expected.get(str(tok), 0.0), float(w))

    assert 0 < len(expected) < len(set(tokens))  # the seed gives some positive and some zero weights
    assert weights.keys() == expected.keys()
    for tok, w in expected.items():
        assert weights[tok] == pytest.approx(w, rel=1e-3, abs=1e-4)


def test_heads_do_not_change_pooled_embeddings(models):
    base, with_heads, _, _ = models
    dense_base, _ = embed(base, "cls")
    dense_heads, _ = embed(with_heads, "cls")
    np.testing.assert_allclose(dense_heads, dense_base, rtol=1e-5, atol=1e-6)


def test_sparse_is_not_oai_compatible(models):
    _, with_heads, _, _ = models
    server = start_server(with_heads, "sparse")
    res = server.make_request("POST", "/v1/embeddings", data={"input": PROMPT})
    assert res.status_code == 400
    assert "not OAI compatible" in res.body["error"]["message"]


def test_sparse_requires_sparse_head(models):
    base, _, _, _ = models
    with pytest.raises(RuntimeError, match="Server process died"):
        start_server(base, "sparse")
