"""Embedders.

``HashingEmbedder`` is the default so everything runs offline: signed feature
hashing of word unigrams/bigrams and character trigrams, L2-normalised. It
captures lexical overlap, not meaning. ``OpenAIEmbedder`` calls any
OpenAI-compatible ``/v1/embeddings`` endpoint (vLLM can serve embedding models).
"""
from __future__ import annotations

import hashlib
import re

import httpx
import numpy as np

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its me my of on or our so that the "
    "their then there this to until was we what when where which who why will with you your".split()
)


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    @staticmethod
    def _stem(w: str) -> str:
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = [self._stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]
        feats = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.5) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.3) for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += weight if (h >> 63) & 1 else -weight
            n = np.linalg.norm(out[row])
            if n:
                out[row] /= n
        return out


class OpenAIEmbedder:
    def __init__(self, base_url: str, model: str, dim: int, api_key: str = "", transport: httpx.BaseTransport | None = None):
        self.dim = dim
        self.name = model
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=30, transport=transport)
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.post("/embeddings", json={"model": self.model, "input": texts})
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        arr = np.array([d["embedding"] for d in data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)
