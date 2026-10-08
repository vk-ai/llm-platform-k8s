"""Rough token estimate (no tokenizer dependency): ~4 characters per token, min 1 per word."""
from __future__ import annotations


def estimate_tokens(text: str) -> int:
    return max(len(text.split()), (len(text) + 3) // 4)


def messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) + 4 for m in messages) + 2
