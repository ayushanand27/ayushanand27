"""Token counting. Uses tiktoken (cl100k_base) when available, else a deterministic regex heuristic."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

log = logging.getLogger("tokenslicer.tokens")

_HEURISTIC = re.compile(r"[A-Za-z]+|\d{1,3}|[^\sA-Za-z\d]")


@lru_cache(maxsize=1)
def _encoder() -> tuple[Any | None, str]:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base"), "tiktoken:cl100k_base"
    except Exception as exc:  # missing package, or BPE file cannot be downloaded offline
        log.warning("tiktoken unavailable (%s); falling back to regex heuristic token counts", exc)
        return None, "heuristic:regex"


def tokenizer_name() -> str:
    return _encoder()[1]


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc, _ = _encoder()
    if enc is not None:
        # disallowed_special=() so source code containing "<|endoftext|>" cannot raise.
        return len(enc.encode(text, disallowed_special=()))
    return len(_HEURISTIC.findall(text))
