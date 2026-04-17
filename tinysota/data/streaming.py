"""Thin wrappers around HuggingFace datasets for streaming raw text."""
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset


DATASET_CONFIGS = {
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",   # start with 10BT sample subset — adjust as needed
        "split": "train",
        "text_field": "text",
    },
    "python_edu": {
        # StackExchange contains code-heavy Q&A (Python, math, CS) — good code-adjacent pretraining
        "path": "allenai/dolmino-mix-1124",
        "name": "stackexchange",
        "split": "train",
        "text_field": "text",
    },
    "openwebmath": {
        "path": "open-web-math/open-web-math",
        "name": None,
        "split": "train",
        "text_field": "text",
    },
    "cosmopedia": {
        "path": "HuggingFaceTB/smollm-corpus",
        "name": "cosmopedia-v2",
        "split": "train",
        "text_field": "text",
    },
}


def stream_dataset(name: str, max_docs: int | None = None) -> Iterator[str]:
    """Stream text documents from a named dataset config."""
    cfg = DATASET_CONFIGS[name]
    kwargs = dict(split=cfg["split"], streaming=True)
    if cfg["name"] is not None:
        kwargs["name"] = cfg["name"]
    ds = load_dataset(cfg["path"], **kwargs)

    field = cfg["text_field"]
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        text = row.get(field, "")
        if text:
            yield text
