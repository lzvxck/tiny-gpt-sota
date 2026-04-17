"""Train SentencePiece BPE tokenizer (32K vocab) on a proportional sample.

Samples ~500MB of text from all four datasets proportionally to their training
weight, then runs spm_train. Takes ~10-20 minutes on a modern CPU.

Usage:
    python scripts/20_train_tokenizer.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinysota.data.streaming import stream_dataset

SAMPLE_CHARS = 500_000_000  # 500MB of text for tokenizer training

DATASET_WEIGHTS = {
    "fineweb_edu": 0.60,
    "python_edu":  0.20,
    "openwebmath": 0.10,
    "cosmopedia":  0.10,
}

OUT_DIR = Path("data/tokenizer")
MODEL_PREFIX = str(OUT_DIR / "tinysota")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = OUT_DIR / "tokenizer_train.txt"
    if corpus_path.exists():
        print(f"Corpus already exists at {corpus_path}, skipping sampling.")
    else:
        print(f"Sampling {SAMPLE_CHARS/1e6:.0f}MB for tokenizer training...")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for dataset_name, weight in DATASET_WEIGHTS.items():
                budget = int(SAMPLE_CHARS * weight)
                written = 0
                print(f"  {dataset_name}: {budget/1e6:.0f}MB target")
                for text in stream_dataset(dataset_name):
                    encoded = text.encode("utf-8", errors="replace")
                    f.write(text + "\n")
                    written += len(encoded)
                    if written >= budget:
                        break
                print(f"  {dataset_name}: {written/1e6:.1f}MB written")

    print("Training SentencePiece BPE tokenizer (vocab=32000)...")
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=MODEL_PREFIX,
        vocab_size=32000,
        model_type="bpe",
        byte_fallback=True,
        character_coverage=0.9999,
        normalization_rule_name="identity",
        split_by_whitespace=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        user_defined_symbols=["<|im_start|>", "<|im_end|>", "<|pad|>"],
        pad_id=0,
        unk_id=1,
        bos_id=-1,     # no BOS
        eos_id=2,
        num_threads=max(1, os.cpu_count() - 2),
    )
    print(f"Tokenizer saved to {MODEL_PREFIX}.model + {MODEL_PREFIX}.vocab")

    # Quick sanity check
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{MODEL_PREFIX}.model")
    test = "Hello world! This is a test of the SentencePiece tokenizer."
    ids = sp.EncodeAsIds(test)
    decoded = sp.Decode(ids)
    print(f"Sanity check: '{test}' → {len(ids)} tokens → '{decoded}'")
    assert test == decoded, "Round-trip mismatch!"
    print("Tokenizer training complete.")


if __name__ == "__main__":
    main()
