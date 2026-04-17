#!/usr/bin/env bash
# =============================================================================
# download_all.sh — Download and tokenize all pretraining datasets
#
# Centralized config: edit the TOKEN_* variables below, then run:
#   bash scripts/download_all.sh
#
# Each dataset streams from HuggingFace and writes packed uint16 .bin shards.
# No raw text is stored — only tokenized shards.
# =============================================================================

set -e  # stop on first error

# --------------- CONFIG -------------------------------------------------------
TOKENIZER="data/tokenizer/tinysota.model"
SHARD_SIZE=100_000_000       # tokens per shard (~200 MB each)

# Token caps — tune these to fit your disk and training budget.
# Rule of thumb: total across all datasets ≈ 1.2× your total_tokens in train config.
# At 2B Stage-A budget (60/20/10/10 mix), the training loop repeats shards by weight,
# so you need: fineweb≥1.2B, stackexchange≥400M, openwebmath≥200M, cosmopedia≥200M.
TOKENS_FINEWEB=1_500_000_000     # ~3 GB on disk
TOKENS_STACKEXCHANGE=500_000_000 # ~1 GB on disk
TOKENS_OPENWEBMATH=300_000_000   # ~600 MB on disk
TOKENS_COSMOPEDIA=300_000_000    # ~600 MB on disk
# Total: ~5.2 GB
# ------------------------------------------------------------------------------

export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "=================================================="
echo " tiny-sota-llm dataset download"
echo "=================================================="
echo " fineweb_edu:   $(echo "$TOKENS_FINEWEB" | tr -d '_') tokens"
echo " stackexchange: $(echo "$TOKENS_STACKEXCHANGE" | tr -d '_') tokens"
echo " openwebmath:   $(echo "$TOKENS_OPENWEBMATH" | tr -d '_') tokens"
echo " cosmopedia:    $(echo "$TOKENS_COSMOPEDIA" | tr -d '_') tokens"
echo "--------------------------------------------------"

echo ""
echo "[1/4] FineWeb-Edu..."
python scripts/10_download_fineweb_edu.py \
    --max_tokens $TOKENS_FINEWEB \
    --shard_size $SHARD_SIZE \
    --tokenizer  $TOKENIZER

echo ""
echo "[2/4] StackExchange (code + math Q&A)..."
python scripts/11_download_stack_edu.py \
    --max_tokens $TOKENS_STACKEXCHANGE \
    --shard_size $SHARD_SIZE \
    --tokenizer  $TOKENIZER

echo ""
echo "[3/4] OpenWebMath..."
python scripts/12_download_openwebmath.py \
    --max_tokens $TOKENS_OPENWEBMATH \
    --shard_size $SHARD_SIZE \
    --tokenizer  $TOKENIZER

echo ""
echo "[4/4] Cosmopedia v2..."
python scripts/13_download_cosmopedia.py \
    --max_tokens $TOKENS_COSMOPEDIA \
    --shard_size $SHARD_SIZE \
    --tokenizer  $TOKENIZER

echo ""
echo "=================================================="
echo " All datasets downloaded."
echo " Next: python scripts/30_pretrain.py"
echo "=================================================="
