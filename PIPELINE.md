# tiny-sota-llm — Full Pipeline Guide

End-to-end command reference for reproducing the pipeline from scratch.

---

## Minimum Requirements

| Component | Minimum | Tested on |
|-----------|---------|-----------|
| GPU VRAM | 8 GB | 12 GB (RTX 5070) |
| CUDA | 12.x | 12.8 |
| Python | 3.10+ | 3.12 |
| Disk | 10 GB free | — |
| RAM | 16 GB | 32 GB |

> **VRAM note:** 8 GB is the minimum with `gradient_checkpointing: true` (default).
> Without it, peak VRAM is ~12 GB and training will OOM on 8 GB cards.

> **CUDA note:** The `requirements.txt` installs PyTorch nightly with CUDA 12.8,
> required for Blackwell (RTX 50xx). For Ampere/Ada (RTX 30xx/40xx), replace
> `cu128` with `cu121` or `cu124` and use the stable PyTorch release instead.

---

## Step 0 — Install dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# PyTorch with CUDA 12.8 — for Blackwell (RTX 50xx)
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --no-cache-dir

# FlashAttention-2 prebuilt wheel — Windows, Python 3.12, CUDA 12.8 (optional)
# SDPA works without this; this gives a small extra speedup.
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.6/flash_attn-2.8.3%2Bcu128torch2.11-cp312-cp312-win_amd64.whl"

# All other dependencies
pip install -r requirements.txt
```

---

## Step 1 — Verify GPU environment

```bash
python scripts/00_verify_env.py
```

**Expected output:**
- `[OK] CUDA available` — GPU detected
- `[OK] Throughput: 20000+ tok/s` — acceptable throughput
- `[OK] Peak VRAM: < 9 GB` — safe headroom
- `[OK] Projected: > 1.5B tokens/day`

Fix any failures before continuing.

---

## Step 2 — Train tokenizer

```bash
python scripts/20_train_tokenizer.py
```

- Streams ~500 MB of text from all 4 datasets
- Trains a BPE tokenizer with 32K vocabulary
- Saves to `data/tokenizer/tinysota.model`
- Takes ~15–20 minutes

---

## Step 3 — Download and tokenize datasets

```bash
bash scripts/download_all.sh
```

Or run each individually:

```bash
# [1/4] FineWeb-Edu — ~3 GB, ~1-2h
python scripts/10_download_fineweb_edu.py --max_tokens 1_500_000_000

# [2/4] StackExchange (code + math Q&A) — ~1 GB, ~30-45min
python scripts/11_download_stack_edu.py --max_tokens 500_000_000

# [3/4] OpenWebMath — ~600 MB, ~20-30min
python scripts/12_download_openwebmath.py --max_tokens 300_000_000

# [4/4] Cosmopedia v2 — ~600 MB, ~20-30min
python scripts/13_download_cosmopedia.py --max_tokens 300_000_000
```

**Total disk:** ~5.2 GB | **Total time:** ~3–4 hours

To change token caps, edit the variables at the top of `scripts/download_all.sh`.

---

## Step 4 — Stage A pretraining

```bash
python scripts/30_pretrain.py --config configs/train_stage_a.yaml
```

**Config summary (`configs/train_stage_a.yaml`):**
- 2B tokens, seq_len=2048, effective batch=256K tokens
- Optimizer: hybrid AdamW + Muon
- bf16 + gradient checkpointing + torch.compile
- **Estimated time: ~21 hours**
- Checkpoints saved every 2000 steps to `checkpoints/stage_a/`
- Logs to WandB project `tiny-sota-llm`

**If interrupted, resume with:**
```bash
python scripts/30_pretrain.py --config configs/train_stage_a.yaml --resume
```

---

## Step 5 — Evaluate

```bash
# Perplexity on a validation corpus
python scripts/40_eval_perplexity.py --ckpt checkpoints/stage_a/latest.txt

# Standard benchmarks (lm-harness): HellaSwag, ARC, WinoGrande, etc.
python scripts/41_eval_lm_harness.py --ckpt checkpoints/stage_a/latest.txt
```

---

## (Optional) Step 6 — Stage B: context extension to 4K

Only if Stage A results look good and you have another day available.

```bash
python scripts/31_context_extension.py --config configs/train_stage_b.yaml
```

- 2B additional tokens, seq_len=4096, NTK-aware RoPE scaling
- Resumes from the Stage A checkpoint
- **Estimated time: ~42 hours** (slower due to 2× context length)

---

## Quick reference

```
00_verify_env.py          GPU OK?
20_train_tokenizer.py     32K BPE tokenizer
download_all.sh           ~5 GB tokenized data
30_pretrain.py            ~21h Stage A (2B tokens)
40_eval_perplexity.py     perplexity
41_eval_lm_harness.py     benchmarks
```

---

## Project structure

```
configs/
  model_150m.yaml          ~145M param model config
  train_stage_a.yaml       Stage A training config
  train_stage_b.yaml       Stage B training config (optional)
data/
  tokenizer/tinysota.model BPE 32K tokenizer
  tokenized/               packed uint16 shards (.bin)
    fineweb_edu/train/
    python_edu/train/
    openwebmath/train/
    cosmopedia/train/
checkpoints/
  stage_a/                 checkpoints + latest.txt pointer
scripts/
  00_verify_env.py
  10-13_download_*.py
  20_train_tokenizer.py
  30_pretrain.py
  31_context_extension.py
  40_eval_perplexity.py
  41_eval_lm_harness.py
  download_all.sh
tinysota/                  model and training source code
knowledge.md               concept reference (what and why)
PIPELINE.md                this file
```
