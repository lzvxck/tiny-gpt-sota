# tiny-sota-llm

A ~145M parameter language model trained from scratch on a single consumer GPU (RTX 5070, 12 GB VRAM).

The goal is not to beat GPT-4 — it's to build every piece of a modern LLM pipeline by hand and understand *why* each decision exists. Every concept used here is documented in [`knowledge.md`](knowledge.md).

---

## What this is

A full pretraining pipeline covering:

- **Architecture** — Llama-style decoder-only transformer with modern components
- **Data pipeline** — streaming tokenization into packed binary shards, no padding waste
- **Training** — hybrid optimizer, mixed precision, gradient checkpointing, compiled model
- **Evaluation** — perplexity + lm-harness benchmarks (HellaSwag, ARC, WinoGrande)

Everything runs on Windows 10 with a single consumer GPU. No distributed training, no cloud.

---

## Model architecture

| Component | Choice | Why |
|-----------|--------|-----|
| Attention | Multi-Query Attention (MQA, n_kv_heads=1) | 7× less KV cache memory |
| MLP | SwiGLU | Better than ReLU/GELU empirically |
| Normalization | RMSNorm (pre-norm) | Faster than LayerNorm, more stable |
| Position encoding | RoPE (θ=100k) + NTK-aware scaling | Generalizes to longer contexts |
| Precision | BFloat16 | Stable range, native on Ampere+ |

```
Parameters:  ~145M
Layers:      16
d_model:     896
Heads:       7 (MQA: 1 KV head)
FFN dim:     2432
Vocab:       32K BPE (SentencePiece, trained from scratch)
Context:     2048 (Stage A) → 4096 (Stage B)
```

---

## Training setup

**Optimizer:** Hybrid AdamW + [Muon](knowledge.md#15-hybrid-optimizer--adamw--muon)
- AdamW handles embeddings and norms (1D parameters)
- Muon handles all linear weight matrices (2D parameters) via Newton-Schulz orthogonalization
- Muon LR (0.02) is 50× higher than AdamW (4e-4) — orthogonalization normalizes scale

**Data mix (Stage A):**
| Dataset | Weight | Content |
|---------|--------|---------|
| FineWeb-Edu | 60% | High-quality web text, education-filtered |
| StackExchange | 20% | Code and math Q&A |
| OpenWebMath | 10% | Mathematical web content |
| Cosmopedia v2 | 10% | Synthetic textbooks |

**Hardware:** RTX 5070 (12 GB) · 26k tok/s · ~21h for 2B tokens

---

## What we're learning

This repo is a hands-on study of every component in a modern LLM. See [`knowledge.md`](knowledge.md) for detailed explanations of:

- Transformer architecture and attention mechanics
- Why RMSNorm instead of LayerNorm
- How RoPE encodes position and why it generalizes
- What MQA is and how it saves VRAM
- SwiGLU and why gated activations work better
- BF16 mixed precision — what it does and what can go wrong
- Gradient checkpointing — trading compute for memory
- Sequence packing — eliminating padding waste
- Cosine LR schedule with warmup
- Muon optimizer — Newton-Schulz orthogonalization of gradients
- Gradient clipping and why it matters
- torch.compile and its constraints (CUDA graphs vs checkpointing)
- NTK-aware RoPE scaling for context extension

---

## Quickstart

See [`PIPELINE.md`](PIPELINE.md) for the full step-by-step guide. Short version:

```bash
# 1. Install deps
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt

# 2. Verify GPU
python scripts/00_verify_env.py

# 3. Train tokenizer
python scripts/20_train_tokenizer.py

# 4. Download datasets (~5 GB, ~3-4h)
bash scripts/download_all.sh

# 5. Train (~21h)
python scripts/30_pretrain.py --config configs/train_stage_a.yaml

# 6. Evaluate
python scripts/40_eval_perplexity.py --ckpt checkpoints/stage_a/latest.txt
python scripts/41_eval_lm_harness.py --ckpt checkpoints/stage_a/latest.txt
```

---

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| GPU VRAM | 8 GB | 12 GB tested; 8 GB needs `gradient_checkpointing: true` |
| CUDA | 12.x | 12.8 for Blackwell (RTX 50xx); cu121/cu124 for RTX 30xx/40xx |
| Python | 3.10+ | 3.12 tested |
| Disk | 10 GB | Tokenized shards + checkpoints |

---

## Project structure

```
configs/          model and training configs
data/             tokenizer + tokenized shards
scripts/          pipeline scripts (download, train, eval)
tinysota/
  model/          transformer, RoPE, weight init
  training/       loop, optimizer (Muon), scheduler, checkpointing
  data/           packing, shard I/O, streaming
  utils/          config loader
tests/            unit tests (30 passing)
knowledge.md      concept reference — what, why, and further reading
PIPELINE.md       full command guide from 0 to trained model
```
