# Knowledge Document — Tiny SOTA LLM

This document explains every concept, architecture decision, optimization, and numerical value used in this project. Written for someone with a general programming background but no prior knowledge of language models or deep learning.

---

## Table of Contents

1. [What are we building?](#1-what-are-we-building)
2. [The Transformer](#2-the-transformer)
3. [Decoder-only architecture (GPT-style)](#3-decoder-only-architecture-gpt-style)
4. [Tokenization](#4-tokenization)
5. [Embeddings](#5-embeddings)
6. [Attention](#6-attention)
7. [RoPE — Rotary Positional Embeddings](#7-rope--rotary-positional-embeddings)
8. [RMSNorm](#8-rmsnorm)
9. [SwiGLU MLP](#9-swiglu-mlp)
10. [Multi-Query Attention (MQA)](#10-multi-query-attention-mqa)
11. [FlashAttention](#11-flashattention)
12. [Weight Initialization](#12-weight-initialization)
13. [Training Objective — Next-Token Prediction](#13-training-objective--next-token-prediction)
14. [Loss, Perplexity, and Cross-Entropy](#14-loss-perplexity-and-cross-entropy)
15. [AdamW Optimizer](#15-adamw-optimizer)
16. [Learning Rate Schedule](#16-learning-rate-schedule)
17. [Gradient Clipping](#17-gradient-clipping)
18. [Mixed Precision (BF16)](#18-mixed-precision-bf16)
19. [Gradient Accumulation](#19-gradient-accumulation)
20. [Sequence Packing](#20-sequence-packing)
21. [Scaling Laws & Why Our Model is This Size](#21-scaling-laws--why-our-model-is-this-size)
22. [Dataset Selection](#22-dataset-selection)
23. [Context Curriculum](#23-context-curriculum)
24. [NTK-Aware RoPE Scaling](#24-ntk-aware-rope-scaling)
25. [torch.compile](#25-torchcompile)
26. [Numerical Values Reference](#26-numerical-values-reference)
27. [Glossary](#27-glossary)

---

## 1. What are we building?

We are building a **language model** — a program that, given a sequence of words, predicts what word comes next. Doing this well enough, for long enough, on enough text, produces a model that can write, answer questions, translate, summarize, and reason.

The specific kind we are building is called a **Large Language Model (LLM)**, although ours is small by industry standards (~145 million parameters). It is "small" relative to GPT-4 (estimated trillions of parameters) but still a fully functional modern language model with the same architectural stack used in production systems.

**Why build from scratch?** The goal is to understand every component — not just use a pre-built one. Every line of code corresponds to a concept explained in this document.

---

## 2. The Transformer

The Transformer is the neural network architecture that underlies virtually every modern language model. It was introduced in 2017 and replaced older sequence models like RNNs and LSTMs.

The key idea is **attention**: instead of processing words one by one left to right (like an RNN), the Transformer looks at all positions simultaneously and lets each position "attend" to every other position to gather context. This parallelism makes Transformers much faster to train on modern GPU hardware.

A Transformer is built from **layers** stacked on top of each other. Each layer contains:
1. An **attention block** — gathers context from other positions.
2. An **MLP block** — processes each position independently using a small neural network.

Both blocks use a **residual connection**: the input is added back to the output (`x = x + block(x)`). This makes it easy for gradients to flow during training and prevents the model from "forgetting" the original input.

**Why we use it here**: every competitive language model uses Transformers. The architecture scales predictably and is well-understood.

**Learn more**:
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the original Transformer paper
- [The Illustrated Transformer (blog)](https://jalammar.github.io/illustrated-transformer/) — visual explainer

---

## 3. Decoder-only architecture (GPT-style)

The original Transformer had two parts: an **encoder** (reads the full input) and a **decoder** (generates the output). For language generation, researchers found you only need the decoder part. This is called a **decoder-only** architecture, also known as GPT-style.

The key property of the decoder: each token can only see tokens that came **before** it (not future tokens). This is called a **causal** or **autoregressive** constraint. It is enforced by a **causal mask** in the attention computation that blocks future positions. This constraint is what allows the model to generate text token by token at inference time.

Our model is decoder-only. It takes a sequence of token IDs as input and produces a probability distribution over the next token for every position in the sequence simultaneously (during training).

**Why we use it here**: decoder-only is the standard for text generation. Encoder-only models (BERT) are better for classification/understanding but cannot generate text. Encoder-decoder models (T5) are more complex and offer no advantage at our scale.

**Learn more**:
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)

---

## 4. Tokenization

Before a model can process text, text must be converted into numbers. **Tokenization** is this conversion. A **tokenizer** splits text into subword units called **tokens** and maps each token to an integer ID.

We use **Byte Pair Encoding (BPE)**, one of the most common tokenization algorithms. BPE starts with individual characters, then iteratively merges the most frequent adjacent pair of tokens into a new token. After enough merges, common words become single tokens, rare words are split into subword pieces, and unknown characters are covered by byte-level fallbacks.

Example: `"unhappiness"` might tokenize as `["un", "happiness"]` (2 tokens), while `"cryptocurrency"` might become `["crypto", "currency"]` or even more pieces.

Our tokenizer has **32,000 tokens** in its vocabulary — meaning the model's output layer predicts one of 32,000 possible next tokens at each step.

We train our own tokenizer on a sample of our training data using **SentencePiece**, a library that handles BPE training and has strong multilingual support. Key option: `byte_fallback=true` ensures any byte sequence can be represented, so the model never encounters unknown tokens.

**Why we use it here**: a custom 32K tokenizer keeps the embedding table small (~145M params at d_model=896) and is tailored to the vocabulary of our training data mix.

**Learn more**:
- [SentencePiece: A simple and language independent subword tokenizer (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226)
- [Neural Machine Translation of Rare Words with Subword Units / BPE (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909)

---

## 5. Embeddings

Once text is tokenized into integer IDs, those integers must be converted into vectors (lists of numbers) that the neural network can process. This is done by an **embedding table**: a matrix of shape `(vocab_size, d_model)` where each row is a learned vector for one token.

Looking up an embedding is just selecting a row from this matrix. The embedding is learned during training — similar tokens end up with similar vectors.

`d_model` is the key dimension of our model. In our case `d_model = 896`. Every token is represented as a 896-dimensional vector throughout the network.

**Tied embeddings**: we reuse the same embedding matrix for the output layer (called **weight tying**). The output layer converts the model's final d_model-dimensional representation back into vocabulary probabilities. Tying input and output embeddings was shown to improve perplexity and halves the parameter cost of the embedding table.

**Why we use it here**: standard practice. Weight tying is used in GPT-2 and most Llama-family models.

**Learn more**:
- [Using the Output Embedding to Improve Language Models (Press & Wolf, 2017)](https://arxiv.org/abs/1608.05859) — weight tying paper

---

## 6. Attention

Attention is the mechanism that allows each token to gather information from other tokens in the sequence.

For each token, we compute three vectors:
- **Query (Q)**: "what am I looking for?"
- **Key (K)**: "what do I contain?"
- **Value (V)**: "what information do I pass along?"

The attention score between token i and token j is computed as the dot product of Q_i and K_j, scaled by `1/sqrt(head_dim)`. These scores are passed through softmax to get weights that sum to 1. The output is a weighted sum of all Value vectors.

**Multi-head attention** runs this process in parallel with different Q/K/V projections (called "heads"), each learning to attend to different aspects of the sequence. The outputs of all heads are concatenated.

In our model: 7 query heads, each with `head_dim=128`, so Q has dimension `7 × 128 = 896 = d_model`. K and V have only 1 head each (see MQA, §10).

**Why we use it here**: attention is the core of every Transformer. There is no replacement.

**Learn more**:
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

---

## 7. RoPE — Rotary Positional Embeddings

Transformers process all positions simultaneously, so they need a way to encode position (which token is first, second, etc.). **RoPE** is the current standard method for this.

The idea: instead of adding a fixed positional vector to each token, RoPE **rotates** the Query and Key vectors by an angle that depends on their position. When you compute the dot product Q·K, the angle between them depends only on their **relative distance**, not their absolute positions. This is a key advantage: the model naturally generalizes to positions it hasn't seen in exactly that context.

Mathematically: each dimension pair in Q and K is treated as a 2D vector and rotated by `position × θ^(2i/d)` where `θ` is a base frequency (we use 100,000) and `i` is the dimension index.

**Why we use it here**: RoPE is used in LLaMA, Mistral, Falcon, Qwen, and almost every modern open LLM since 2023. It handles context extension much better than the original learned absolute positional embeddings.

**Learn more**:
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)

---

## 8. RMSNorm

**Normalization** is applied inside each Transformer layer to keep activations at a reasonable scale. Without it, values can explode or vanish, making training unstable.

The original Transformer used **LayerNorm**, which computes mean and variance then normalizes. **RMSNorm** is simpler: it only divides by the root-mean-square of the activations (no mean subtraction). This removes one operation and was shown to be just as effective.

```
RMSNorm(x) = x / RMS(x) * weight
RMS(x) = sqrt(mean(x²))
```

We apply RMSNorm **before** each block (pre-norm), not after. Pre-norm makes training more stable at large scale — this is the LLaMA convention.

We compute RMSNorm in **FP32** even when training in BF16, then cast back. This prevents numerical instability at the normalization step.

**Why we use it here**: RMSNorm + pre-norm is the standard in every Llama-family model.

**Learn more**:
- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)

---

## 9. SwiGLU MLP

Each Transformer layer contains an **MLP** (multi-layer perceptron) block that transforms each token's representation independently.

The original Transformer used a simple two-layer MLP with ReLU activation. Modern LLMs use **SwiGLU**, a gated variant:

```
SwiGLU(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

There are **three** weight matrices instead of two. The `gate_proj` output, passed through SiLU (a smooth version of ReLU), acts as a multiplicative gate on the `up_proj` output. This gating mechanism gives the model more expressive control over which information to pass through.

Our MLP dimension: `ffn_dim = 2432 ≈ 2.75 × d_model`. SwiGLU models use a slightly smaller intermediate dimension than standard MLP to keep parameter count comparable despite the extra projection.

**Why we use it here**: SwiGLU is used in LLaMA, PaLM, Mistral, and most SOTA open models. It consistently improves perplexity over ReLU/GELU at no compute cost.

**Learn more**:
- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)

---

## 10. Multi-Query Attention (MQA)

Standard multi-head attention uses separate K and V matrices for each head. **Multi-Query Attention (MQA)** uses a single shared K and V across all heads. Our model has 7 query heads but only 1 K/V head.

This reduces:
- Memory used by K/V projections (from `n_heads × head_dim` to `1 × head_dim`)
- KV cache size at inference (7× smaller)
- Memory bandwidth during training

The quality loss compared to full multi-head attention is small at our model size — especially with a high-quality training mix.

**Grouped-Query Attention (GQA)** is the generalized version: 2–4 K/V heads instead of 1. The paper uses GQA; we use MQA (the extreme case, 1 KV head) because our model is smaller and the memory savings are more valuable.

**Why we use it here**: MQA is standard at small model sizes. Llama-3-8B uses GQA; models smaller than ~1B often use MQA.

**Learn more**:
- [Fast Transformer Decoding: One Write-Head is All You Need / MQA (Shazeer, 2019)](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)

---

## 11. FlashAttention

Standard attention computes the full `(seq_len × seq_len)` attention matrix and stores it in GPU memory (HBM). For seq_len=2048 and BF16, that's `2048² × 2 bytes = 8MB per head per batch element` — it adds up fast.

**FlashAttention** is an algorithm that computes attention in tiles without ever materializing the full attention matrix. It keeps intermediate results in the GPU's fast on-chip SRAM (much smaller but much faster), writes the final result back to HBM. This is called **IO-aware** computation.

Result: same mathematical output as standard attention, but:
- 2–4× less memory
- 2–4× faster (fewer slow HBM reads/writes)

**FlashAttention-2** improves parallelism and is the version we use. FlashAttention-3 is optimized for Hopper (H100) and not useful on our Blackwell consumer GPU.

We access FlashAttention-2 through **PyTorch SDPA** (`F.scaled_dot_product_attention`), which automatically dispatches to FA2 kernels when the input is BF16 and contiguous on CUDA. We don't call FA2 directly.

**Why we use it here**: mandatory for training at reasonable speed. On the RTX 5070, this is the difference between fitting and not fitting in 12GB VRAM at seq_len=2048 with batch_size=8.

**Learn more**:
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)

---

## 12. Weight Initialization

Neural networks start training from random weights. The distribution of those initial weights matters — if they are too large, activations explode; too small, gradients vanish.

We use **truncated normal** initialization with `std=0.02` for all linear and embedding layers. This is the same convention as GPT-2.

For **output projections** (the `o_proj` in attention and `down_proj` in MLP), we scale the initial weights by `1 / sqrt(2 × n_layers)`. This is the **GPT-NeoX** convention. The intuition: each layer contributes to the residual stream, and with 16 layers the variance would accumulate 16× without scaling. Dividing by `sqrt(2 * 16) = ~5.7` keeps the residual stream variance at ~1 at initialization.

**Why we use it here**: standard for Llama-style models. Bad initialization can cost 10–20% of training time just recovering.

**Learn more**:
- [Language Models are Unsupervised Multitask Learners / GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-NeoX-20B (Black et al., 2022)](https://arxiv.org/abs/2204.06745)

---

## 13. Training Objective — Next-Token Prediction

We train the model by showing it billions of tokens from real text and asking it to predict the next token at every position. This is called **next-token prediction** or **causal language modeling**.

At each position `t`, the model sees tokens `[0, 1, ..., t-1]` and outputs a probability distribution over all 32,000 vocabulary entries. The correct answer is the actual token at position `t`. The model is penalized (via loss) for assigning low probability to the correct token.

No labels need to be manually created — the text itself IS the label. This is **self-supervised learning**: the supervision signal comes from the data itself.

**Teacher forcing**: during training, we always feed the model the true previous tokens (not its own predictions). This makes training stable and efficient. At inference, the model feeds its own predictions back in.

**Why we use it here**: the only training objective for pretraining an LLM. Everything else (RLHF, instruction tuning) comes after pretraining.

**Learn more**:
- [Language Models are Unsupervised Multitask Learners / GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## 14. Loss, Perplexity, and Cross-Entropy

**Cross-entropy loss** measures how wrong the model's prediction is. If the model assigns probability `p` to the correct token, the loss is `-log(p)`. Perfect prediction (p=1) gives loss=0. Predicting probability 0 to the correct token gives infinite loss.

During training we average this loss over all tokens in the batch.

**Perplexity** is `exp(loss)`. It has a natural interpretation: a perplexity of 10 means the model is "as confused as if it had to choose uniformly among 10 options." Lower is better.

A random model over 32,000 tokens has perplexity ≈ 32,000. A good pretrained model reaches perplexity of 10–20 on held-out web text.

**Why we track it here**: loss is what we optimize; perplexity is more interpretable for reporting. We log both during training.

---

## 15. AdamW Optimizer

An **optimizer** is the algorithm that updates model weights to reduce the loss. We use **AdamW**.

**Adam** (Adaptive Moment Estimation) maintains a running average of past gradients (momentum, `m`) and past squared gradients (variance, `v`). The update for each weight is:
```
m = β1 * m + (1 - β1) * grad
v = β2 * v + (1 - β2) * grad²
weight -= lr * m / (sqrt(v) + eps)
```

This adapts the learning rate per-parameter. Parameters with consistently large gradients get a smaller effective learning rate; sparse parameters get a larger one.

**AdamW** adds **weight decay** correctly. Naive L2 regularization added to the loss interacts with Adam's scaling in a way that weakens its effect. AdamW applies weight decay directly to the weights, bypassing the gradient scaling. This is the correct way to regularize Adam.

Our values: `β1=0.9, β2=0.95, eps=1e-8, weight_decay=0.1`. The `β2=0.95` (instead of the original 0.999) makes the variance estimate adapt faster — standard for LLM training.

**Why we use it here**: AdamW is the optimizer for virtually every LLM. The Llama series, GPT-4, Mistral — all use AdamW or a close variant.

**Learn more**:
- [Decoupled Weight Decay Regularization / AdamW (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)

---

## 16. Learning Rate Schedule

The **learning rate (LR)** controls how large each optimizer step is. Using a fixed LR is suboptimal — we use a schedule:

**1. Linear warmup** (first 2% of training, ~180M tokens): LR grows linearly from 0 to the peak value. This prevents large early updates from destabilizing random initialized weights.

**2. Cosine decay** (remaining 98%): LR follows a cosine curve from peak down to `min_lr_ratio × peak`. Cosine is smoother than linear decay and consistently outperforms it empirically.

**3. Minimum LR ratio = 0.1**: we decay to 10% of peak (4e-4 → 4e-5), not to zero. Decaying all the way to zero can hurt the final model.

**Stage B** uses **linear decay** instead of cosine, since it's a short annealing phase — we want to reach near-zero LR by the end.

Our peak LR: `4e-4`. This is slightly higher than the Llama-3 recipe (3e-4) because our model is smaller — smaller models tolerate higher learning rates.

**Why we use it here**: warmup + cosine is universal for LLM pretraining. Without warmup, early training diverges. Without decay, the model doesn't converge tightly.

**Learn more**:
- [SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2016)](https://arxiv.org/abs/1608.03983)

---

## 17. Gradient Clipping

During training, occasionally the gradient of the loss with respect to weights can become very large (a **gradient spike**). If left unchecked, this causes the optimizer to take an enormous step that destroys previously learned weights.

**Gradient clipping** caps the global norm of all gradients to a maximum value (we use `1.0`). If the total gradient norm exceeds 1.0, every gradient is scaled down proportionally.

```
norm = sqrt(sum of all grad²)
if norm > 1.0:
    grad *= 1.0 / norm
```

**Why we use it here**: standard for LLM training. Spikes happen, especially early in training with difficult data batches. Clipping makes training robust.

---

## 18. Mixed Precision (BF16)

Storing and computing with 32-bit floats (FP32) uses 4 bytes per number. **BF16** (Brain Float 16) uses 2 bytes — half the memory.

BF16 has the same exponent range as FP32 (so no overflow/underflow issues) but less mantissa precision. For neural network weights and activations, this precision reduction is empirically fine.

**Why BF16 over FP16**: FP16 has a smaller exponent range and needs a `GradScaler` to prevent underflow. BF16 doesn't need any special handling — it just works. All Blackwell GPUs support BF16 natively.

In practice: model weights and activations use BF16; RMSNorm computations temporarily upcast to FP32 for numerical stability; optimizer states (AdamW m and v) stay in FP32 (this is called **mixed precision** — the "mixed" refers to using both FP32 optimizer states and BF16 for everything else).

**Why we use it here**: BF16 halves VRAM use and speeds up computation. On Blackwell GPUs, BF16 tensor core throughput is 2× that of FP32.

**Learn more**:
- [Mixed Precision Training (Micikevicius et al., 2017)](https://arxiv.org/abs/1710.03740)

---

## 19. Gradient Accumulation

The **batch size** affects training stability and final model quality. Larger batches mean more stable gradient estimates but also mean more VRAM used at once.

**Gradient accumulation** decouples batch size from VRAM: we run multiple small **micro-batches** through the model, accumulate (sum) their gradients, and only call `optimizer.step()` once after `grad_accum_steps` micro-batches.

Our config: `micro_batch=8, grad_accum=16, seq_len=2048` → effective batch = `8 × 16 × 2048 = 262,144 tokens` per update.

This is equivalent to running a single batch of 262K tokens, but we only keep 8 × 2048 = 16K tokens in VRAM at once.

The 262K token effective batch is in line with Llama-3 pretraining (4M tokens/step at much larger scale), scaled appropriately for our model size.

**Why we use it here**: the 12GB VRAM constraint makes it impossible to fit a large batch. Gradient accumulation gets us to a reasonable effective batch size without OOM.

---

## 20. Sequence Packing

Each training document (a web page, a code file, a math problem) has a different length. The naïve approach pads shorter documents to the maximum length — but padding tokens contribute nothing to learning and waste compute.

**Sequence packing** concatenates multiple documents end-to-end (separated by an EOS token) to fill exactly `block_size` tokens. No padding is ever needed.

```
[doc1_tok1, ..., doc1_tokN, <EOS>, doc2_tok1, ..., doc2_tokM, <EOS>, doc3_tok1, ...]
```

The blocks are then cut to fixed length. A document can span multiple blocks or multiple documents can fit in one block.

This gives ~10–30% throughput improvement on typical web text (which has variable-length documents) because the GPU never processes padding tokens.

**Why we use it here**: mandatory for efficient pretraining. Mentioned explicitly in the research paper as "a major hidden performance multiplier."

---

## 21. Scaling Laws & Why Our Model is This Size

**Scaling laws** describe how model quality (loss) improves as you scale up parameters and training tokens. The landmark paper is **Chinchilla (Hoffmann et al., 2022)**, which showed:

> For a given compute budget, the optimal model has roughly **20 tokens of training data per parameter**.

For 145M parameters → optimal training: ~3B tokens.

**Modern over-training**: Chinchilla was optimizing for compute efficiency during training. For small models intended to be deployed and run many times, it's better to train longer (more tokens) so the model is better at inference. SmolLM-135M was trained on 600B tokens (~4500× Chinchilla). We target 10B tokens (~70× Chinchilla) — a pragmatic middle ground given our 1–2 day training budget.

**Why 145M params**: it fits in 12GB VRAM with a large enough batch for stable training, trains in ~2 days, and is large enough to demonstrate real language capabilities. The paper's range is 50M–300M; we sit near the middle.

**Learn more**:
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)

---

## 22. Dataset Selection

**Data quality dominates capability at small scale.** Training a 145M model on 10B tokens of carefully filtered data outperforms the same model trained on 10× more noisy data.

Our four-source mix (the **SmolLM-Corpus recipe**):

**FineWeb-Edu (60%)**: web pages filtered for educational quality using a LLaMA-based classifier. The classifier scored 15B web pages; only those scoring ≥3/5 are included. Outperforms raw Common Crawl by +4 MMLU points on the same token budget.

**Python-Edu (20%)**: Python code from The Stack v2, filtered using the same educational scoring (≥4/5). Code training improves reasoning, structured output, and logic — disproportionate benefit relative to its size.

**OpenWebMath (10%)**: mathematical web text extracted from Common Crawl, with LaTeX preserved. 14.7B tokens total. Math training improves multi-step reasoning.

**Cosmopedia v2 (10%)**: synthetic textbooks and stories generated by Mixtral-8x7B, covering 34,000 topics. Synthetic data is dense with concepts per token — more "teaching signal" per byte than raw web text.

**Why this mix**: this is exactly what HuggingFace used for SmolLM-135M and SmolLM-360M. It's the best publicly documented recipe for models in our size range.

**Learn more**:
- [FineWeb: Decanting the Web for the Finest Text Data at Scale (Penedo et al., 2024)](https://arxiv.org/abs/2406.17557)
- [SmolLM: Blazingly Fast and Remarkably Powerful (HuggingFace, 2024)](https://huggingface.co/blog/smollm)
- [Textbooks Are All You Need / Phi-1 (Gunasekar et al., 2023)](https://arxiv.org/abs/2306.11644) — the paper that introduced synthetic textbook training

---

## 23. Context Curriculum

A context window of 32K tokens requires much more memory and compute than 2K. Training at long context from the start is wasteful — the model doesn't yet know language, so long-range dependencies are not meaningful to learn yet.

**Context curriculum**: we start training at 2K context and only extend once the model has a solid base:
- **Stage A** (85% of budget): 2048 tokens context
- **Stage B** (15% of budget): extend to 4096 context + anneal

Each time we double context, memory per step roughly quadruples (attention is O(seq²)). We compensate by halving the micro-batch.

**Why we stop at 4K**: our 12GB VRAM budget. 8K at our model size would require micro_batch=1 with gradient checkpointing, slowing training below our throughput target. We explicitly skip 16K/32K for this build.

---

## 24. NTK-Aware RoPE Scaling

When we extend context from 2K to 4K in Stage B, the model has never seen position IDs beyond 2048 during Stage A. Naively, those positions are out-of-distribution.

**NTK-aware interpolation** solves this by scaling the RoPE base frequency `θ`. The observation (from bloc97, 2023): RoPE encodes position information across different frequencies. Low-frequency components handle long-range relationships; high-frequency components handle local ones. Simply rescaling all frequencies equally (as in the earlier "Position Interpolation" method) blurs the high-frequency components.

NTK-aware scaling instead scales `θ` itself:
```
new_theta = base_theta × scale_factor^(d / (d - 2))
```

For `scale_factor=2` (doubling context from 2K to 4K): `new_theta = 100000 × 2^(128/126) ≈ 101111`. This preserves local high-frequency patterns while stretching the overall positional range.

**Why we use it here**: cheap (one hyperparameter change), empirically works well for 2× context extension without any additional training budget.

**Learn more**:
- [NTK-Aware Scaled RoPE (bloc97, 2023)](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) — blog/reddit post where this was first described
- [YaRN: Efficient Context Window Extension of Large Language Models (Peng et al., 2023)](https://arxiv.org/abs/2309.00071) — more principled version of the same idea

---

## 25. torch.compile

`torch.compile` is a compiler for PyTorch models introduced in PyTorch 2.0. It traces the model's computation graph, fuses operations together, and emits optimized GPU kernels via the **Triton** compiler (which we install as `triton-windows` on Windows).

Key optimizations it performs:
- **Kernel fusion**: instead of separate GPU kernels for `x + bias`, `layernorm`, and `relu`, it compiles them into one kernel. Reduces memory reads/writes.
- **Memory planning**: allocates intermediate buffers more efficiently.
- **Loop unrolling and other compiler tricks**.

Typical speedup: 20–40% for transformer models. At our throughput target of ≥12k tok/s, this is the difference between finishing in 2 days or 3.

**Windows caveat**: `triton-windows` is a fork of Triton that adds Windows support. We use `triton-windows>=3.2.0`. If it causes issues, we set `compile=false` in the config and accept the throughput penalty.

**Why we use it here**: crucial for hitting the throughput target on a 1–2 day training budget.

**Learn more**:
- [torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [Inductor: The new PyTorch compiler backend (PyTorch Blog, 2022)](https://pytorch.org/blog/pytorch-2.0-release/)

---

## 26. Numerical Values Reference

Every number in the config has a reason. This table is the canonical explanation.

| Value | Where | Why |
|---|---|---|
| `d_model = 896` | model | 7 × 128 (head_dim). Chosen to give ~145M params when combined with 16 layers. Rounded to multiples of 128 for tensor core alignment. |
| `n_layers = 16` | model | Primary lever to hit the 145M param target. Fewer layers → smaller model than needed; more layers → too slow per step. |
| `n_heads = 7` | model | d_model / head_dim = 896 / 128. Odd number is fine. |
| `n_kv_heads = 1` | model | MQA. Saves KV memory. Standard for small models. |
| `head_dim = 128` | model | Matches LLaMA convention. Powers of 2 are efficient on tensor cores. |
| `ffn_dim = 2432` | model | ~2.75 × d_model, rounded to 256. SwiGLU standard ratio. |
| `rope_theta = 100000` | model | Higher theta = larger "range" of distinguishable positions. 10000 (original) is too small for 4K ctx; Llama-3 uses 500000 for 128K. We target 4K so 100000 is sufficient. |
| `vocab_size = 32000` | tokenizer | SentencePiece BPE, per paper spec. Small enough to keep embedding table at 33M params. |
| `lr = 4e-4` | training | Standard small-LLM range is 3e-4 to 5e-4. Smaller models tolerate higher LR. Llama-3-8B uses 3e-4. |
| `β1 = 0.9` | optimizer | Standard Adam momentum. |
| `β2 = 0.95` | optimizer | Faster variance adaptation than default 0.999. Used by GPT-3, Llama, Mistral. |
| `weight_decay = 0.1` | optimizer | Strong regularization. Standard for LLM pretraining. |
| `eps = 1e-8` | optimizer | Prevents division by zero. Standard. |
| `warmup_pct = 0.02` | schedule | 2% of 34,332 steps = ~687 warmup steps ≈ 180M tokens. Short warmup is standard for over-trained small models. |
| `min_lr_ratio = 0.1` | schedule | LR decays to 10% of peak (4e-5). Decaying all the way to 0 consistently hurts final model quality. |
| `grad_clip = 1.0` | training | Standard for LLM training. Prevents catastrophic gradient spikes. |
| `micro_batch = 8` | training | Largest value that fits in 12GB VRAM at seq=2048 without gradient checkpointing. |
| `grad_accum = 16` | training | 8 × 16 × 2048 = 262K effective tokens/step. Matches reasonable batch size for our model scale. |
| `block_size = 2048` | data | Stage A context length. Equivalent to ~1500 English words per sample. |
| `total_tokens_A = 9B` | training | ~62× Chinchilla. Within 1–2 day budget. Enough for real language capabilities. |

---

## 27. Glossary

**Activation**: the output of a neuron or layer.

**Autoregressive**: generating output one token at a time, feeding each output back as input for the next step.

**Batch size**: how many training examples are processed together in one step.

**BF16**: Brain Float 16, a 2-byte floating point format with the same exponent range as FP32.

**Causal mask**: a matrix that prevents attention from looking at future positions.

**Context window**: the maximum number of tokens a model can see at once.

**Embedding**: a dense vector representation of a token.

**Forward pass**: running input through the model to get output (no gradient computation).

**Backward pass**: computing gradients of the loss with respect to all model weights (via backpropagation).

**Gradient**: the direction and magnitude in which a weight should change to reduce the loss.

**GFLOPS/TFLOPS**: billions/trillions of floating point operations per second. Measure of GPU compute throughput.

**HBM**: High Bandwidth Memory. The main VRAM on a GPU. Slower than SRAM but much larger.

**Head**: one parallel attention computation unit. Multi-head attention runs many of these in parallel.

**Inference**: running a trained model to generate output (no training, no gradients).

**KV cache**: cached Key and Value tensors from previous positions, used during autoregressive inference to avoid recomputing.

**Layer**: one attention + MLP block with residual connections.

**Loss**: a scalar number measuring how wrong the model's predictions are. Lower is better.

**Memmap**: memory-mapped file — a file on disk accessed as if it were in RAM, enabling datasets larger than RAM.

**Perplexity**: exp(loss). Intuitively, "how many choices the model is confused between." Lower is better.

**Residual connection**: adding a layer's input to its output: `x = x + layer(x)`.

**Shard**: one chunk of a large dataset file, stored as a separate file.

**SRAM**: Static RAM. The fast on-chip memory inside a GPU, used by FlashAttention for intermediate computations.

**Token**: one unit of text after tokenization (a word, subword, or character).

**VRAM**: Video RAM. The GPU's memory. On the RTX 5070: 12GB.

**Weight**: a learnable parameter in the model (a number that changes during training).
