"""Environment verification — run this first after installing deps.

Checks:
  1. PyTorch version + CUDA availability
  2. Blackwell sm_120 GPU detected
  3. BF16 matmul round-trip
  4. FlashAttention-2 import
  5. Triton import
  6. SDPA with causal mask (the actual attention path we use)
  7. torch.compile smoke test
  8. Model forward pass (tiny config)
"""
import os
import sys
import time

# Reduce allocator fragmentation — set before any CUDA calls
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def section(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

def ok(msg: str):
    print(f"  [OK] {msg}")

def warn(msg: str):
    print(f"  [WARN] {msg}")

def fail(msg: str):
    print(f"  [FAIL] {msg}")
    sys.exit(1)


section("1. PyTorch + CUDA")
try:
    import torch
    ok(f"torch {torch.__version__}")
except ImportError:
    fail("torch not installed")

if not torch.cuda.is_available():
    fail("CUDA not available — check driver and torch install")
ok(f"CUDA available: {torch.cuda.get_device_name(0)}")

cap = torch.cuda.get_device_capability()
ok(f"Compute capability: sm_{cap[0]}{cap[1]}")
if cap[0] < 8:
    warn("BF16 requires sm_80+. This GPU may not support bfloat16 natively.")
if cap == (12, 0):
    ok("Blackwell sm_120 confirmed")


section("2. BF16 matmul")
try:
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    assert c.shape == (512, 512)
    ok("BF16 matmul on GPU works")
except Exception as e:
    fail(f"BF16 matmul failed: {e}")


section("3. FlashAttention-2")
try:
    import flash_attn
    ok(f"flash_attn {flash_attn.__version__}")
except ImportError:
    warn("flash_attn not installed — SDPA will fall back to math backend. Performance will be lower.")


section("4. Triton")
try:
    import triton
    ok(f"triton {triton.__version__}")
except ImportError:
    warn("triton not installed — torch.compile will run without Triton backend (slower or disabled).")


section("5. SDPA causal attention")
try:
    import torch.nn.functional as F
    B, H, T, D = 2, 8, 128, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.shape == (B, H, T, D)
    ok("SDPA causal attention forward pass OK")
except Exception as e:
    fail(f"SDPA failed: {e}")


section("6. torch.compile")
try:
    @torch.compile(mode="default")
    def add(x, y):
        return x + y
    x = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    _ = add(x, x)
    ok("torch.compile smoke test passed")
except Exception as e:
    warn(f"torch.compile failed: {e}\n  Set compile=false in train configs.")


section("7. Model forward pass (tiny config)")
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from tinysota.model import LlamaModel, LlamaConfig
    cfg = LlamaConfig(
        vocab_size=32000, n_layers=2, d_model=256, n_heads=2,
        n_kv_heads=1, head_dim=128, ffn_dim=512, max_seq_len=128,
    )
    m = LlamaModel(cfg).to("cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 32000, (2, 64), device="cuda")
    logits = m(ids)
    assert logits.shape == (2, 64, 32000)
    params = m.num_params() / 1e6
    ok(f"Model forward pass OK — {params:.1f}M params")
    del m
    torch.cuda.empty_cache()
except Exception as e:
    fail(f"Model forward pass failed: {e}")


section("8. Throughput — gradient checkpointing ON, batch=8, seq=2048")
try:
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    from tinysota.model import LlamaModel, LlamaConfig
    lm_cfg = LlamaConfig(
        vocab_size=32000, n_layers=16, d_model=896, n_heads=7,
        n_kv_heads=1, head_dim=128, ffn_dim=2432, max_seq_len=2048,
    )
    # gradient_checkpointing=True: recompute each layer during backward instead of storing
    # activations. Frees ~8 GB at batch=8 seq=2048, costs ~33% throughput.
    m = LlamaModel(lm_cfg, gradient_checkpointing=True).to("cuda", dtype=torch.bfloat16)

    # mode="default": compatible with gradient checkpointing.
    # reduce-overhead uses CUDA graphs which conflict with checkpointing.
    print("  Compiling with default mode...")
    m = torch.compile(m, mode="default")

    BATCH, SEQ = 8, 2048
    ids = torch.randint(0, 32000, (BATCH, SEQ), device="cuda")

    # Simulate optimizer memory: allocate FP32 tensors matching param count
    # (AdamW keeps m and v in FP32 — 2 × params × 4 bytes)
    n_params = lm_cfg.n_layers * (4 * lm_cfg.d_model**2 + 3 * lm_cfg.d_model * lm_cfg.ffn_dim)
    fake_optim = torch.empty(n_params * 2, dtype=torch.float32, device="cuda")

    print("  Warmup (triggering compile, ~30s)...")
    m.train()
    for i in range(3):
        logits = m(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids.view(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)
        if i == 0:
            print("  First warmup step done...")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    STEPS = 20
    for _ in range(STEPS):
        logits = m(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids.view(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_per_sec = (STEPS * BATCH * SEQ) / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tokens_per_day = tok_per_sec * 86400

    ok(f"Throughput: {tok_per_sec:.0f} tok/s  (batch={BATCH}, seq={SEQ}, checkpointing=ON)")
    ok(f"Peak VRAM (model+grads+fake optim+activations): {peak_gb:.2f} GB  (of 12 GB)")
    ok(f"Projected: {tokens_per_day/1e9:.2f}B tokens/day  →  {tokens_per_day*2/1e9:.2f}B in 2 days")

    # Recommend total_tokens based on measured throughput
    recommended = int(tokens_per_day * 2 * 0.85)  # 85% efficiency over 2 days
    ok(f"Recommended total_tokens for Stage A: {recommended/1e9:.1f}B")
    ok(f"  → update configs/train_stage_a.yaml: total_tokens: {recommended:,}")

    if peak_gb > 11:
        warn("VRAM very tight. If training OOMs, reduce micro_batch to 6 in train_stage_a.yaml")
    else:
        ok(f"VRAM safe: {12 - peak_gb:.1f} GB headroom")

    del m, fake_optim
    torch.cuda.empty_cache()
except Exception as e:
    warn(f"Throughput estimate failed: {e}")


section("Summary")
print("  If all [OK] above (or only [WARN] for optional deps), you're ready to proceed.")
print("  Next: run scripts/20_train_tokenizer.py then scripts/10-13_download_*.py")
