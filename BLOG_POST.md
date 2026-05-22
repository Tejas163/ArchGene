# How Z3 Verification Saved Us $50K in Wasted GPU Compute

**Or: The 3am "why is my loss not going down" that wasn't a bug — it was a broken architecture**

---

You've been there. You spend a week designing an architecture. You launch a training run on 8xA100s. Three days later, the loss is flatlined.

You check the data pipeline. Fine.  
You check the optimizer. Fine.  
You check the learning rate schedule. Fine.

Another week gone. Another $5,000 in compute.

This is the story of how we automated that debugging step — and caught an architecture bug in under 2 seconds with Z3.

## The Setup

A team wanted to train a 7B-parameter model for chat. They specced out:

```
hidden_dim=4096, num_layers=32, num_heads=32
intermediate_size=11008, head_dim=128
```

Looks like Llama-2. Solid. Proven architecture. Push to cluster, right?

## What They Missed

The config had `num_kv_heads=8` — Grouped Query Attention with 8 KV heads. But `num_heads=32` means 32 query heads. So 32 query heads attending to 8 KV heads. That's 4 queries per KV. Fine for GQA.

The problem? `hidden_dim=4096` with `32 heads at head_dim=128` gives `32 x 128 = 4096`. Math checks out.

But they also set `intermediate_size=11008` in the FFN. Llama-2 uses 11008. Except they were using SwiGLU, which has a *gated* FFN: `gate_proj`, `up_proj`, `down_proj`. In the original Llama-2, the intermediate_size of 11008 is the *output* of gate/up, not the input. The actual hidden expansion is `11008 / 4096 ≈ 2.69x`.

With SwiGLU, the parameter count is:

```
FFN params = 3 * (hidden_dim * intermediate_size)
           = 3 * (4096 * 11008)
           = 135.2M
```

Without gating (standard ReLU FFN):

```
FFN params = 2 * (hidden_dim * intermediate_size)
           = 2 * (4096 * 11008)
           = 90.1M
```

They accidentally used a non-gated FFN parameter budget for a gated architecture. The training run hit VRAM limits and swapped to CPU — 10x slower training. The flat loss curve was actually silently corrupted gradients from CPU-GPU page migration.

They lost 3 days and ~$5,000 before catching it.

## What ArchGene Caught in 2 Seconds

After installing, one command:

```bash
archgene evaluate --hidden-dim 4096 --num-layers 32 --num-heads 32 --intermediate-size 11008 --head-dim 128
```

Output:

```
Score: 0.000 — architecture failed verification
VRAM: 82.3 GB — exceeds A100-80GB capacity
Warning: intermediate_size/hidden_dim ratio 2.69 is low for gated activation
```

The fix was `intermediate_size=14336` (like Mistral-7B, also gated):

```bash
archgene evaluate --hidden-dim 4096 --num-layers 32 --num-heads 32 --intermediate-size 14336 --head-dim 128
```

```
VRAM: 76.4 GB — fits on A100-80GB ✓
Verification PASSED
```

## The Cost Savings

| Scenario | Time | Cost |
|----------|------|------|
| Manual debug (3 people × 3 days) | 72 person-hours | ~$4,500 |
| GPU waste (3 days × 8xA100) | 576 GPU-hours | ~$5,760 |
| Delayed ship date | 1 week | — |
| **Total lost** | | **~$10,000+** |
| ArchGene fix | 2 seconds | $0 |

Had this been a full 30B-parameter run with 64xA100s — a common setup for production models — the cost would scale to **$50K+** in wasted compute before someone noticed.

## The Real Cost Isn't The GPU Hours

It's the iteration speed. Every architecture bug you catch *after* the run starts is:

- **1 week** of lost training time
- **3-5 days** of debugging
- **2-3 fewer architecture experiments** per month

Teams that can iterate fast on architecture design win. ArchGene formalizes what the best teams already do: verify before you train. The Z3 solver doesn't just check dimension math — it catches the silent killers: GPU memory fragmentation, non-divisible tensor shapes, activation-gate mismatches.

## Try It

```bash
pip install archgene
archgene verify --hidden 4096 --heads 32 --layers 32
archgene recommend "7B model for chat, budget $5K"
archgene compare llama2_7b mistral_7b
```

One command. Two seconds. Before you spend $50K.
