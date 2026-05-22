# Show HN: ArchGene — Verify LLM architectures before you waste $50K

**URL:** https://github.com/Tejas163/ArchGene

## Title options

A: **Show HN: ArchGene – Z3-proven LLM architectures with auto-generated PyTorch code**

B: **Show HN: I built an LLM architecture verifier that generates runnable PyTorch code**

## Post body

LLM training costs are insane — a single failed run on a 7B model costs thousands in GPU time. I built a tool that mathematically proves your architecture is valid BEFORE you train.

**How it works:**
1. Answer 6 questions (use case, budget, constraints)
2. ArchGene designs a verified architecture using Z3 solver
3. Get cost estimates (training hours, inference latency)
4. Download runnable PyTorch code with RoPE, GQA, SwiGLU, RMSNorm

**What's included:**
- Z3 formal verification — proves head divisibility, hidden/layer/head ratios
- Cost estimator — GPU hours × $ for training and inference
- Architecture designer — conversational Q&A, CLI or web UI
- Code generator — produces model.py, config.json, train.py
- Model zoo — 17 reference architectures (GPT-2, Llama, Mistral, Qwen)

**Example:**
```
$ pip install archgene
$ archgene design
→ "I'm doing research with mid budget, need 8K context"
→ Designed: 4096 hidden, 24 layers, 32 heads
→ Z3: PASS, VRAM: 0.1GB, Cost: $0.02/1K tokens
$ archgene generate → runnable PyTorch model
```

**Tech:** Python, Z3, PyTorch. Web UI via Streamlit (`pip install archgene[web]`).

**Try it:** https://github.com/Tejas163/ArchGene — feedback welcome!
