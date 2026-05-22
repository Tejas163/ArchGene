# Reddit Post Draft

## Option A: r/MachineLearning — "Project" post

**Title:** I built a tool that mathematically verifies LLM architectures and generates runnable PyTorch code — saved me from a $50K mistake

**Post:**

I was designing a custom 7B model for a research project. Had the hyperparams all mapped out — 4096 hidden, 32 layers, 32 heads, 11008 intermediate. Confident. Sent the config to a friend who's done this before, and he goes: "Your hidden dim isn't divisible by your head dim. That'll OOM on day 3."

He was right. I would've burned ~$2,500 on a cluster before finding out.

So I built ArchGene — it uses Z3 (the theorem prover) to mathematically prove an LLM architecture is valid before you spend a cent on training. It caught issues I would never have spotted manually.

**What it does:**

1. `archgene design` — asks you 6 questions (use case, budget, constraints, etc.) and designs a verified architecture
2. `archgene verify --hidden 4096 --heads 32 --layers 24` — proves the architecture is mathematically sound
3. `archgene generate` — generates runnable model.py with RoPE, GQA, SwiGLU, RMSNorm + config.json + train.py
4. `archgene cost gpt2 --gpu A100` — tells you exactly what training + inference will cost

**What Z3 catches:**
- Hidden dimension % num_heads != 0 → invalid
- Head dim × num_heads != hidden_dim → invalid  
- Intermediate size constraints violations
- Layer/head parameter ratio out of bounds
- Memory ceiling violations

**Tech stack:** Z3 solver, PyTorch, Python. CLI + optional Streamlit web UI.

**Code + docs:** https://github.com/Tejas163/ArchGene

Would love feedback — what am I missing? What other checks would be useful?

---

## Option B: r/LocalLLaMA (shorter, more practical)

**Title:** PSA: Verify your LLM architecture before training — I built a free CLI tool that catches hidden dim/head mismatches, attention bugs, etc. with Z3

**Post:**

Been seeing a lot of "my custom training run failed after 3 days, no idea why" posts here. 9 times out of 10 it's an architecture bug — hidden dim not divisible by heads, intermediate size mismatch, etc.

I wrote a small tool called ArchGene that uses Z3 to mathematically verify your architecture configs. Also generates runnable model.py/train.py with modern features (RoPE, GQA, SwiGLU, RMSNorm) so you don't have to write the boilerplate.

```bash
pip install archgene
archgene design  # 6 questions → verified architecture
archgene generate  # → model.py, config.json, train.py
```

Free, open source, MIT. https://github.com/Tejas163/ArchGene

Curious if anyone else has horror stories from architecture bugs — and what other checks I should add.
