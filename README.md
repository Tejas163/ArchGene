# ArchGene

**Verify your LLM architecture before you waste $50K on compute.**

Training an LLM costs $10K–$100K+. The #1 reason training fails? Architecture misconfiguration — hidden dimension misalignment, attention bugs, incompatible layer configurations.

ArchGene catches these issues BEFORE you spend on GPU time.

## The Problem

```
You spend $50K on GPU cluster
↓
Start training
↓
Day 3: OOM errors, NaN outputs, training crashes
↓
Why? Hidden dimension not divisible by attention heads
↓
$50K wasted
```

ArchGene prevents this.

## What It Does

| Feature | What It Tells You |
|---------|------------------|
| **Z3 Verification** | "Your architecture is mathematically valid" or "Here's what's broken" |
| **Cost Estimation** | "This will cost $12K to train on 8x A100s" |
| **Benchmark Projections** | "Expected MMLU score: ~42%" |
| **Model Zoo** | Compare against GPT-2, Llama-2, Mistral, etc. |

## Quick Start

```bash
# Install
pip install archgene

# Verify your architecture BEFORE training
python -m archgene verify --hidden 4096 --heads 32 --layers 24

# Get cost estimate
python -m archgene cost --model gpt2 --gpu A100

# Check against known architectures
python -m archgene zoo-evaluate llama2_7b
```

## Why This Matters

- **Don't waste compute**: Catch bugs before GPU costs begin
- **Know your bill**: Estimate training cost before you start
- **Validate fast**: Z3 proves correctness mathematically

## Use Cases

1. **Building a custom LLM?** Verify architecture before training
2. **Fine-tuning an existing model?** Check your config is valid
3. **Comparing architectures?** Benchmark against model zoo

## CLI Examples

```bash
# Verify custom architecture
archgene verify --hidden 4096 --heads 32 --layers 24

# Cost estimation
archgene cost gpt2 --gpu H100 --batch-size 16

# List pre-trained architectures
archgene zoo-list

# Benchmark estimate
archgene benchmark llama2_7b
```

## Architecture Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| vocab_dim | Vocabulary size | 32000, 50257 |
| hidden_dim | Hidden dimension | 768, 4096, 8192 |
| num_layers | Layer count | 12, 24, 32 |
| num_heads | Attention heads | 8, 16, 32 |
| head_dim | Head dimension | 64, 128 |
| intermediate_size | FFN hidden | 2048, 11008 |

## Cost Reference

| Model | Parameters | VRAM (FP16) | Training Cost (1T tokens) |
|-------|------------|--------------|-------------------------|
| GPT-2 | 176M | 0.4 GB | ~$50 |
| Llama-2-7B | 6.4B | 14 GB | ~$2,500 |
| Llama-2-70B | 70B | 145 GB | ~$25,000 |

## Tech Stack

- Python 3.12+
- Z3 theorem prover (formal verification)
- FastAPI (optional REST API)
- Streamlit (optional web UI)

## Links

- [PyPI](https://pypi.org/project/archgene/)
- [GitHub](https://github.com/Tejas163/ArchGene)

## License

MIT