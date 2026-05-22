# ArchGene User Guide

## Installation

```bash
pip install archgene
```

Verify it works:

```bash
archgene --version
```

## Quick Start

### Verify any architecture before training

```bash
archgene verify --hidden 4096 --heads 32 --layers 24
```

Z3 mathematically proves your architecture is valid — or tells you exactly what's wrong.

### Estimate costs for a known model

```bash
archgene cost gpt2 --gpu H100
```

Shows VRAM, inference cost, training hours, and GPU bill before you spend a cent.

### Browse the model zoo

```bash
archgene zoo-list
archgene zoo-info llama2_7b
archgene zoo-evaluate llama2_7b
```

## Design an Architecture (Conversational Q&A)

The fastest way to get a verified architecture tailored to your needs:

```bash
archgene design
```

You'll answer 6 questions:
1. **Use case** — research, inference, fine-tuning, or edge
2. **Budget** — free, mid, high, or enterprise
3. **Constraints** — long context, low latency, low memory, quantization
4. **Architecture family** — Transformer, linear/SSM, or let the system decide
5. **Target size** — <1B, 1B-7B, 7B-30B, 30B+
6. **Context length** — 2K, 8K, 32K, or 128K

The system designs a verified architecture, shows cost estimates, and asks if you want to generate runnable PyTorch code.

## Generate Runnable Code

From any architecture, generate ready-to-run files:

```bash
# From CLI flags
archgene generate -d 4096 -l 32 -n 16 -i 11008 -o my_llm

# Or after running `archgene design`, say "yes" when prompted
```

Output:

```
my_llm/
├── model.py          # Full model definition (Attention, MLP, Transformer, LM head)
├── config.json       # HuggingFace-compatible configuration
├── train.py          # Training script with scheduler, validation, AMP
└── requirements.txt  # Dependencies
```

Run the model:

```bash
cd my_llm
pip install -r requirements.txt
python train.py
```

### Generated code adapts to your Gene flags

| Flag | Effect on generated code |
|------|--------------------------|
| `use_rope=True` | Adds `RotaryEmbedding` with sin/cos position cache |
| `num_kv_heads < num_heads` | Grouped Query Attention with KV head repetition |
| `use_gated_activation=True` | SwiGLU-style gated MLP (`gate_proj`, `up_proj`, `down_proj`) |
| `use_rms_norm=True` | Uses `RMSNorm` instead of `LayerNorm` |
| `use_flash_attention=True` | Imports FlashAttention, falls back to manual if unavailable |
| Training script | Cosine LR scheduler, validation split, AMP, gradient clipping |

## Compare Models for Your Budget

```bash
# Find the best model for $50 on an A100
archgene recommend -b 50 -g A100-40GB -u inference

# For fine-tuning on a budget
archgene recommend -b 100 -g A100-40GB -u fine-tuning
```

## Research a Custom Architecture

Describe your needs in natural language:

```bash
archgene research -p "efficient inference on mobile with 1-bit quantization"
archgene research -p "long context 128k for document analysis"
```

The research engine analyzes your prompt, selects an architecture family, designs parameters, and provides evidence-backed reasoning.

## Get Deployment Instructions

```bash
archgene deploy llama2_7b --platform huggingface
archgene deploy mistral_7b --platform vllm
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `archgene evaluate` | Score an architecture |
| `archgene verify` | Z3 formal verification |
| `archgene cost <model>` | GPU cost estimates |
| `archgene recommend` | Best model for your budget |
| `archgene design` | Conversational Q&A architecture designer |
| `archgene generate` | Generate runnable PyTorch code |
| `archgene research` | LLM-guided architecture design |
| `archgene zoo-list` | List pre-trained models |
| `archgene zoo-info <name>` | Show model details |
| `archgene zoo-evaluate <name>` | Score a zoo model |
| `archgene benchmark <name>` | MMLU/HumanEval estimates |
| `archgene deploy <name>` | Deployment instructions |
| `archgene guide` | Interactive educational guide |
| `archgene history` | Past evaluations |
| `archgene visualize` | ASCII/mermaid/JSON diagram |
| `archgene export` | Export to PyTorch/ONNX/config |

## Examples by Use Case

### Building a custom LLM

```bash
# 1. Design the architecture
archgene design

# 2. Or verify your config
archgene verify --hidden 4096 --heads 32 --layers 24

# 3. Estimate the bill
archgene cost custom --hidden 4096 --layers 32 --gpu A100

# 4. Generate code
archgene generate -d 4096 -l 32 -n 16 -i 11008

# 5. Train
cd generated && python train.py --steps 1000 --amp
```

### Fine-tuning an existing model

```bash
archgene recommend -b 200 -g A100-40GB -u fine-tuning
archgene zoo-evaluate llama2_7b
archgene cost llama2_7b --gpu A100 --batch-size 16
```

### Comparing architectures

```bash
archgene zoo-evaluate gpt2
archgene zoo-evaluate llama2_7b
archgene zoo-evaluate mistral_7b
```

## Generation Tracking

Each design session is saved to `~/.archgene/generations.json`. This tracks your monthly usage for the free tier (future feature).

View your stats:

```bash
cat ~/.archgene/generations.json
```
