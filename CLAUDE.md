# ArchGene — Verify LLM Architectures Before You Train

## What It Does

- **Z3 Formal Verification** — mathematically proves architectures are valid
- **Cost Estimation** — GPU training/inference costs before you spend
- **Benchmark Projections** — MMLU, HumanEval, throughput estimates
- **Model Zoo** — compare against GPT-2, Llama-2, Mistral, Qwen, 17 architectures

## Quick Start

```bash
pip install archgene

# Verify your architecture BEFORE training
archgene verify --hidden 4096 --heads 32 --layers 24

# Get cost estimate
archgene cost gpt2 --gpu H100

# Browse the model zoo
archgene zoo-list
```

## Commands

| Command | Description |
|---------|-------------|
| `archgene evaluate` | Evaluate architecture |
| `archgene verify` | Z3 formal verification |
| `archgene cost <model>` | GPU cost estimates |
| `archgene recommend` | AI-powered recommendation |
| `archgene research` | LLM-guided architecture design |
| `archgene zoo-list` | List pre-trained models |
| `archgene design` | Conversational Q&A architecture designer |
| `archgene generate` | Generate runnable PyTorch code from architecture |
| `archgene guide` | Interactive educational guide |

## Architecture

```
ArchGene/
├── cli/main.py              # CLI entry point
├── src/archgene/
│   ├── gene_schema.py       # Gene dataclass
│   ├── verifier.py          # Z3 verification
│   ├── evaluation.py        # Scoring
│   ├── cost_estimator.py    # GPU costs
│   ├── model_zoo.py         # 17 architectures
│   ├── research_engine.py   # LLM-guided design
│   ├── design_session.py    # Conversational Q&A + headless API
│   └── kernel_generator.py  # Runnable PyTorch code generation + zip
├── web_app.py               # Streamlit web UI (pip install archgene[web])
├── tests/                   # 25 tests
└── index.html               # Landing page (Netlify)
```

## Tech Stack

- Python 3.10+
- z3-solver (formal verification)
- torch (visualization/export + kernel generation)
- rich (CLI output)
- streamlit (web UI, optional: `pip install archgene[web]`)

## Generated Code Features

When using `archgene generate` or the design session's code gen:

| Gene Flag | Generated Code |
|-----------|---------------|
| `use_rope=True` | RotaryEmbedding class with half-rotary application |
| `num_kv_heads < num_heads` | Grouped Query Attention with KV head repeat |
| `use_gated_activation=True` | SwiGLU-style gated MLP (gate/up/down) |
| `use_rms_norm=True` | RMSNorm (LayerNorm otherwise) |
| `use_flash_attention=True` | FlashAttention import with manual fallback |
| Training script | Cosine LR scheduler, validation loop, AMP support |

## Skill Routing

- Product ideas, brainstorming → invoke office-hours
- Bugs, errors, "why is this broken" → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site → invoke qa

## Development

```bash
pytest tests/ -v
python main.py --help
```

## Notes

- v0.4.0 — stripped consulting bloat, focused on core verification + cost estimation
- Platform vision: conversational Q&A product built on top of ArchGene core
- Web UI: `streamlit run web_app.py` (requires `pip install archgene[web]`)