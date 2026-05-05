# ArchGene

Self-healing multi-agent cognitive architecture evaluation system. Evaluates LLM architectures using formal verification (Z3) and optionally generates novel architectures that pass checks.

## What It Does

- **Evaluates** transformer architectures with formal verification (Z3)
- **Estimates** real-world costs (inference, training) before you build
- **Benchmarks** expected performance (MMLU, HumanEval)
- **Deploys** with instructions for HuggingFace, vLLM, Replicate

## Quick Start

```bash
# Clone and install
pip install -e .

# Evaluate the default architecture
python main.py evaluate

# Verify with Z3 formal verification
python main.py verify
```

## CLI Commands

### Evaluation
```bash
python main.py evaluate              # Evaluate default Gene
python main.py evaluate --vocab 32000 --hidden 768 --layers 12
```

### Verification (Z3 formal verification)
```bash
python main.py verify               # Verify with Z3
python main.py verify --verbose   # Show constraint violations
```

### Benchmarking
```bash
python main.py benchmark           # Run genetic algorithm benchmark
python benchmark.py --pop 10 --gen 5
```

### Model Zoo (pre-trained architectures)
```bash
python main.py zoo-list           # List all architectures
python main.py zoo-info gpt2   # Show gpt2 details
python main.py zoo-evaluate gpt2
```

### Cost Estimation
```bash
python main.py cost gpt2                 # Default GPU (A100-40GB)
python main.py cost gpt2 --gpu H100        # Specify GPU
python main.py cost gpt2 --gpu A10 --batch-size 8
```

### Benchmark Estimates
```bash
python main.py benchmark gpt2
python main.py benchmark llama2_7b
```

### Deployment
```bash
python main.py deploy gpt2 --platform huggingface
python main.py deploy gpt2 --platform vllm
python main.py deploy gpt2 --platform replicate
```

## Architecture CLI

```bash
# Evaluate custom gene
python main.py evaluate --vocab 32000 --hidden 4096 --layers 32 --heads 32

# Visualize architecture
python main.py visualize

# Show history
python main.py history
```

## Gene Schema

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| vocab_dim | Vocabulary size | 2048, 4096, 32000, 50257 |
| hidden_dim | Hidden dimension | 256, 512, 768, 1024, 4096 |
| num_layers | Number of layers | 2, 4, 6, 8, 12, 32 |
| num_heads | Attention heads | 4, 8, 12, 16, 32 |
| head_dim | Head dimension | 64, 128 |
| intermediate_size | FFN hidden size | hidden_dim * 4 |

## Cost Estimation

| Architecture | Parameters | VRAM | Inference Cost | Training (1T tokens) |
|---------------|------------|------|---------------|---------------------|
| gpt2 | 176M | 0.4GB | $0.01/M | 1 GPU hour |
| gpt2-medium | 455M | 0.9GB | $0.02/M | 3 GPU hours |
| llama2-7b | 6.4B | 6.9GB | $0.15/M | 56 GPU hours |
| mistral-7b | 7.2B | 11.5GB | $0.18/M | 62 GPU hours |
| qwen2-1.5b | 1.4B | 2.8GB | $0.06/M | 12 GPU hours |

## API Server

```bash
# Start API server
python api_server.py

# Or with custom host/port
python api_server.py --host 0.0.0.0 --port 8000

# Set API key (required)
export ARCHGENE_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### API Endpoints

- `POST /evaluate` — Evaluate a gene
- `POST /verify` — Verify with Z3
- `GET /zoo` — List Model Zoo
- `GET /zoo/{name}` — Get architecture details

## Web UI (Local)

```bash
streamlit run web_ui.py
```

Opens a local Streamlit UI for interactive visualization.

## Tech Stack

- Python 3.12+
- z3-solver (formal verification)
- smolagents (agent framework)
- Ollama (llama3.2:1b for reasoning)
- FastAPI (API server)
- Streamlit (local UI)

## Files

```
ArchGene/
├── main.py                 # CLI entry point
├── api_server.py           # FastAPI server
├── web_ui.py              # Streamlit UI
├── benchmark.py           # Genetic algorithm benchmark
├── multi_objective.py     # Pareto optimization
├── core/
│   ├── gene_schema.py     # Gene dataclass
│   ├── verifier.py       # Z3 verification
│   ├── evaluation.py    # Scoring
│   ├── cost_estimator.py # Cost estimation
│   ├── benchmark_integration.py # Benchmark estimates
│   ├── deployment.py    # Deployment instructions
│   ├── model_zoo.py     # Pre-trained architectures
│   └── exporter.py     # Model export (PyTorch, ONNX, HF)
└── agents/
    └── smolagents_system.py # Multi-agent system
```

## License

MIT