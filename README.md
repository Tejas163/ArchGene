# ArchGene

Self-healing multi-agent cognitive architecture evaluation system. Evaluates LLM architectures using formal verification (Z3) and optionally generates novel architectures that pass checks.

## Features

- **Gene Schema** — Encode transformer architectures as Gene objects
- **Z3 Verification** — Formally verify architecture properties
- **smolagents** — Multi-agent system with tool-based agents
- **LLM Integration** — Uses Ollama (llama3.2:1b) for reasoning
- **Evolution** — Genetic algorithms for architecture optimization
- **Multi-objective** — Pareto optimization across multiple fitness targets

## Quick Start

```bash
# Run evaluation
python main.py evaluate

# Run benchmark
python benchmark.py --pop=10 --gen=5

# Multi-objective evolution
python multi_objective.py
```

## Architecture

```
┌─────────────┐     ┌─────────┐     ┌─────────────┐
│ Research   │────▶│ Planner │────▶│Implementor│
│ Agent      │     │ Agent  │     │ Agent      │
└─────────────┘     └─────────┘     └─────────────┘
      ▲                                    │
      │                                    ▼
┌─────────────┐     ┌─────────┐     ┌─────────────┐
│Evolutionary│◀────│ Code   │◀────│Simulation │
│ Agent     │     │ Judge  │     │ Agent     │
└─────────────┘     └─────────┘     └─────────────┘
```

## Tech Stack

- Python 3.12+
- z3-solver (formal verification)
- smolagents (agent framework)
- Ollama (llama3.2:1b for reasoning)
- PyTorch (model export)

## CLI Commands

```bash
python main.py evaluate [gene_config]  # Evaluate architecture
python main.py verify [gene_config] # Verify with Z3
python main.py benchmark            # Run benchmarks
python main.py visualize          # Visualize architecture
```

## License

MIT