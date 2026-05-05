# ArchGene - AI Architecture Evaluation System

## What It Does

- **Evaluates** transformer architectures with formal verification (Z3)
- **Estimates** real-world costs (inference, training) before you build
- **Benchmarks** expected performance (MMLU, HumanEval)
- **Deploys** with instructions for HuggingFace, vLLM, Replicate

## Quick Start

```bash
# Evaluate default architecture
python main.py evaluate

# Verify with Z3
python main.py verify

# Get cost estimates
python main.py cost gpt2
```

## Commands

| Command | Description |
|--------|-------------|
| `python main.py evaluate` | Evaluate architecture |
| `python main.py verify` | Z3 formal verification |
| `python main.py cost <model>` | GPU cost estimates |
| `python main.py recommend` | AI-powered recommendation |
| `python main.py research` | LLM-guided architecture design |
| `python main.py evolve` | Genetic algorithm evolution |
| `python main.py guide` | Interactive educational guide |
| `python main.py zoo-list` | List pre-trained models |

## Architecture

```
ArchGene/
├── main.py                 # CLI entry point
├── core/
│   ├── gene_schema.py     # Gene dataclass
│   ├── verifier.py        # Z3 verification
│   ├── evaluator.py      # Scoring
│   ├── cost_estimator.py # GPU costs
│   ├── model_zoo.py      # 17 architectures
│   └── ...
├── training/              # Training module
├── tests/                # Test suite (44 tests)
└── api_server.py         # REST API
```

## Tech Stack

- Python 3.12+
- z3-solver (formal verification)
- smolagents (agent framework)
- Ollama (llama3.2:1b for reasoning)

## Skill Routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill tool as your FIRST action.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken" → invoke investigate  
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa

## Development

```bash
# Run tests
pytest tests/ -v

# Run CLI
python main.py --help
```

## Notes

- CLAUDE.md created for gstack skill routing
- Module reorganization deferred (60 files, 41 internal refs to update)