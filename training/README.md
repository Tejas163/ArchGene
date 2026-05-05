# Training Module

This module provides autonomous architecture evolution based on the Karpathy AutoResearch approach.

## Directory Structure

```
training/
├── train.py           # Single-file training (edit by agents)
├── program.md         # Agent instructions
├── experiments/      # Experiment logs
├── agents/           # Agent system
└── README.md
```

## Quick Start

```bash
# Run autonomous evolution
python -m training.evolve --budget 60

# Run single training session
python -m training.train --depth 8

# List experiments
python -m training.experiments --list
```

## Architecture

The training system combines:
1. **ArchGene** - Architecture evaluation & Z3 verification
2. **AutoResearch** - Autonomous agent-based evolution
3. **Research Engine** - LLM-powered architecture design

## Key Features

- Fixed time budget (not fixed steps)
- val_bpb metric (validation bits per byte)
- Auto-evolution with agents
- Z3 formal verification before training
- Integration with ArchGene cost estimation