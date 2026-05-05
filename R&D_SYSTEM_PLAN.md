# R&D System Plan — Self-Healing Multi-Agent Cognitive Architecture Generator

**Goal:** Build a system that evaluates LLM architectures using formal verification (Z3/CVC5), and optionally generates novel architectures that pass those checks.

**Refocused from:** "generate novel ideas" → "evaluation-first, then generation if evaluation works"

**Core hypothesis to test:** Can an 8GB laptop evaluate architecture properties formally? If yes, can Llama 1B explore meaningfully better than random?

**Constraints:**
- 8GB RAM laptop (upgraded from 4GB)
- No external LLMs for code execution (pure algorithmic)
- LLM (Llama 3.2 1B via Ollama) only for reasoning/guidance/judgment
- Formal verification using Z3/CVC5 solvers
- Hybrid approach: algorithmic execution + LLM augmentation
- Code Judge runs ALWAYS before execution
- Trusted providers only (Meta/Ollama)

---

## Architecture

```
┌─────────────┐     ┌─────────┐     ┌─────────────┐
│ Research   │────▶│ Planner │────▶│Implementor│
│ Agent     │     │ Agent  │     │ Agent     │
└─────────────┘     └─────────┘     └─────────────┘
      ▲                                    │
      │                                    ▼
┌─────────────┐     ┌─────────┐     ┌─────────────┐
│Evolutionary│◀────│ Code   │◀────│Simulation│
│ Agent    │     │ Judge  │     │ Agent    │
└─────────────┘     └─────────┘     └─────────────┘
```

### Agent Definitions

| Agent | Role | LLM Usage | Type |
|-------|------|----------|-----|
| Research | Generate architecture ideas | Yes | Algorithmic + LLM |
| Planner | Structure into executable plans | Yes | Algorithmic + LLM |
| Implementor | Write working code | No | Pure algorithmic |
| Code Judge | Verify code before execution | Yes | Hybrid |
| Evolutionary | Apply genetic operators | No | Pure algorithmic |
| Simulation | Test for emergence | No | Pure algorithmic |

### Feedback Loops

- Research ⇄ Code Judge: Idea → verify feasibility → iterate
- Planner ⇄ Evolutionary: Plan → mutate → re-plan
- Implementor ⇄ Simulation: Code → test → fix → re-test
- All agents: Bidirectional (human brain-like cognition)

---

## Recovery Flow

Every generated code passes through:

1. **Code Judge** — Always on before execution
2. **Auto-revise** — 3 attempts (LLM self-fix)
3. **Evolutionary fix** — 3 attempts (genetic operators)
4. **Human review** — Escalation if all else fails

---

## Technical Decisions

- **Framework:** PyTorch + torch.compile (NumPy → PyTorch for production ecosystem, 2.3x speedup)
- **LLM:** Llama 3.2 1B via Ollama (~2GB RAM)
- **Memory:** ~2GB PyTorch + ~1.5GB Llama 3.2 1B = ~3.5GB (8GB budget leaves comfortable buffer)
- **Verifier:** Z3/CVC5 for formal verification

---

## Directory Structure

```
main.py
requirements.txt
agents/
  research_agent.py
  planner_agent.py
  implementor_agent.py
  code_judge.py
  evolutionary_agent.py
  simulation_agent.py
core/
  gene_schema.py
  verifier.py
  code_generator.py
```

---

## Phase 1: Core + Research Agent (Evaluation-First)

1. Project setup (requirements.txt)
2. Gene schema for architecture encoding
3. Z3/CVC5 verification layer
4. Evaluation API + scoring
5. Real-time progress tracking
6. Architecture visualization
7. Multiple export formats (PyTorch, ONNX, HuggingFace)
8. History + persistence

## Deferred to TODOS.md

- Generation agent (after evaluation proves viable)
- Evolutionary agent (if generation added)
- All agents from original plan (except Research for evaluation)

---

## NOT In Scope

- Full 5-agent implementation (Phase 1 only)
- Production deployment pipeline
- Evaluation/sbenchmarking suite

## Open Questions

1. Ollama installation — user must run `ollama pull llama3.2:1b` manually
2. Exact gene schema structure — to be designed in Phase 1
3. Simulation environment — to be designed after Implementor is complete