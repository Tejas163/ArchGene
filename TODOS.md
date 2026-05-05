# TODOS — Deferred Work

## Phase 1 Complete ✓

- Gene schema ✓
- Z3 verifier ✓
- smolagents agents ✓
- Benchmark suite ✓

---

## Phase 2 In Progress

### High Priority

| Item | Status | Notes |
|------|--------|-------|
| Evaluation benchmarking suite | DONE ✓ | benchmark.py verified |
| Production CI/CD pipeline | Pending | Not in Phase 1 scope |
| Model export to ONNX | DONE ✓ | PyTorch export works |
| Model export to HuggingFace | DONE ✓ | Config export works |

### Medium Priority

| Item | Status | Notes |
|------|--------|-------|
| LLM-guided generation | DONE ✓ | Using llama3.2:1b for reasoning |
| Crossover operator | DONE ✓ | Fixed, tested |
| Multi-objective optimization | DONE ✓ | Pareto optimization working |
| Archive old agents/ | DONE ✓ | Archived to agents/legacy_backup/ |

### Low Priority

| Item | Status | Notes |
|------|--------|-------|
| Production CI/CD pipeline | Pending | Not in scope |
| Web UI | Deferred | Streamlit/dash for visualization |
| API server | Deferred | REST API for remote evaluation |
| Model Zoo | Deferred | Pre-trained architectures library |

---

## Open Questions

1. **Exact gene schema** — designed in Phase 1, may evolve
2. **Simulation environment** — needs Implementor complete first
3. **Ollama model selection** — could upgrade to 3B for better reasoning

---

## Done

- Project setup
- Gene schema
- Z3 verification layer
- Evaluation API + scoring
- Real-time progress tracking
- Architecture visualization
- Multiple export formats
- History + persistence
- Custom agents (legacy)
- smolagents refactor
- Benchmark suite

Last updated: 2026-05-05