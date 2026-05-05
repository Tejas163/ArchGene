"""
Agent package for ArchGene.
Uses smolagents for multi-agent orchestration.
"""

# smolagents-based system
from agents.smolagents_system import (
    generate_gene,
    verify_gene,
    mutate_gene,
    crossover_genes,
    compute_fitness,
    generate_architecture_idea,
    call_llm,
    LLM_MODEL,
    run_evolution,
)

__all__ = [
    "generate_gene",
    "verify_gene",
    "mutate_gene",
    "crossover_genes",
    "compute_fitness",
    "generate_architecture_idea",
    "call_llm",
    "LLM_MODEL",
    "run_evolution",
]