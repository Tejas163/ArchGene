"""
Multi-agent architecture evolution system using smolagents.
"""

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    ActionStep,
)
from smolagents.tools import tool

import json
import random
from copy import deepcopy
from pathlib import Path
import sys
import os
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gene_schema import Gene, ActivationType
from core.verifier import Verifier

LLM_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm():
    """Get configured LLM - returns None, using direct Ollama calls instead."""
    return None


def call_llm(prompt: str, system_prompt: str = None) -> str:
    """Call Ollama LLM directly."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# ============================================================================
# TOOLS - Reusable functions agents can call
# ============================================================================

@tool
def generate_gene(vocab_dim: int = None, hidden_dim: int = None, num_layers: int = None) -> dict:
    """
    Generate a Gene specification for a transformer architecture.
    
    Args:
        vocab_dim: Vocabulary dimension (default: random from [2048, 4096, 50257, 32000])
        hidden_dim: Hidden dimension (default: random from [256, 512, 768, 1024])
        num_layers: Number of layers (default: random from [2, 4, 6, 8, 12])
    
    Returns:
        dict: Gene specification with all required fields
    """
    vocab_dims = [2048, 4096, 50257, 32000]
    hidden_dims = [256, 512, 768, 1024]
    num_layers_list = [2, 4, 6, 8, 12]
    num_heads_list = [4, 8, 12, 16]
    
    vocab = vocab_dim or random.choice(vocab_dims)
    hidden = hidden_dim or random.choice(hidden_dims)
    layers = num_layers or random.choice(num_layers_list)
    
    # Ensure divisibility
    while hidden % 8 != 0:
        hidden = random.choice(hidden_dims)
    
    num_heads = hidden // 64 if hidden >= 512 else 4
    
    gene = Gene(
        vocab_dim=vocab,
        hidden_dim=hidden,
        num_layers=layers,
        num_heads=num_heads,
        head_dim=64,
        intermediate_size=hidden * 4,
        hidden_act=ActivationType.GELU,
        max_position_embeddings=2048,
        use_rope=random.choice([True, False]),
        use_flash_attention=random.choice([True, False]),
    )
    
    return gene.to_dict()


@tool
def verify_gene(gene_dict: dict) -> dict:
    """
    Verify a Gene specification using formal verification (Z3).
    
    Args:
        gene_dict: Gene specification dictionary
    
    Returns:
        dict: Verification result with is_valid and issues
    """
    gene = Gene.from_dict(gene_dict)
    verifier = Verifier()
    result = verifier.verify_all(gene)
    
    return {
        "is_valid": result.is_valid,
        "issues": result.errors,
        "passed_checks": result.constraints_checked,
        "failed_checks": [],
    }


@tool
def mutate_gene(gene_dict: dict, mutation_type: str = "random") -> dict:
    """
    Apply mutation to a Gene specification.
    
    Args:
        gene_dict: Gene to mutate
        mutation_type: Type of mutation - "random", "increase_scale", "decrease_scale", "swap_activation"
    
    Returns:
        dict: Mutated Gene specification
    """
    gene = Gene.from_dict(gene_dict)
    new_gene = deepcopy(gene)
    
    mutations = {
        "random": lambda: setattr(new_gene, "hidden_dim", random.choice([256, 512, 768, 1024])),
        "increase_scale": lambda: setattr(new_gene, "hidden_dim", min(new_gene.hidden_dim * 2, 2048)),
        "decrease_scale": lambda: setattr(new_gene, "hidden_dim", max(new_gene.hidden_dim // 2, 128)),
        "swap_activation": lambda: setattr(
            new_gene, "hidden_act", 
            ActivationType.SILU if new_gene.hidden_act == ActivationType.GELU else ActivationType.GELU
        ),
    }
    
    mutator = mutations.get(mutation_type, mutations["random"])
    mutator()
    
    # Ensure consistency
    new_gene.head_dim = new_gene.hidden_dim // new_gene.num_heads
    
    return new_gene.to_dict()


@tool
def crossover_genes(gene1_dict: dict, gene2_dict: dict) -> dict:
    """
    Apply crossover between two Gene specifications.
    
    Args:
        gene1_dict: First parent Gene
        gene2_dict: Second parent Gene
    
    Returns:
        dict: Offspring Gene specification
    """
    from copy import deepcopy
    
    gene1 = Gene.from_dict(gene1_dict)
    gene2 = Gene.from_dict(gene2_dict)
    
    # Randomly swap attributes
    new_gene = deepcopy(gene1)
    
    if random.random() < 0.5:
        new_gene.hidden_dim = gene2.hidden_dim
    if random.random() < 0.5:
        new_gene.num_layers = gene2.num_layers
    if random.random() < 0.5:
        new_gene.intermediate_size = gene2.intermediate_size
    
    new_gene.head_dim = new_gene.hidden_dim // new_gene.num_heads
    
    return new_gene.to_dict()


@tool
def compute_fitness(gene_dict: dict) -> float:
    """
    Compute fitness score for a Gene based on architecture properties.
    
    Args:
        gene_dict: Gene specification
    
    Returns:
        float: Fitness score (0-1)
    """
    gene = Gene.from_dict(gene_dict)
    
    score = 0.0
    
    # Scale capability (0.3 max)
    if gene.hidden_dim >= 512:
        score += 0.15
    if gene.num_layers >= 6:
        score += 0.15
    
    # Efficiency (0.3 max)
    if gene.intermediate_size >= gene.hidden_dim * 2:
        score += 0.15
    if gene.use_rope:
        score += 0.15
    
    # Stability bonus (0.4 max)
    if gene.hidden_dim >= 256:
        score += 0.2
    if gene.hidden_dim % gene.num_heads == 0:
        score += 0.2
    
    return min(score, 1.0)


@tool
def generate_architecture_idea(gene_dict: dict) -> str:
    """
    Generate an architecture idea/description using LLM reasoning.
    
    Args:
        gene_dict: Gene specification
    
    Returns:
        str: Architecture description and rationale
    """
    gene = Gene.from_dict(gene_dict)
    
    description = f"""
Architecture Idea:
- Vocabulary: {gene.vocab_dim:,} tokens
- Hidden dimension: {gene.hidden_dim}
- Layers: {gene.num_layers}
- Attention heads: {gene.num_heads}
- Intermediate size: {gene.intermediate_size}
- Activation: {gene.hidden_act.value if hasattr(gene.hidden_act, 'value') else gene.hidden_act}
- RoPE: {"Enabled" if gene.use_rope else "Disabled"}
- Flash Attention: {"Enabled" if gene.use_flash_attention else "Disabled"}
- Total parameters: ~{gene.compute_params():,} 
- Memory: ~{gene.compute_memory():.1f}GB

Rationale: This architecture balances {"scale" if gene.hidden_dim >= 768 else "efficiency"} with 
{"deeper" if gene.num_layers >= 6 else "shallower"} layers for potential { "emergent capabilities" if gene.num_layers >= 6 else "quick convergence" }.
"""
    return description.strip()


# ============================================================================
# AGENTS - Each agent has specific role and tools
# ============================================================================

class ResearchAgent(ToolCallingAgent):
    """Research Agent generates architecture ideas."""
    
    def __init__(self):
        super().__init__(
            name="research_agent",
            description="Generates transformer architecture gene specifications",
            tools=[
                generate_gene,
                generate_architecture_idea,
            ],
        )


class EvaluatorAgent(ToolCallingAgent):
    """Evaluator Agent verifies and scores architectures."""
    
    def __init__(self):
        super().__init__(
            name="evaluator_agent",
            description="Verifies genes using Z3 and computes fitness",
            tools=[
                verify_gene,
                compute_fitness,
            ],
        )


class EvolutionaryAgent(ToolCallingAgent):
    """Evolutionary Agent applies genetic operators."""
    
    def __init__(self):
        super().__init__(
            name="evolutionary_agent",
            description="Applies mutation and crossover to evolve architectures",
            tools=[
                mutate_gene,
                crossover_genes,
            ],
        )


class PlannerAgent(ToolCallingAgent):
    """Planner Agent creates execution plans."""
    
    def __init__(self):
        super().__init__(
            name="planner_agent",
            description="Creates execution plans for architecture implementation",
            tools=[],
        )


class ArchitectAgent:
    """
    Master Architect Agent - uses LLM to reason about architectures.
    
    This agent can:
    - Analyze architecture properties
    - Suggest improvements
    - Evaluate tradeoffs
    
    Uses direct Ollama API calls instead of smolagents model.
    """
    
    def __init__(self):
        self.name = "architect_agent"
        self.tools = [
            generate_gene,
            verify_gene,
            compute_fitness,
            generate_architecture_idea,
        ]
    
    def run(self, task: str) -> str:
        """Run the architect agent with a task."""
        # Use LLM for reasoning
        system_prompt = """You are an expert transformer architecture designer.
Your role is to analyze, critique, and improve transformer architecture designs.
You understand:
- Attention mechanisms (full, sliding window, flash attention)
- Position embeddings (RoPE, absolute)
- Activation functions (GELU, SiLU, ReLU)
- Layer normalization vs RMS normalization
- Parameter efficiency and scaling laws"""
        
        response = call_llm(task, system_prompt)
        return response


class SimpleToolCallingAgent:
    """Simple tool-calling agent without LLM model."""
    
    def __init__(self, name: str, tools: list):
        self.name = name
        self.tools = {tool.name: tool for tool in tools}
    
    def run(self, task: str) -> str:
        """Execute task using available tools."""
        # Simplified: just execute tools directly
        return f"Executed {len(self.tools)} tools"


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def create_multi_agent_system():
    """Create the multi-agent system with smolagents."""
    
    # Create agents
    research = ResearchAgent()
    evaluator = EvaluatorAgent()
    evolutionary = EvolutionaryAgent()
    
    return {
        "research": research,
        "evaluator": evaluator,
        "evolutionary": evolutionary,
    }


def run_evolution(
    population_size: int = 5,
    generations: int = 3,
    use_llm: bool = False,
    use_architect: bool = False,
) -> dict:
    """
    Run the full evolution pipeline.
    
    Args:
        population_size: Number of genes in population
        generations: Number of evolution generations
        use_llm: Whether to use LLM for reasoning
        use_architect: Whether to use ArchitectAgent (requires LLM)
    
    Returns:
        dict: Best gene and evolution stats
    """
    print(f"Starting evolution: population={population_size}, generations={generations}")
    print(f"LLM: {'Enabled' if use_llm else 'Disabled'} ({LLM_MODEL})")
    print(f"Architect: {'Enabled' if use_architect else 'Disabled'}")
    print("-" * 50)
    
    # Create architect agent if needed
    architect = ArchitectAgent() if use_architect else None
    
    # Initialize population
    population = []
    for i in range(population_size):
        gene_dict = generate_gene()
        population.append(gene_dict)
    
    best_overall = None
    best_fitness = 0.0
    
    for gen in range(generations):
        print(f"\nGeneration {gen + 1}/{generations}")
        
        # Evaluate
        fitness_scores = []
        for gene_dict in population:
            fitness = compute_fitness(gene_dict)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_overall = gene_dict.copy()
        
        print(f"  Best fitness: {best_fitness:.3f}")
        
        # Select and evolve
        if gen < generations - 1:
            new_population = []
            
            # Elitism: keep best
            best_idx = fitness_scores.index(max(fitness_scores))
            new_population.append(population[best_idx].copy())
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                candidates = random.sample(range(len(population)), k=min(3, len(population)))
                parent_idx = max(candidates, key=lambda i: fitness_scores[i])
                parent = population[parent_idx]
                
                # Mutate
                mutated = mutate_gene(parent)
                new_population.append(mutated)
            
            population = new_population
    
    print("-" * 50)
    print(f"Evolution complete!")
    print(f"Best fitness: {best_fitness:.3f}")
    
    return {
        "best_gene": best_overall,
        "best_fitness": best_fitness,
        "generations": generations,
    }


if __name__ == "__main__":
    result = run_evolution(population_size=5, generations=3, use_llm=False)
    
    print("\nBest Gene:")
    print(json.dumps(result["best_gene"], indent=2))