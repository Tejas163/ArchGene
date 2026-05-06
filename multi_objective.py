"""
Multi-objective optimization for ArchGene.
Optimizes across multiple fitness targets simultaneously.
"""

import random
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

from src.archgene.gene_schema import Gene, ActivationType, ArchitectureFamily, QuantizationType


@dataclass
class FitnessTargets:
    """Multiple fitness objectives."""
    capability: float = 0.0   # 0-1: larger models have more capability
    efficiency: float = 0.0    # 0-1: parameter efficiency
    stability: float = 0.0    # 0-1: training stability
    memory_fit: float = 0.0   # 0-1: fits in memory budget
    quantization_boost: float = 0.0  # 0-1: bonus for quantized
    
    def total(self) -> float:
        return (self.capability + self.efficiency + 
                self.stability + self.memory_fit + self.quantization_boost) / 5


def compute_multi_objective_fitness(gene_dict: dict) -> FitnessTargets:
    """Compute fitness across multiple objectives."""
    gene = Gene.from_dict(gene_dict)
    
    targets = FitnessTargets()
    
    # Capability: larger is better, with bonuses for efficient architectures
    if gene.hidden_dim >= 768:
        targets.capability = 0.8
    elif gene.hidden_dim >= 512:
        targets.capability = 0.6
    elif gene.hidden_dim >= 256:
        targets.capability = 0.4
    else:
        targets.capability = 0.2
    
    if gene.num_layers >= 8:
        targets.capability += 0.2
    elif gene.num_layers >= 4:
        targets.capability += 0.1
    
    # Architecture family bonuses
    if gene.arch_family == ArchitectureFamily.BITNET:
        targets.capability += 0.3  # 1-bit matches 7B capability at smaller size
    elif gene.arch_family == ArchitectureFamily.MAMBA:
        targets.capability += 0.1  # SSM provides long context
    elif gene.arch_family == ArchitectureFamily.RWKV:
        targets.capability += 0.1  # Efficient attention
    
    # Efficiency: parameter per capability
    params = gene.compute_params()
    effective_params = gene.compute_effective_params()
    
    if gene.quant_type != QuantizationType.NONE:
        # Quantized models are more efficient
        targets.efficiency = 0.9
    elif effective_params < 100_000_000:
        targets.efficiency = 0.8
    elif effective_params < 500_000_000:
        targets.efficiency = 0.5
    else:
        targets.efficiency = 0.2
    
    # Stability: known good configurations
    targets.stability = 0.8
    if gene.hidden_act == ActivationType.GELU:
        targets.stability += 0.2
    if gene.arch_family in [ArchitectureFamily.MAMBA, ArchitectureFamily.RWKV]:
        targets.stability += 0.1  # Newer but proven
    
    # Memory fit: 8GB budget
    memory_gb = gene.compute_memory()
    if memory_gb < 2:
        targets.memory_fit = 1.0
    elif memory_gb < 4:
        targets.memory_fit = 0.8
    elif memory_gb < 6:
        targets.memory_fit = 0.5
    else:
        targets.memory_fit = 0.2
    
    # Quantization bonus (BitNet performs above its weight class)
    if gene.quant_type == QuantizationType.BITNET_1BIT:
        targets.quantization_boost = 0.8  # 1-bit is very efficient
    elif gene.quant_type == QuantizationType.BITNET_2BIT:
        targets.quantization_boost = 0.5
    elif gene.quant_type in [QuantizationType.GPTQ, QuantizationType.AWQ]:
        targets.quantization_boost = 0.3
    
    return targets


def pareto_dominant(a: FitnessTargets, b: FitnessTargets) -> bool:
    """Check if a dominates b (Pareto optimal - not worse in any objective)."""
    better_in_any = False
    
    for obj in ['capability', 'efficiency', 'stability', 'memory_fit', 'quantization_boost']:
        av = getattr(a, obj)
        bv = getattr(b, obj)
        
        if av < bv - 0.01:  # a is worse
            return False
        if av > bv + 0.01:  # a is better
            better_in_any = True
    
    return better_in_any


def select_pareto_parents(
    population: list[dict],
    fitness_scores: list[FitnessTargets],
    tournament_size: int = 3,
) -> tuple[dict, dict]:
    """Select parents using Pareto dominance."""
    # Find Pareto front
    pareto_front = []
    for i, f in enumerate(fitness_scores):
        is_dominated = False
        for j, g in enumerate(fitness_scores):
            if i != j and pareto_dominant(g, f):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(i)
    
    # Tournament selection from Pareto front
    if len(pareto_front) >= 2:
        parents = random.sample(pareto_front, 2)
        return population[parents[0]], population[parents[1]]
    
    # Fallback: random tournament
    indices = random.sample(range(len(population)), 
                          min(tournament_size, len(population)))
    best = max(indices, key=lambda i: fitness_scores[i].total())
    second_best = max([i for i in indices if i != best], 
                    key=lambda i: fitness_scores[i].total())
    
    return population[best], population[second_best]


def run_multi_objective_evolution(
    population_size: int = 10,
    generations: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Run multi-objective evolution using Pareto optimization.
    """
    from agents.smolagents_system import generate_gene, mutate_gene, crossover_genes
    
    # Initialize population
    population = [generate_gene() for _ in range(population_size)]
    
    best_overall = None
    best_total = 0.0
    
    for gen in range(generations):
        # Evaluate: compute multi-objective fitness
        fitness_scores = [compute_multi_objective_fitness(g) for g in population]
        
        # Track best
        for i, f in enumerate(fitness_scores):
            if f.total() > best_total:
                best_total = f.total()
                best_overall = population[i].copy()
        
        if verbose:
            avg_cap = sum(f.capability for f in fitness_scores) / len(fitness_scores)
            avg_eff = sum(f.efficiency for f in fitness_scores) / len(fitness_scores)
            avg_stab = sum(f.stability for f in fitness_scores) / len(fitness_scores)
            avg_mem = sum(f.memory_fit for f in fitness_scores) / len(fitness_scores)
            print(f"Gen {gen+1}: cap={avg_cap:.2f}, eff={avg_eff:.2f}, "
                  f"stab={avg_stab:.2f}, mem={avg_mem:.2f}")
        
        # Evolve using Pareto selection
        new_population = []
        
        # Elitism: keep top performer
        best_idx = max(range(len(fitness_scores)), 
                    key=lambda i: fitness_scores[i].total())
        new_population.append(deepcopy(population[best_idx]))
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1, parent2 = select_pareto_parents(population, fitness_scores)
            
            # Crossover
            if random.random() < 0.7:
                child = crossover_genes(parent1, parent2)
            else:
                child = deepcopy(parent1)
            
            # Mutation
            if random.random() < 0.3:
                child = mutate_gene(child)
            
            new_population.append(child)
        
        population = new_population
    
    return {
        "best_gene": best_overall,
        "best_fitness": compute_multi_objective_fitness(best_overall) if best_overall else None,
        "generations": generations,
    }


if __name__ == "__main__":
    print("Running multi-objective evolution...")
    result = run_multi_objective_evolution(population_size=5, generations=3)
    
    print("\nBest solution:")
    print(f"  Gene: {result['best_gene']}")
    if result['best_fitness']:
        f = result['best_fitness']
        print(f"  Fitness: cap={f.capability:.2f}, eff={f.efficiency:.2f}, "
              f"stab={f.stability:.2f}, mem={f.memory_fit:.2f}")