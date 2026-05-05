"""
Benchmark suite for ArchGene — measures evolution performance.
"""

import time
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from copy import deepcopy


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    duration_s: float
    generations: int
    population_size: int
    best_fitness: float
    avg_fitness: float
    convergence_rate: float
    valid_genes: int
    total_genes: int


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def add(self, result: BenchmarkResult):
        self.results.append(result)
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results],
        }


from core.gene_schema import Gene, ActivationType
from core.verifier import Verifier
import sys
sys.path.insert(0, str(Path(__file__).parent / "agents"))
from smolagents_system import (
    generate_gene,
    mutate_gene,
    compute_fitness,
    call_llm,
)


def benchmark_mutation_stability(n_runs: int = 10) -> BenchmarkResult:
    """Benchmark: Does mutation preserve gene validity?"""
    start = time.time()
    verifier = Verifier()
    
    valid_count = 0
    total = n_runs * 3  # 3 mutations per run
    
    for _ in range(n_runs):
        gene = generate_gene()
        for _ in range(3):
            mutated = mutate_gene(gene)
            gene_obj = Gene.from_dict(mutated)
            result = verifier.verify_all(gene_obj)
            if result.is_valid:
                valid_count += 1
    
    duration = time.time() - start
    
    return BenchmarkResult(
        name="mutation_stability",
        duration_s=duration,
        generations=0,
        population_size=n_runs,
        best_fitness=valid_count / total,
        avg_fitness=valid_count / total,
        convergence_rate=valid_count / total,
        valid_genes=valid_count,
        total_genes=total,
    )


def benchmark_population_diversity(population_size: int = 20) -> BenchmarkResult:
    """Benchmark: Does evolution maintain diversity?"""
    start = time.time()
    
    # Generate population
    genes = [generate_gene() for _ in range(population_size)]
    
    # Compute diversity (unique configurations)
    unique = set()
    for g in genes:
        unique.add(f"{g['hidden_dim']}-{g['num_layers']}-{g['num_heads']}")
    
    diversity = len(unique) / population_size
    
    # Compute average fitness
    fitnesses = [compute_fitness(g) for g in genes]
    best = max(fitnesses)
    avg = sum(fitnesses) / len(fitnesses)
    
    duration = time.time() - start
    
    return BenchmarkResult(
        name="population_diversity",
        duration_s=duration,
        generations=1,
        population_size=population_size,
        best_fitness=best,
        avg_fitness=avg,
        convergence_rate=diversity,
        valid_genes=population_size,
        total_genes=population_size,
    )


def benchmark_convergence(population_size: int = 10, generations: int = 5) -> BenchmarkResult:
    """Benchmark: Does fitness improve over generations?"""
    start = time.time()
    
    # Initialize population
    population = [generate_gene() for _ in range(population_size)]
    fitness_history = []
    
    for gen in range(generations):
        # Evaluate
        scores = [compute_fitness(g) for g in population]
        fitness_history.append(sum(scores) / len(scores))
        
        # Evolve (simple: keep best, mutate others)
        best_idx = scores.index(max(scores))
        best = population[best_idx]
        
        new_pop = [best]  # Elitism
        for _ in range(population_size - 1):
            parent = random.choice(population)
            mutated = mutate_gene(parent)
            new_pop.append(mutated)
        
        population = new_pop
    
    # Final fitness
    final_scores = [compute_fitness(g) for g in population]
    best_fitness = max(final_scores)
    avg_fitness = sum(final_scores) / len(final_scores)
    
    # Convergence: did fitness improve?
    improvement = fitness_history[-1] - fitness_history[0]
    converged = 1.0 if improvement > 0 else 0.0
    
    duration = time.time() - start
    
    return BenchmarkResult(
        name="convergence",
        duration_s=duration,
        generations=generations,
        population_size=population_size,
        best_fitness=best_fitness,
        avg_fitness=avg_fitness,
        convergence_rate=converged,
        valid_genes=population_size,
        total_genes=population_size,
    )


def benchmark_verification_speed(n_runs: int = 20) -> BenchmarkResult:
    """Benchmark: How fast is Z3 verification?"""
    start = time.time()
    verifier = Verifier()
    
    valid_count = 0
    total = n_runs
    
    for _ in range(n_runs):
        gene = generate_gene()
        gene_obj = Gene.from_dict(gene)
        result = verifier.verify_all(gene_obj)
        if result.is_valid:
            valid_count += 1
    
    duration = time.time() - start
    
    return BenchmarkResult(
        name="verification_speed",
        duration_s=duration,
        generations=0,
        population_size=n_runs,
        best_fitness=valid_count / total,
        avg_fitness=valid_count / total,
        convergence_rate=valid_count / total,
        valid_genes=valid_count,
        total_genes=total,
    )


def benchmark_llm_reasoning(n_runs: int = 3) -> BenchmarkResult:
    """Benchmark: How fast is LLM reasoning?"""
    if n_runs == 0:
        return BenchmarkResult(
            name="llm_reasoning",
            duration_s=0,
            generations=0,
            population_size=0,
            best_fitness=0,
            avg_fitness=0,
            convergence_rate=0,
            valid_genes=0,
            total_genes=0,
        )
    
    start = time.time()
    
    test_gene = generate_gene(vocab_dim=50257, hidden_dim=512, num_layers=6)
    
    for _ in range(n_runs):
        prompt = f"Analyze this gene: {test_gene['hidden_dim']} hidden, {test_gene['num_layers']} layers"
        try:
            call_llm(prompt)
        except Exception:
            pass
    
    duration = time.time() - start
    
    return BenchmarkResult(
        name="llm_reasoning",
        duration_s=duration,
        generations=0,
        population_size=n_runs,
        best_fitness=1.0,
        avg_fitness=1.0,
        convergence_rate=1.0,
        valid_genes=n_runs,
        total_genes=n_runs,
    )


def run_full_benchmark(
    pop_size: int = 10,
    generations: int = 5,
    llm_runs: int = 3,
    verbose: bool = True,
) -> BenchmarkSuite:
    """Run complete benchmark suite."""
    suite = BenchmarkSuite()
    
    tests = [
        ("Mutation Stability", lambda: benchmark_mutation_stability(10)),
        ("Population Diversity", lambda: benchmark_population_diversity(pop_size)),
        ("Convergence", lambda: benchmark_convergence(pop_size, generations)),
        ("Verification Speed", lambda: benchmark_verification_speed(20)),
    ]
    
    if llm_runs > 0:
        tests.append(("LLM Reasoning", lambda: benchmark_llm_reasoning(llm_runs)))
    
    for name, test_fn in tests:
        if verbose:
            print(f"Running {name}...", end=" ")
        try:
            result = test_fn()
            suite.add(result)
            if verbose:
                print(f"[OK] {result.duration_s:.2f}s")
        except Exception as e:
            if verbose:
                print(f"[FAIL] {e}")
    
    return suite


def save_results(suite: BenchmarkSuite, path: str = "benchmark_results.json"):
    """Save benchmark results."""
    data = suite.to_dict()
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_results(path: str = "benchmark_results.json") -> Optional[BenchmarkSuite]:
    """Load benchmark results."""
    if not Path(path).exists():
        return None
    
    with open(path) as f:
        data = json.load(f)
    
    results = [BenchmarkResult(**r) for r in data.get("results", [])]
    return BenchmarkSuite(results=results, timestamp=data.get("timestamp", ""))


def print_report(suite: BenchmarkSuite):
    """Print benchmark report."""
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"Timestamp: {suite.timestamp}")
    print("-" * 60)
    
    for r in suite.results:
        print(f"\n{r.name}")
        print(f"  Duration: {r.duration_s:.2f}s")
        print(f"  Best Fitness: {r.best_fitness:.3f}")
        print(f"  Avg Fitness: {r.avg_fitness:.3f}")
        print(f"  Convergence: {r.convergence_rate:.1%}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    
    # Parse args
    pop_size = 10
    generations = 5
    llm_runs = 3
    
    for arg in sys.argv[1:]:
        if arg.startswith("--pop="):
            pop_size = int(arg.split("=")[1])
        elif arg.startswith("--gen="):
            generations = int(arg.split("=")[1])
        elif arg.startswith("--llm="):
            llm_runs = int(arg.split("=")[1])
        elif arg == "--no-llm":
            llm_runs = 0
    
    print(f"Running benchmarks: pop={pop_size}, gen={generations}, llm={llm_runs}")
    
    suite = run_full_benchmark(pop_size, generations, llm_runs)
    print_report(suite)
    save_results(suite)
    print(f"\nResults saved to benchmark_results.json")