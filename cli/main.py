import warnings
warnings.filterwarnings("ignore")

import click
from rich.console import Console
from rich.panel import Panel
from datetime import datetime
from pathlib import Path
import json
import time
import subprocess
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

__version__ = "0.2.0"

from src.archgene.gene_schema import Gene, ActivationType
from src.archgene.verifier import Verifier
from src.archgene.evaluation import Evaluator, EvaluationRecord, EvaluationHistory
from src.archgene.progress import ProgressTracker
from src.archgene.visualization import ArchitectureVisualizer
from src.archgene.exporter import Exporter
from src.archgene.model_zoo import ModelZoo
from src.archgene.cost_estimator import CostEstimator
from src.archgene.benchmark_integration import BenchmarkIntegration
from src.archgene.deployment import ModelDeployment
from src.archgene.interactive_guide import run_guide


console = Console()
tracker = ProgressTracker()
verifier = Verifier()
evaluator = Evaluator()

DATA_DIR = Path("~/.archgene").expanduser()


def print_welcome():
    """Print welcome message with helpful tips."""
    console.print(Panel.fit(
        "[bold cyan]ArchGene[/bold cyan] - AI Architecture Evaluator\n"
        "[dim]Find the right LLM architecture for your budget[/dim]\n\n"
        "[bold]Quick Start:[/bold]\n"
        "  python main.py evaluate          # Evaluate default architecture\n"
        "  python main.py cost gpt2       # Estimate costs for GPT-2\n"
        "  python main.py zoo-list        # Browse pre-trained models\n\n"
        "[bold]Need Help?[/bold]\n"
        "  python main.py --help         # Full command list\n"
        "  python main.py demo           # Run interactive demo",
        title="Welcome to ArchGene v" + __version__,
        border_style="cyan"
    ))


@click.group()
@click.version_option(version=__version__)
def cli():
    """ArchGene: Evaluate LLM architectures using formal verification.
    
    Find the right architecture for your budget and use case.
    
    Examples:
        python main.py evaluate              # Quick evaluation
        python main.py cost llama2_7b      # Cost estimate
        python main.py recommend           # AI-powered recommendation
    """
    pass


@cli.command()
@click.option("--guide", "-g", is_flag=True, help="Run full interactive guide instead of quick demo")
def demo(guide):
    """Run an interactive demo of ArchGene features.
    
    Quick demo (default):
        python main.py demo
    
    Full interactive guide:
        python main.py demo --guide
    """
    if guide:
        run_guide(save_to_history=True)
        return
    
    from rich.spinner import Spinner
    from rich.progress import Progress
    
    print_welcome()
    
    console.print("\n[bold cyan]Running demo...[/bold cyan]\n")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Evaluating architectures...", total=3)
        
        gene = Gene()
        progress.update(task, advance=1)
        
        result = evaluator.evaluate(gene)
        progress.update(task, advance=1)
        
        est = CostEstimator.full_estimate(gene)
        progress.update(task, advance=1)
    
    console.print("\n[green]Demo complete![/green]\n")
    
    console.print(Panel.fit(
        f"[bold]Your Architecture:[/bold]\n"
        f"  Parameters: {gene.compute_params():,}\n"
        f"  VRAM: {est.vram_gb:.1f} GB\n"
        f"  Score: {result.score:.3f}\n\n"
        f"[bold]Try these:[/bold]\n"
        f"  python main.py cost gpt2           # Cost for GPT-2\n"
        f"  python main.py zoo-list           # Browse models\n"
        f"  python main.py recommend        # Get recommendations",
        title="Results",
        border_style="green"
    ))


@cli.group()
def measure():
    """DX measurement tools."""
    pass


@measure.command()
@click.option("--command", "-c", required=True, help="CLI command to measure")
def tthw(command):
    """Measure Time to Hello World."""
    start = time.perf_counter()
    result = subprocess.run(command, shell=True, capture_output=True)
    elapsed = time.perf_counter() - start
    
    record = {
        "command": command,
        "elapsed": elapsed,
        "timestamp": datetime.now().isoformat()
    }
    
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "tthw.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    
    tier = "CHAMPION" if elapsed < 60 else "COMPETITIVE" if elapsed < 300 else "NEEDS_WORK"
    status = "OK" if result.returncode == 0 else "FAIL"
    click.echo(f"{elapsed:.2f}s ({tier}) {status}")


@measure.command()
def benchmark():
    """Run full CLI benchmark."""
    commands = [
        ("python main.py version", "version"),
        ("python main.py evaluate", "evaluate"),
        ("python main.py verify", "verify"),
    ]
    
    results = []
    for cmd, name in commands:
        start = time.perf_counter()
        result = subprocess.run(cmd, shell=True, capture_output=True)
        elapsed = time.perf_counter() - start
        results.append({"cmd": name, "time": elapsed, "ok": result.returncode == 0})
        status = "OK" if result.returncode == 0 else "FAIL"
        click.echo(f"{name}: {elapsed:.3f}s {status}")
    
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    avg = sum(r["time"] for r in results) / len(results)
    click.echo(f"\nAverage: {avg:.3f}s")


@measure.command()
def feedback():
    """Submit feedback."""
    click.echo("Submit feedback at: https://github.com/your-repo/archgene/issues")
    try:
        import webbrowser
        webbrowser.open("https://github.com/your-repo/archgene/issues")
    except ImportError:
        pass  # webbrowser not available


@cli.command()
def version():
    """Show version information."""
    click.echo(f"ArchGene version {__version__}")
    click.echo("See CHANGELOG.md for release notes")


@cli.command()
@click.option("--vocab-dim", "-v", default=4096, help="Vocabulary dimension (e.g., 4096, 32000)")
@click.option("--hidden-dim", "-d", default=512, help="Hidden dimension (e.g., 256, 512, 1024)")
@click.option("--num-layers", "-l", default=4, help="Number of layers (e.g., 2, 4, 8, 12)")
@click.option("--num-heads", "-n", default=8, help="Number of attention heads (e.g., 4, 8, 16)")
@click.option("--intermediate-size", "-i", default=2048, help="FFN intermediate size (e.g., 1024, 2048, 4096)")
@click.option("--max-pos", "-m", default=2048, help="Max position embeddings (e.g., 512, 2048, 4096)")
@click.option("--hidden-act", "-a", default="gelu", type=click.Choice(["relu", "gelu", "silu", "sigmoid", "tanh"]), help="Hidden activation function: gelu (recommended), silu, relu")
@click.option("--save", "-s", is_flag=True, help="Save evaluation to history")
@click.option("--notes", "-t", default="", help="Notes for history (use with --save)")
def evaluate(vocab_dim, hidden_dim, num_layers, num_heads, intermediate_size, max_pos, hidden_act, save, notes):
    """Evaluate an LLM architecture.
    
    Examples:
        python main.py evaluate                              # default (vocab=4096, hidden=512)
        python main.py evaluate -d 256 -l 2                # small model
        python main.py evaluate -d 1024 -l 8 -n 16         # large model  
        python main.py evaluate -a silu --save -t "my exp" # save to history
    """
    tracker.print_info(f"Evaluating architecture...")
    
    gene = Gene(
        vocab_dim=vocab_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_pos,
        hidden_act=ActivationType(hidden_act)
    )
    
    tracker.print_gene_info(gene)
    
    result = evaluator.evaluate(gene)
    
    click.echo(f"\nScore: {result.score:.3f}")
    tracker.print_score(result.score)
    click.echo(f"Parameters: {result.params:,}")
    click.echo(f"Memory: {result.memory_bytes / 1e9:.2f} GB")
    
    if result.score == 0:
        click.echo("\n[yellow]Score is 0 — architecture failed validation.[/yellow]")
        if result.validation_errors:
            click.echo("Validation errors:")
            for err in result.validation_errors:
                click.echo(f"  - {err}")
        if result.verification_errors:
            click.echo("Verification errors:")
            for err in result.verification_errors:
                click.echo(f"  - {err}")
        click.echo("\nRun with --help to see valid parameter ranges")
    
    if save:
        record = EvaluationRecord(
            gene=gene,
            score=result.score,
            timestamp=datetime.now().isoformat(),
            notes=notes
        )
        history = EvaluationHistory()
        history.add(record)
        tracker.print_status(f"Saved to {history.path}")


@cli.command()
@click.argument("config_file")
def evaluate_file(config_file):
    """Evaluate architecture from config file."""
    with open(config_file) as f:
        config = json.load(f)
    
    gene = Gene.from_dict(config)
    result = evaluator.evaluate(gene)
    
    tracker.print_gene_info(gene)
    click.echo(f"\nScore: {result.score:.3f}")
    tracker.print_score(result.score)
    click.echo(f"Parameters: {result.params:,}")
    click.echo(f"Memory: {result.memory_bytes / 1e9:.2f} GB")


@cli.command()
@click.option("--notes", "-n", default="", help="Notes about this evaluation")
def save(notes):
    """Save the current evaluation to history."""
    gene = Gene()
    result = evaluator.evaluate(gene)
    
    record = EvaluationRecord(
        gene=gene,
        score=result.score,
        timestamp=datetime.now().isoformat(),
        notes=notes
    )
    
    history = EvaluationHistory()
    history.add(record)
    
    tracker.print_status(f"Saved evaluation to {history.path}")


@cli.command()
@click.option("--format", "-f", default="ascii", help="Visualization format (ascii, mermaid, json)")
def visualize(format):
    """Visualize the default architecture."""
    gene = Gene()
    viz = ArchitectureVisualizer()
    
    output = viz.visualize(gene, format)
    click.echo(output)


@cli.command()
@click.option("--format", "-f", default="pytorch", help="Export format (pytorch, onnx, config, huggingface)")
@click.option("--output", "-o", default=None, help="Output path")
def export(format, output):
    """Export architecture to various formats."""
    gene = Gene()
    exporter = Exporter()
    
    if format == "pytorch":
        path = exporter.to_pytorch(gene, output)
    elif format == "onnx":
        path = exporter.to_onnx(gene, output)
    elif format == "config":
        path = exporter.to_config(gene, output)
    elif format == "huggingface":
        path = exporter.to_huggingface(gene, output)
    else:
        click.echo(f"Unknown format: {format}")
        return
    
    click.echo(f"Exported to: {path}")


@cli.command()
@click.option("--top", "-t", default=10, help="Number of top architectures to show")
def history(top):
    """Show evaluation history."""
    hist = EvaluationHistory()
    records = hist.get_top(top)
    
    if not records:
        click.echo("No evaluation history found.")
        return
    
    tracker.print_table("Evaluation History", [
        {"Score": r.score, "Params": r.gene.compute_params(), "Timestamp": r.timestamp[:19]}
        for r in records
    ])


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed constraint violations")
def verify(verbose):
    """Verify the default architecture with Z3."""
    gene = Gene()
    
    click.echo("Running Z3 verification...")
    
    result = verifier.verify_all(gene)
    
    if result.is_valid:
        click.echo("[green]Verification PASSED[/green]")
    else:
        click.echo("[red]Verification FAILED[/red]")
    
    if result.constraints_checked:
        click.echo("\nConstraints checked:")
        for c in result.constraints_checked:
            click.echo(f"  - {c}")
    
    if result.errors:
        click.echo("\nErrors:")
        for error in result.errors:
            click.echo(f"  - {error}")
        
        if verbose:
            click.echo("\n[bold]Fix:[/bold] Ensure hidden_dim is divisible by num_heads (e.g., 512/8=64)")


@cli.command()
def bench():
    """Run quick benchmark of default architecture."""
    import torch
    
    gene = Gene()
    from src.archgene.exporter import LLMArchitecture
    
    model = LLMArchitecture(gene)
    
    batch_size = 1
    seq_len = min(128, gene.max_position_embeddings)
    input_ids = torch.randint(0, gene.vocab_dim, (batch_size, seq_len))
    
    click.echo(f"Running inference benchmark...")
    
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(10):
            logits = model(input_ids)
        end = time.perf_counter()
    
    elapsed = end - start
    per_iter = elapsed / 10
    
    click.echo(f"Time per iteration: {per_iter * 1000:.2f} ms")
    click.echo(f"Tokens/sec: {seq_len / per_iter:.0f}")


@cli.command()
@click.option("--generations", "-g", default=10, help="Number of generations")
@click.option("--population", "-p", default=10, help="Population size")
@click.option("--objectives", "-o", default="capability,efficiency,memory", help="Objectives to optimize")
def evolve(generations, population, objectives):
    """Evolve novel architectures using genetic algorithm.
    
    Examples:
        python main.py evolve                         # Quick evolution
        python main.py evolve -g 50 -p 20            # Full evolution
        python main.py evolve -o capability,efficiency  # Multi-objective
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from multi_objective import run_multi_objective_evolution, compute_multi_objective_fitness
    from src.archgene.gene_schema import Gene
    
    obj_list = objectives.split(",")
    
    console.print(Panel.fit(
        f"[bold cyan]Starting evolution...[/bold cyan]\n\n"
        f"  Generations: {generations}\n"
        f"  Population: {population}\n"
        f"  Objectives: {objectives}",
        title="Evolution Run",
        border_style="cyan"
    ))
    
    result = run_multi_objective_evolution(
        population_size=population,
        generations=generations,
        verbose=True
    )
    
    if result["best_gene"]:
        gene = Gene.from_dict(result["best_gene"])
        f = result["best_fitness"]
        
        console.print(Panel.fit(
            f"[bold green]Best Architecture Found![/bold green]\n\n"
            f"  Parameters: {gene.compute_params():,}\n"
            f"  Memory: {gene.compute_memory() / 1e6:.1f} MB\n\n"
            f"[bold]Fitness:[/bold]\n"
            f"  Capability: {f.capability:.2f}\n"
            f"  Efficiency: {f.efficiency:.2f}\n"
            f"  Stability: {f.stability:.2f}\n"
            f"  Memory Fit: {f.memory_fit:.2f}\n\n"
            f"[bold]Gene:[/bold]\n"
            f"  hidden_dim={gene.hidden_dim}, num_layers={gene.num_layers}\n"
            f"  num_heads={gene.num_heads}, intermediate_size={gene.intermediate_size}",
            title="Results",
            border_style="green"
        ))
    
    console.print("\n[dim]Run 'python main.py save' to record this evaluation.[/dim]")


@cli.command()
@click.option("--prompt", "-p", default="efficient architecture for long context", help="Research prompt")
@click.option("--seed", "-s", default=None, help="Seed architecture from zoo")
def research(prompt, seed):
    """Research novel architectures using enhanced LLM reasoning.
    
    The research engine now:
    1. Analyzes your task to extract requirements
    2. Selects the best architecture family (BitNet, Mamba, RWKV, Transformer)
    3. Designs task-specific parameters
    4. Provides evidence-based rationale
    
    Examples:
        python main.py research -p "efficient inference on mobile"
        python main.py research -p "long context 128k"
        python main.py research -p "low memory 1-bit"
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from research_engine import design_architecture_advanced
    from src.archgene.gene_schema import Gene
    from src.archgene.verifier import Verifier
    
    console.print(Panel.fit(
        f"[bold cyan]Researching:[/bold cyan] {prompt}\n\n"
        f"[dim]Analyzing task... selecting architecture... designing parameters...[/dim]",
        title="Enhanced Research Mode",
        border_style="cyan"
    ))
    
    seed_gene = None
    if seed:
        seed_gene = ModelZoo.get(seed)
    
    result = design_architecture_advanced(prompt, seed_gene)
    gene_dict = result["gene"]
    reasoning = result["reasoning"]
    requirements = result["requirements"]
    family = result["family"]
    
    gene = Gene.from_dict(gene_dict)
    verifier = Verifier()
    verification = verifier.verify_all(gene)
    
    console.print(Panel.fit(
        f"[bold cyan]Requirements:[/bold cyan]\n"
        f"  Use case: {requirements['use_case']}\n"
        f"  Priority: {requirements['priority']}\n"
        f"  Constraints: {', '.join(requirements['constraints']) if requirements['constraints'] else 'none'}\n\n"
        f"[bold cyan]Architecture:[/bold cyan] {family}\n"
        f"[dim]{reasoning}[/dim]",
        title="Research Analysis",
        border_style="cyan"
    ))
    
    console.print(Panel.fit(
        f"[bold green]Final Architecture:[/bold green]\n\n"
        f"  hidden_dim={gene.hidden_dim}, num_layers={gene.num_layers}\n"
        f"  num_heads={gene.num_heads}, intermediate_size={gene.intermediate_size}\n"
        f"  vocab_dim={gene.vocab_dim}, max_pos={gene.max_position_embeddings}\n"
        f"  Family: {gene.arch_family.value}\n"
        f"  Quantization: {gene.quant_type.value}\n\n"
        f"  Parameters: {gene.compute_params():,}\n"
        f"  Effective params: {gene.compute_effective_params():,}\n\n"
        f"[bold]Verification:[/bold] {'PASSED' if verification.is_valid else 'FAILED'}",
        title="Research Result",
        border_style="green" if verification.is_valid else "yellow"
    ))


@cli.command()
@click.option("--budget", "-b", default=100.0, help="Budget in GPU hours")
@click.option("--gpu", "-g", default="A100-40GB", help="GPU type")
@click.option("--use-case", "-u", default="research", help="Use case: research, inference, fine-tuning")
@click.option("--tokens", "-t", default=1_000_000_000, help="Training tokens")
def recommend(budget, gpu, use_case, tokens):
    """Recommend architectures based on budget and use case.
    
    Examples:
        python main.py recommend -b 50 -g A100-40GB -u research
        python main.py recommend --budget 100 --use-case inference
    """
    candidates = []
    
    for name in ModelZoo.list_all():
        gene = ModelZoo.get(name)
        est = CostEstimator.estimate_training(gene, gpu=gpu, training_tokens=tokens)
        
        if est.training_hours <= budget * 1.5:
            score = 1.0
            if use_case == "research" and gene.num_layers >= 20:
                score += 0.2
            if use_case == "inference" and gene.hidden_dim <= 2048:
                score += 0.2
            if use_case == "fine-tuning" and 3000 <= gene.hidden_dim <= 8000:
                score += 0.3
            
            candidates.append({
                "name": name,
                "score": score,
                "training_hours": est.training_hours,
                "vram_gb": est.vram_gb,
                "cost": est.training_hours * CostEstimator.GPU_COST_PER_HOUR.get(gpu, 2.0),
                "params": gene.compute_params(),
            })
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    console.print(Panel.fit(
        f"[bold cyan]Recommendations for {use_case} on {gpu}[/bold cyan]\n\n"
        f"Budget: {budget} hours | Tokens: {tokens:,}",
        title="Architecture Recommendations",
        border_style="cyan"
    ))
    
    for i, c in enumerate(candidates[:5], 1):
        console.print(Panel.fit(
            f"[bold]{i}. {c['name']}[/bold]\n"
            f"    Score: {c['score']:.1f} | Params: {c['params']:,} | VRAM: {c['vram_gb']:.1f} GB\n"
            f"    Training: {c['training_hours']:.0f} hours | Est. cost: ${c['cost']:.0f}",
            border_style="green" if i == 1 else "blue"
        ))
    
    if not candidates:
        console.print("[yellow]No architectures fit within budget. Try increasing budget.[/yellow]")


@cli.command()
def zoo_list():
    """List all pre-trained architectures in the Model Zoo."""
    tracker.print_table("Model Zoo", [
        {"Name": name, "Params": f"{ModelZoo.info(name)['parameters']:,}", "Memory": f"{ModelZoo.info(name)['memory_mb']:.1f}MB"}
        for name in ModelZoo.list_all()
    ])


@cli.command()
@click.argument("name")
def zoo_info(name):
    """Show details of a specific architecture."""
    try:
        info = ModelZoo.info(name)
        click.echo(f"\n{name}:")
        click.echo(f"  Parameters: {info['parameters']:,}")
        click.echo(f"  Memory: {info['memory_mb']:.1f}MB")
        gene = info["gene"]
        click.echo(f"  Vocab: {gene['vocab_dim']:,}")
        click.echo(f"  Hidden: {gene['hidden_dim']}")
        click.echo(f"  Layers: {gene['num_layers']}")
        click.echo(f"  Heads: {gene['num_heads']}")
        click.echo(f"  RoPE: {gene['use_rope']}")
    except ValueError as e:
        click.echo(f"[red]{e}[/red]")


@cli.command()
@click.argument("name")
def zoo_evaluate(name):
    """Evaluate an architecture from the Model Zoo."""
    gene = ModelZoo.get(name)
    tracker.print_gene_info(gene)
    result = evaluator.evaluate(gene)
    click.echo(f"\nScore: {result.score:.3f}")
    tracker.print_score(result.score)


@cli.command()
@click.argument("name")
@click.option("--gpu", default="A100-40GB", help="GPU type")
@click.option("--batch-size", default=1, help="Batch size")
def cost(name, gpu, batch_size):
    """Estimate costs for an architecture."""
    from src.archgene.gene_schema import QuantizationType
    
    gene = ModelZoo.get(name)
    est = CostEstimator.full_estimate(gene, gpu=gpu, batch_size=batch_size)

    click.echo(f"\n{name}:")
    click.echo(f"  Family: {gene.arch_family.value}")
    click.echo(f"  Parameters: {gene.compute_params():,}")
    
    if gene.quant_type != QuantizationType.NONE:
        click.echo(f"  Effective params: {gene.compute_effective_params():,} ({gene.quant_type.value})")
        click.echo(f"  VRAM bits: {gene.compute_vram_bits():,} bits")
    
    click.echo(f"  VRAM required: {est.vram_gb:.1f} GB")
    click.echo(f"  Inference cost: ${est.inference_cost_per_1m_tokens:.2f}/M tokens")
    click.echo(f"  Inference latency: {est.inference_latency_ms:.2f} ms/token")
    click.echo(f"  Training (1T tokens): {est.training_hours:.0f} GPU hours")
    click.echo(f"  Training cost: ${est.training_cost_per_1k_tokens:.2f}/1K tokens")


@cli.command()
@click.argument("name")
def benchmark(name):
    """Estimate benchmark scores for an architecture."""
    gene = ModelZoo.get(name)
    result = BenchmarkIntegration.full_benchmark(gene)

    click.echo(f"\n{name}:")
    click.echo(f"  Parameters: {result['parameters']:,}")
    click.echo(f"  MMLU estimate: {result['estimates']['mmlu']}")
    click.echo(f"  HumanEval estimate: {result['estimates']['humaneval']}")
    click.echo(f"  Inference speed: {result['metrics']['inference_speed']} tok/s")


@cli.command()
@click.argument("name")
@click.option("--platform", type=click.Choice(["huggingface", "vllm", "replicate"]), default="huggingface")
@click.option("--repo-id", help="Repo ID for HuggingFace")
def deploy(name, platform, repo_id):
    """Get deployment instructions for an architecture."""
    gene = ModelZoo.get(name)

    click.echo(f"\nDeploying {name} to {platform}:")
    instructions = ModelDeployment.get_deployment_instructions(platform)
    click.echo(f"\n{instructions}")


class ArchGeneError(Exception):
    """Base exception for ArchGene errors."""
    pass


class ModelNotFoundError(ArchGeneError):
    """Raised when a model is not found in the zoo."""
    pass


class VerificationError(ArchGeneError):
    """Raised when architecture verification fails."""
    pass


def handle_error(error):
    """Handle errors with helpful messages."""
    from rich.traceback import Traceback
    
    error_type = type(error).__name__
    
    if isinstance(error, ModelNotFoundError):
        console.print(Panel.fit(
            f"[bold red]Model Not Found[/bold red]\n\n{error}\n\n"
            "[bold]Available models:[/bold]\n"
            "  gpt2, llama2_7b, llama2_13b, mistral_7b,\n"
            "  qwen2_7b, phi3_3b, tinylamma, gemma2_2b\n\n"
            "Run [cyan]python main.py zoo-list[/cyan] to see all models",
            title="Error",
            border_style="red"
        ))
    elif isinstance(error, VerificationError):
        console.print(Panel.fit(
            f"[bold red]Verification Failed[/bold red]\n\n{error}\n\n"
            "[bold]Common fixes:[/bold]\n"
            "  • Check hidden_size is divisible by num_heads * head_dim\n"
            "  • Ensure intermediate_size >= hidden_size\n"
            "  • max_position_embeddings should be >= 512\n\n"
            "Run [cyan]python main.py evaluate --help[/cyan] for valid ranges",
            title="Error",
            border_style="red"
        ))
    elif isinstance(error, FileNotFoundError):
        console.print(Panel.fit(
            f"[bold red]File Not Found[/bold red]\n\n{error}\n\n"
            "[bold]Try:[/bold]\n"
            "  python main.py demo          # Run interactive demo\n"
            "  python main.py zoo-list     # List available models",
            title="Error",
            border_style="red"
        ))
    elif isinstance(error, ImportError):
        console.print(Panel.fit(
            f"[bold red]Missing Dependency[/bold red]\n\n{error}\n\n"
            "[bold]Install:[/bold]\n"
            "  pip install -e .",
            title="Error",
            border_style="red"
        ))
    elif isinstance(error, Exception):
        console.print(Panel.fit(
            f"[bold red]{error_type}[/bold red]\n\n{error}\n\n"
            "[bold]Need help?[/bold]\n"
            "  python main.py demo     # Run interactive demo\n"
            "  python main.py --help  # See all commands",
            title="Error",
            border_style="red"
        ))
    
    sys.exit(1)


@cli.command()
@click.option("--non-interactive", is_flag=True, help="Exit immediately for CI/CD")
@click.option("--no-save", is_flag=True, help="Don't save to history")
def interactive(non_interactive, no_save):
    """Interactive guide to ArchGene.
    
    Full guided experience:
        python main.py interactive
    
    Quick exit for CI/CD:
        python main.py interactive --non-interactive
    
    Alias: python main.py guide
    """
    if non_interactive:
        console.print("[dim]Interactive mode disabled. Use other commands.[/dim]")
        return
    
    run_guide(save_to_history=not no_save)


# Register alias
cli.add_command(interactive, name="guide")


if __name__ == "__main__":
    try:
        cli()
    except ModelNotFoundError as e:
        handle_error(e)
    except VerificationError as e:
        handle_error(e)
    except FileNotFoundError as e:
        handle_error(e)
    except ImportError as e:
        handle_error(e)
    except Exception as e:
        handle_error(e)