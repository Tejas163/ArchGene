import click
from rich.console import Console
from datetime import datetime
from pathlib import Path
import json
import time
import subprocess

__version__ = "0.1.0"

from core.gene_schema import Gene, ActivationType
from core.verifier import Verifier
from core.evaluation import Evaluator, EvaluationRecord, EvaluationHistory
from core.progress import ProgressTracker
from core.visualization import ArchitectureVisualizer
from core.exporter import Exporter


console = Console()
tracker = ProgressTracker()
verifier = Verifier()
evaluator = Evaluator()

DATA_DIR = Path("~/.archgene").expanduser()


@click.group()
def cli():
    """ArchGene: Evaluate LLM architectures using formal verification."""
    pass


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
    from core.exporter import LLMArchitecture
    
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


if __name__ == "__main__":
    cli()