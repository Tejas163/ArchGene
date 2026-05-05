"""
ArchGene Training - Simplified training module.

Usage:
    python -m training train
    python -m training evolve
"""

import click
from rich.console import Console

console = Console()


@click.group()
def cli():
    """ArchGene Training submodule."""
    pass


@cli.command("train")
@click.option("--depth", "-d", default=8, help="Number of layers")
def train_cmd(depth):
    """Quick train a model."""
    import torch
    
    vocab_size = 32768
    aspect_ratio = 64
    head_dim = 128
    seq_len = 2048
    
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    
    model = torch.nn.Embedding(vocab_size, model_dim)
    num_params = sum(p.numel() for p in model.parameters())
    
    console.print(f"[cyan]Model:[/cyan]")
    console.print(f"  Layers: {depth}")
    console.print(f"  Hidden: {model_dim}")
    console.print(f"  Parameters: {num_params:,}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[green]Device: {device}[/green]")


@cli.command("evolve")
@click.option("--generations", "-g", default=10, help="Generations")
def evolve_cmd(generations):
    """Run autonomous evolution."""
    console.print(f"[cyan]Evolution - {generations} generations[/cyan]")
    console.print("[dim]Use 'python main.py evolve' instead.[/dim]")


if __name__ == "__main__":
    cli()