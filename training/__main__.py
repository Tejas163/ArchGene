"""Training module CLI entry point."""

import click


@click.group()
def cli():
    """ArchGene Training - Autonomous evolution and training."""
    pass


@cli.command("train")
@click.option("--depth", "-d", default=8, help="Number of layers")
def train(depth):
    """Quick train a model."""
    import torch
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    
    vocab_size = 32768
    aspect_ratio = 64
    head_dim = 128
    
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    
    model = torch.nn.Embedding(vocab_size, model_dim)
    num_params = sum(p.numel() for p in model.parameters())
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(Panel.fit(
        f"[bold cyan]ArchGene Training[/bold cyan]\n\n"
        f"  Depth: {depth}\n"
        f"  Hidden: {model_dim}\n"
        f"  Heads: {num_heads}\n"
        f"  Parameters: {num_params:,}\n"
        f"  Device: {device}",
        title="Model",
        border_style="cyan"
    ))


@cli.command("evolve")
@click.option("--generations", "-g", default=10, help="Generations")
def evolve(generations):
    """Run autonomous evolution."""
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    
    console.print(Panel.fit(
        f"[bold cyan]AutoEvolution[/bold cyan]\n\n"
        f"  Generations: {generations}",
        title="Evolution",
        border_style="cyan"
    ))
    
    console.print("[dim]Use 'python main.py evolve' instead for full evolution.[/dim]")


if __name__ == "__main__":
    cli()