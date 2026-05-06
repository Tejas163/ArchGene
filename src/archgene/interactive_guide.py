"""Interactive Guide for ArchGene - Educational walkthroughs."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from .gene_schema import Gene, ActivationType
from .evaluation import Evaluator
from .cost_estimator import CostEstimator
from .model_zoo import ModelZoo
from .verifier import Verifier
from .evaluation import EvaluationRecord, EvaluationHistory

console = Console()
evaluator = Evaluator()
verifier = Verifier()


def print_welcome():
    """Welcome banner."""
    console.print(Panel.fit(
        "[bold cyan]ArchGene Interactive Guide[/bold cyan]\n\n"
        "Learn how to evaluate, choose, and research LLM architectures.\n\n"
        "[bold]Why this guide?[/bold]\n"
        "  • Understand tradeoffs before spending GPU hours\n"
        "  • Find the right architecture for your budget\n"
        "  • Learn about different architecture families\n\n"
        "[dim]Press Ctrl+C to exit at any time[/dim]",
        title="Welcome",
        border_style="cyan"
    ))


def print_menu():
    """Print main menu."""
    console.print("\n[bold]What would you like to do?[/bold]\n")
    
    table = Table(show_header=False, box=None)
    table.add_column("Option", style="bold cyan", width=4)
    table.add_column("Description", style="white")
    
    options = [
        ("1", "Evaluate an architecture"),
        ("2", "Estimate costs for a model"),
        ("3", "Find the right architecture for my budget"),
        ("4", "Research a new architecture family"),
        ("5", "Compare pre-trained models"),
        ("6", "Get deployment instructions"),
        ("7", "Learn about architecture families"),
    ]
    
    for num, desc in options:
        table.add_row(num, desc)
    
    console.print(table)
    console.print()


def get_choice(max_option=7):
    """Get user choice with validation."""
    while True:
        choice = Prompt.ask(
            "[bold]Enter option[/bold] (or press Enter for default)",
            default="1",
            show_choices=False
        )
        if choice == "":
            return "1"
        if choice.isdigit() and 1 <= int(choice) <= max_option:
            return choice
        console.print(f"[red]Please enter a number between 1 and {max_option}[/red]")


def learn_more(topic):
    """Show learn more content."""
    content = {
        "vocab": (
            "[bold]Vocabulary Size (vocab_dim)[/bold]\n\n"
            "The number of unique tokens the model can recognize.\n\n"
            "[bold]Tradeoffs:[/bold]\n"
            "  • Larger vocab = more expressiveness\n"
            "  • Larger vocab = more parameters = more memory\n\n"
            "[bold]Typical values:[/bold]\n"
            "  • 4096: Compact models, languages with small alphabets\n"
            "  • 32000: Standard for English-language models\n"
            "  • 50000+: Multilingual models\n\n"
            "[bold]When to increase:[/bold]\n"
            "  • Multilingual tasks (>100 languages)\n"
            "  • Code generation (many keywords)\n"
            "  • Scientific domains (special symbols)"
        ),
        "hidden": (
            "[bold]Hidden Dimension (hidden_dim)[/bold]\n\n"
            "The width of each hidden layer - controls model capacity.\n\n"
            "[bold]Tradeoffs:[/bold]\n"
            "  • Larger = more parameters = more capability\n"
            "  • Larger = more VRAM required\n\n"
            "[bold]Memory formula:[/bold]\n"
            "  ~ hidden_dim² × 4 bytes per parameter\n\n"
            "[bold]Typical values:[/bold]\n"
            "  • 256-512: Simple tasks, 4GB GPU\n"
            "  • 768-1024: General purpose, 8GB GPU\n"
            "  • 2048-4096: Research models, 24GB+ GPU"
        ),
        "layers": (
            "[bold]Number of Layers (num_layers)[/bold]\n\n"
            "How many transformer blocks stacked - controls depth.\n\n"
            "[bold]Tradeoffs:[/bold]\n"
            "  • More layers = more complex reasoning\n"
            "  • More layers = slower inference\n"
            "  • Diminishing returns > 32 layers\n\n"
            "[bold]Rules of thumb:[/bold]\n"
            "  • 2-4: Simple classification\n"
            "  • 8-12: General language tasks\n"
            "  • 32+: Research-grade models"
        ),
        "heads": (
            "[bold]Attention Heads (num_heads)[/bold]\n\n"
            "Parallel attention mechanisms - each learns different patterns.\n\n"
            "[bold]Why multiple heads?[/bold]\n"
            "  • Some heads learn syntax\n"
            "  • Some heads learn semantics\n"
            "  • Some heads learn relations\n\n"
            "[bold]Constraint:[/bold]\n"
            "  hidden_dim must be divisible by num_heads\n"
            "  (e.g., 512 ÷ 8 = 64 per head)"
        ),
        "cost": (
            "[bold]Why estimate costs?[/bold]\n\n"
            "GPU time is expensive and limited. Before training:\n\n"
            "  1. Know VRAM requirements\n"
            "  2. Estimate training time\n"
            "  3. Predict inference cost\n\n"
            "[bold]Real example:[/bold]\n"
            "  LLaMA 2-7B on 1T tokens:\n"
            "  • ~56 GPU hours on A100\n"
            "  • ~$112 per 1K tokens (cloud)\n"
            "  • 14GB VRAM needed\n\n"
            "[bold]Saving tip:[/bold]\n"
            "  Use quantization to reduce VRAM by 4-8x"
        ),
        "family": (
            "[bold]Architecture Families[/bold]\n\n"
            "[bold]Transformer[/bold] - Standard, proven\n"
            "  • Pros: Best benchmark scores, well-understood\n"
            "  • Cons: O(n²) attention, slow for long context\n\n"
            "[bold]BitNet[/bold] - 1-bit quantization\n"
            "  • Pros: 4-8x memory reduction\n"
            "  • Cons: Newer, less tooling\n\n"
            "[bold]Mamba/RWKV[/bold] - State space models\n"
            "  • Pros: O(n) inference, linear scaling\n"
            "  • Cons: Emerging research\n\n"
            "[bold]Linear/Attention-free[/bold] - Newest\n"
            "  • Pros: Fastest inference\n"
            "  • Cons: Early research stage"
        ),
    }
    
    if topic in content:
        console.print(Panel.fit(
            content[topic],
            title=f"Learn More: {topic.title()}",
            border_style="cyan"
        ))


def path_evaluate():
    """Educational path: Evaluate architecture."""
    console.print(Panel.fit(
        "[bold cyan]Path 1: Evaluate Architecture[/bold cyan]\n\n"
        "Learn how to evaluate an LLM architecture using formal verification.\n\n"
        "[bold]Why evaluate?[/bold]\n"
        "  • Catch architectural issues before training\n"
        "  • Understand memory requirements\n"
        "  • Get a quality score (0-1)\n\n"
        "[bold]We'll evaluate:[/bold]\n"
        "  • Parameter count\n"
        "  • Memory footprint\n"
        "  • Structural validity",
        title="Evaluate Architecture",
        border_style="cyan"
    ))
    
    console.print("\n[bold]Step 1: Choose parameters[/bold]")
    console.print("[dim]Press Enter for recommended defaults[/dim]\n")
    
    vocab = Prompt.ask(
        "[bold]Vocabulary size[/bold] (typical: 4096, 32000)",
        default="4096"
    )
    hidden = Prompt.ask(
        "[bold]Hidden dimension[/bold] (typical: 512, 768, 1024)",
        default="512"
    )
    layers = Prompt.ask(
        "[bold]Number of layers[/bold] (typical: 4, 8, 12)",
        default="4"
    )
    heads = Prompt.ask(
        "[bold]Number of heads[/bold] (typical: 4, 8, 16)",
        default="8"
    )
    intermediate = Prompt.ask(
        "[bold]FFN intermediate size[/bold] (typical: hidden × 4)",
        default="2048"
    )
    
    learn = Prompt.ask(
        "\n[bold]Learn more about these parameters?[/bold]",
        choices=["y", "n"],
        default="n"
    )
    if learn.lower() == "y":
        learn_more("vocab")
        learn_more("hidden")
        learn_more("layers")
        learn_more("heads")
    
    console.print("\n[bold green]Creating architecture...[/bold green]")
    
    gene = Gene(
        vocab_dim=int(vocab),
        hidden_dim=int(hidden),
        num_layers=int(layers),
        num_heads=int(heads),
        intermediate_size=int(intermediate),
    )
    
    console.print(f"\n[bold]Created Gene:[/bold]")
    console.print(f"  vocab_dim: {gene.vocab_dim}")
    console.print(f"  hidden_dim: {gene.hidden_dim}")
    console.print(f"  num_layers: {gene.num_layers}")
    console.print(f"  num_heads: {gene.num_heads}")
    console.print(f"  Parameters: {gene.compute_params():,}")
    
    console.print("\n[bold]Step 2: Verify with Z3[/bold]")
    result = verifier.verify_all(gene)
    
    if result.is_valid:
        console.print("[green]✓ Verification PASSED[/green]")
    else:
        console.print("[red]✗ Verification FAILED[/red]")
        if result.errors:
            console.print("[bold]Errors:[/bold]")
            for err in result.errors:
                console.print(f"  - {err}")
    
    console.print("\n[bold]Step 3: Evaluate score[/bold]")
    eval_result = evaluator.evaluate(gene)
    
    console.print(f"\n[bold green]Score: {eval_result.score:.3f}[/bold green]")
    console.print(f"  Parameters: {eval_result.params:,}")
    console.print(f"  Memory: {eval_result.memory_bytes / 1e9:.2f} GB")
    
    if eval_result.score == 0:
        console.print("\n[yellow]Note: Score 0 means verification failed.[/yellow]")
        console.print("[dim]Fix the errors and try again.[/dim]")
        return None
    
    return gene


def path_costs():
    """Educational path: Estimate costs."""
    console.print(Panel.fit(
        "[bold cyan]Path 2: Estimate Costs[/bold cyan]\n\n"
        "Learn how to estimate GPU costs before training.\n\n"
        "[bold]Why estimate?[/bold]\n"
        "  • Avoid running out of VRAM mid-training\n"
        "  • Budget accurately for cloud GPU time\n"
        "  • Compare efficiency across models",
        title="Estimate Costs",
        border_style="cyan"
    ))
    
    learn = Prompt.ask(
        "[bold]Learn why cost estimation matters?[/bold]",
        choices=["y", "n"],
        default="n"
    )
    if learn.lower() == "y":
        learn_more("cost")
    
    console.print("\n[bold]Step 1: Select a model[/bold]")
    
    models = ModelZoo.list_all()[:10]
    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan", width=4)
    table.add_column("Model", style="white")
    
    for i, name in enumerate(models, 1):
        info = ModelZoo.info(name)
        table.add_row(str(i), f"{name} ({info['parameters']:,} params)")
    
    console.print(table)
    
    choice = Prompt.ask(
        "\n[bold]Select model[/bold]",
        default="1"
    )
    
    try:
        idx = int(choice) - 1
        model_name = models[idx]
    except (ValueError, IndexError):
        console.print("[red]Invalid selection, using gpt2[/red]")
        model_name = "gpt2"
    
    gene = ModelZoo.get(model_name)
    
    console.print("\n[bold]Step 2: Choose GPU[/bold]")
    gpu_options = ["A100-40GB", "A100-80GB", "H100", "RTX 4090", "A10"]
    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan", width=4)
    table.add_column("GPU", style="white")
    table.add_column("VRAM", style="dim")
    
    vram_map = {
        "A100-40GB": "40GB",
        "A100-80GB": "80GB", 
        "H100": "80GB",
        "RTX 4090": "24GB",
        "A10": "24GB"
    }
    
    for i, gpu in enumerate(gpu_options, 1):
        table.add_row(str(i), gpu, vram_map[gpu])
    
    console.print(table)
    
    gpu_choice = Prompt.ask(
        "\n[bold]Select GPU[/bold]",
        default="1"
    )
    
    try:
        idx = int(gpu_choice) - 1
        gpu = gpu_options[idx]
    except (ValueError, IndexError):
        gpu = "A100-40GB"
    
    est = CostEstimator.full_estimate(gene, gpu=gpu)
    
    console.print(Panel.fit(
        f"[bold]Cost Estimate: {model_name}[/bold]\n\n"
        f"  GPU: {gpu}\n"
        f"  Parameters: {gene.compute_params():,}\n"
        f"  VRAM required: {est.vram_gb:.1f} GB\n"
        f"  Inference: ${est.inference_cost_per_1m_tokens:.2f}/M tokens\n"
        f"  Training (1T tokens): {est.training_hours:.0f} hours\n"
        f"  Training cost: ${est.training_cost_per_1k_tokens:.2f}/1K tokens",
        title="Results",
        border_style="green"
    ))
    
    return None


def path_recommend():
    """Educational path: Get recommendations."""
    console.print(Panel.fit(
        "[bold cyan]Path 3: Find Right Architecture[/bold cyan]\n\n"
        "Get AI-powered architecture recommendations based on your constraints.\n\n"
        "[bold]We'll ask:[/bold]\n"
        "  • Your budget (GPU hours)\n"
        "  • Your use case (research, inference, fine-tuning)\n"
        "  • Your GPU type\n\n"
        "[bold]You'll get:[/bold]\n"
        "  • Top 5 architectures ranked by fit\n"
        "  • Detailed cost breakdowns",
        title="Find Architecture",
        border_style="cyan"
    ))
    
    budget = Prompt.ask(
        "[bold]Budget (GPU hours)[/bold]",
        default="50"
    )
    
    use_case = Prompt.ask(
        "[bold]Use case[/bold]",
        choices=["research", "inference", "fine-tuning"],
        default="research"
    )
    
    gpu = Prompt.ask(
        "[bold]GPU type[/bold]",
        default="A100-40GB"
    )
    
    console.print(f"\n[bold green]Finding best architectures...[/bold green]\n")
    
    candidates = []
    for name in ModelZoo.list_all():
        gene = ModelZoo.get(name)
        est = CostEstimator.estimate_training(gene, gpu=gpu)
        
        if est.training_hours <= float(budget) * 1.5:
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
                "params": gene.compute_params(),
            })
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    console.print(Panel.fit(
        f"[bold cyan]Top Recommendations[/bold cyan]\n\n"
        f"Budget: {budget} hours | GPU: {gpu} | Use: {use_case}",
        border_style="cyan"
    ))
    
    for i, c in enumerate(candidates[:5], 1):
        console.print(Panel.fit(
            f"[bold]{i}. {c['name']}[/bold]\n"
            f"    Score: {c['score']:.1f} | Params: {c['params']:,}\n"
            f"    VRAM: {c['vram_gb']:.1f} GB | Training: {c['training_hours']:.0f}h",
            border_style="green" if i == 1 else "blue"
        ))
    
    if not candidates:
        console.print("[yellow]No architectures fit within budget. Try increasing budget.[/yellow]")
    
    return None


def path_research():
    """Educational path: Research architecture."""
    console.print(Panel.fit(
        "[bold cyan]Path 4: Research Architecture[/bold cyan]\n\n"
        "Research novel architectures using AI-powered analysis.\n\n"
        "[bold]The research engine:[/bold]\n"
        "  1. Analyzes your task requirements\n"
        "  2. Selects best architecture family\n"
        "  3. Designs task-specific parameters\n"
        "  4. Provides evidence-based rationale",
        title="Research Architecture",
        border_style="cyan"
    ))
    
    learn = Prompt.ask(
        "[bold]Learn about architecture families?[/bold]",
        choices=["y", "n"],
        default="n"
    )
    if learn.lower() == "y":
        learn_more("family")
    
    prompt = Prompt.ask(
        "\n[bold]Describe your task[/bold]\n"
        "[dim]e.g., 'efficient inference on mobile device'[/dim]",
        default="efficient architecture for long context"
    )
    
    console.print(f"\n[bold green]Analyzing: {prompt}[/bold green]")
    console.print("[dim]This may take a moment...[/dim]\n")
    
    from research_engine import design_architecture_advanced
    
    result = design_architecture_advanced(prompt)
    gene_dict = result["gene"]
    reasoning = result["reasoning"]
    requirements = result["requirements"]
    family = result["family"]
    
    gene = Gene.from_dict(gene_dict)
    verification = verifier.verify_all(gene)
    
    console.print(Panel.fit(
        f"[bold cyan]Requirements:[/bold cyan]\n"
        f"  Use case: {requirements['use_case']}\n"
        f"  Priority: {requirements['priority']}\n\n"
        f"[bold cyan]Architecture Family:[/bold cyan] {family}\n\n"
        f"[dim]{reasoning}[/dim]",
        title="Research Results",
        border_style="cyan"
    ))
    
    console.print(Panel.fit(
        f"[bold green]Recommended Architecture[/bold green]\n\n"
        f"  hidden_dim: {gene.hidden_dim}\n"
        f"  num_layers: {gene.num_layers}\n"
        f"  num_heads: {gene.num_heads}\n"
        f"  Family: {gene.arch_family.value}\n"
        f"  Quantization: {gene.quant_type.value}\n\n"
        f"  Parameters: {gene.compute_params():,}\n\n"
        f"  Verification: {'PASSED' if verification.is_valid else 'FAILED'}",
        title="Results",
        border_style="green" if verification.is_valid else "yellow"
    ))
    
    return gene


def path_compare():
    """Educational path: Compare models."""
    console.print(Panel.fit(
        "[bold cyan]Path 5: Compare Models[/bold cyan]\n\n"
        "Compare pre-trained models side-by-side.\n\n"
        "[bold]We'll compare:[/bold]\n"
        "  • Parameter count\n"
        "  • Memory requirements\n"
        "  • Architecture family\n"
        "  • Use cases",
        title="Compare Models",
        border_style="cyan"
    ))
    
    models = ModelZoo.list_all()
    
    console.print("\n[bold]Available models:[/bold]")
    for i, name in enumerate(models, 1):
        console.print(f"  {i}. {name}")
    
    console.print("\n[bold]Select 2 models to compare (e.g., '1 3'):[/bold]")
    selection = Prompt.ask(
        "[bold]Enter numbers[/bold]",
        default="1 2"
    )
    
    try:
        parts = selection.split()
        idx1, idx2 = int(parts[0]) - 1, int(parts[1]) - 1
        name1, name2 = models[idx1], models[idx2]
    except:
        console.print("[red]Invalid selection, using gpt2 and gpt2-medium[/red]")
        name1, name2 = "gpt2", "gpt2-medium"
    
    gene1 = ModelZoo.get(name1)
    gene2 = ModelZoo.get(name2)
    
    console.print(Panel.fit(
        f"[bold]Comparison[/bold]\n\n"
        f"[bold]{name1}[/bold] vs [bold]{name2}[/bold]\n\n"
        f"Parameters:      {gene1.compute_params():>10,}  {gene2.compute_params():>10,}\n"
        f"Memory (MB):  {gene1.compute_memory()/1e6:>10.1f}  {gene2.compute_memory()/1e6:>10.1f}\n"
        f"Layers:       {gene1.num_layers:>10}  {gene2.num_layers:>10}\n"
        f"Hidden:      {gene1.hidden_dim:>10}  {gene2.hidden_dim:>10}\n"
        f"Family:      {gene1.arch_family.value:>10}  {gene2.arch_family.value:>10}",
        title="Comparison",
        border_style="cyan"
    ))
    
    return None


def path_deploy():
    """Educational path: Deployment instructions."""
    console.print(Panel.fit(
        "[bold cyan]Path 6: Deployment Instructions[/bold cyan]\n\n"
        "Get platform-specific deployment instructions.\n\n"
        "[bold]Supported platforms:[/bold]\n"
        "  • HuggingFace\n"
        "  • vLLM\n"
        "  • Replicate",
        title="Deployment",
        border_style="cyan"
    ))
    
    models = ModelZoo.list_all()[:5]
    for i, name in enumerate(models, 1):
        console.print(f"  {i}. {name}")
    
    choice = Prompt.ask(
        "\n[bold]Select model[/bold]",
        default="1"
    )
    
    try:
        model_name = models[int(choice) - 1]
    except:
        model_name = "gpt2"
    
    platforms = ["huggingface", "vllm", "replicate"]
    for i, p in enumerate(platforms, 1):
        console.print(f"  {i}. {p}")
    
    plat_choice = Prompt.ask(
        "\n[bold]Select platform[/bold]",
        default="1"
    )
    
    try:
        platform = platforms[int(plat_choice) - 1]
    except:
        platform = "huggingface"
    
    gene = ModelZoo.get(model_name)
    
    from .deployment import ModelDeployment
    instructions = ModelDeployment.get_deployment_instructions(platform)
    
    console.print(Panel.fit(
        f"[bold]Deploy {model_name} to {platform}[/bold]\n\n{instructions}",
        title="Deployment",
        border_style="cyan"
    ))
    
    return None


def path_learn_families():
    """Educational path: Learn architecture families."""
    learn_more("family")
    
    console.print("\n[bold]Architecture Family Comparison[/bold]\n")
    
    table = Table()
    table.add_column("Family", style="bold")
    table.add_column("Attention", style="cyan")
    table.add_column("Pros", style="green")
    table.add_column("Cons", style="red")
    table.add_column("Use Case")
    
    table.add_row(
        "Transformer",
        "O(n²)",
        "Best benchmarks",
        "Slow for long ctx",
        "General purpose"
    )
    table.add_row(
        "BitNet",
        "O(n²)",
        "4-8x smaller",
        "Newer",
        "Mobile/edge"
    )
    table.add_row(
        "Mamba",
        "O(n)",
        "Linear scaling",
        "Emerging",
        "Long context"
    )
    table.add_row(
        "RWKV",
        "O(n)",
        "Fast inference",
        "Less tuning",
        "Realtime"
    )
    
    console.print(table)
    
    return None


def run_guide(save_to_history=True):
    """Run the interactive guide."""
    print_welcome()
    
    while True:
        print_menu()
        choice = get_choice()
        
        gene = None
        
        if choice == "1":
            gene = path_evaluate()
        elif choice == "2":
            path_costs()
        elif choice == "3":
            path_recommend()
        elif choice == "4":
            gene = path_research()
        elif choice == "5":
            path_compare()
        elif choice == "6":
            path_deploy()
        elif choice == "7":
            path_learn_families()
        
        if gene and save_to_history:
            console.print("\n[bold]Save to history?[/bold]")
            do_save = Prompt.ask(
                "[bold]Save evaluation[/bold]",
                choices=["y", "n"],
                default="y"
            )
            if do_save.lower() == "y":
                notes = Prompt.ask(
                    "[bold]Add notes[/bold] (optional)",
                    default=""
                )
                result = evaluator.evaluate(gene)
                record = EvaluationRecord(
                    gene=gene,
                    score=result.score,
                    timestamp=__import__("datetime").datetime.now().isoformat(),
                    notes=notes
                )
                history = EvaluationHistory()
                history.add(record)
                console.print(f"[green]✓ Saved to {history.path}[/green]")
        
        console.print("\n[bold]Continue guide?[/bold]")
        cont = Prompt.ask(
            "[bold]Continue[/bold]",
            choices=["y", "n"],
            default="y"
        )
        if cont.lower() != "y":
            console.print("\n[bold cyan]Thanks for using ArchGene Guide![/bold cyan]\n")
            break


@click.command(name="guide")
@click.option("--non-interactive", is_flag=True, help="Exit immediately for CI/CD")
@click.option("--no-save", is_flag=True, help="Don't save to history")
def guide(non_interactive, no_save):
    """Interactive guide to ArchGene.
    
    Run without flags for full guided experience:
        python main.py guide
    
    Quick exit for CI/CD:
        python main.py guide --non-interactive
    
    Examples:
        python main.py guide
        python main.py --interactive
    """
    if non_interactive:
        console.print("[dim]Use --non-interactive for CI/CD mode[/dim]")
        return
    
    run_guide(save_to_history=not no_save)


# Alias command for --interactive flag compatibility
@click.command(name="interactive", cls=click.Command)
def interactive():
    """Alias for 'guide' command."""
    ctx = click.get_current_context()
    ctx.invoke(guide)


if __name__ == "__main__":
    guide()