"""Design Session — conversational Q&A that designs architectures from user requirements."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .gene_schema import Gene, ArchitectureFamily, QuantizationType
from .verifier import Verifier
from .cost_estimator import CostEstimator
from .evaluation import Evaluator
from .research_engine import TaskRequirements, ArchitectureSelector, design_for_requirements, generate_reasoning

console = Console()
DATA_DIR = Path.home() / ".archgene"


@dataclass
class DesignAnswers:
    use_case: str = "research"
    budget_tier: str = "mid"
    constraints: list[str] = field(default_factory=list)
    arch_family: str = "any"
    target_params: str = "1b-7b"
    context_length: int = 8192


@dataclass
class GenerationRecord:
    timestamp: str = ""
    use_case: str = ""
    hidden_dim: int = 0
    num_layers: int = 0
    num_heads: int = 0
    params: int = 0
    verified: bool = False
    vram_gb: float = 0.0


class UsageTracker:
    def __init__(self):
        self.path = DATA_DIR / "generations.json"
        self._ensure_dir()
        self.records = self._load()

    def _ensure_dir(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return json.load(f)

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

    def count_this_month(self) -> int:
        now = datetime.now()
        count = 0
        for r in self.records:
            ts = datetime.fromisoformat(r["timestamp"])
            if ts.year == now.year and ts.month == now.month:
                count += 1
        return count

    def add(self, record: GenerationRecord):
        self.records.append({
            "timestamp": record.timestamp,
            "use_case": record.use_case,
            "hidden_dim": record.hidden_dim,
            "num_layers": record.num_layers,
            "num_heads": record.num_heads,
            "params": record.params,
            "verified": record.verified,
            "vram_gb": record.vram_gb,
        })
        self._save()

    def total(self) -> int:
        return len(self.records)


class DesignSession:
    def __init__(self):
        self.answers = DesignAnswers()
        self.verifier = Verifier()
        self.evaluator = Evaluator()
        self.cost_estimator = CostEstimator()
        self.tracker = UsageTracker()

    def run(self) -> Optional[Gene]:
        self._print_welcome()
        self._show_usage()

        self._ask_use_case()
        self._ask_budget()
        self._ask_constraints()
        self._ask_arch_family()
        self._ask_target_params()
        self._ask_context_length()

        console.print("\n[bold cyan]Designing your architecture...[/bold cyan]")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task(description="Generating...", total=None)
            gene = self._design_architecture()

        self._show_results(gene)
        self._save_generation(gene)

        console.print("\n[dim]Tip: Run [bold]archgene verify --hidden {} --heads {} --layers {}[/bold] to re-verify[/dim]".format(
            gene.hidden_dim, gene.num_heads, gene.num_layers
        ))

        return gene

    def _print_welcome(self):
        console.print(Panel.fit(
            "[bold cyan]ArchGene Architecture Designer[/bold cyan]\n\n"
            "Answer a few questions and I'll design a verified architecture\n"
            "tailored to your use case and budget.\n\n"
            "[dim]Press Ctrl+C to exit at any time[/dim]",
            title="Design Session",
            border_style="cyan"
        ))

    def _show_usage(self):
        monthly = self.tracker.count_this_month()
        total = self.tracker.total()
        console.print(f"[dim]Generations this month: {monthly} | Total: {total}[/dim]\n")

    def _ask_use_case(self):
        console.print("[bold]What's your primary use case?[/bold]\n")
        console.print("  [1] Research / experimentation")
        console.print("  [2] Production inference")
        console.print("  [3] Fine-tuning existing models")
        console.print("  [4] Edge / mobile deployment")
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"], default="1")
        mapping = {"1": "research", "2": "inference", "3": "finetuning", "4": "edge"}
        self.answers.use_case = mapping[choice]
        console.print()

    def _ask_budget(self):
        console.print("[bold]What's your budget range?[/bold]")
        console.print("  [1] Free / low (<$100) — hobby, experimentation")
        console.print("  [2] Mid ($100–$1K) — serious fine-tuning")
        console.print("  [3] High ($1K–$10K) — production training")
        console.print("  [4] Enterprise ($10K+) — large-scale training")
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"], default="2")
        mapping = {"1": "free", "2": "mid", "3": "high", "4": "enterprise"}
        self.answers.budget_tier = mapping[choice]
        console.print()

    def _ask_constraints(self):
        console.print("[bold]Any specific constraints? (comma-separated, or 'none')[/bold]")
        console.print("  • [1] Long context (>32K tokens)")
        console.print("  • [2] Low latency (<50ms per token)")
        console.print("  • [3] Low memory (<8GB VRAM)")
        console.print("  • [4] Quantization support")
        console.print("  • [0] None")
        choice = Prompt.ask("Choose", default="0")
        mapping = {
            "1": ["long_context"],
            "2": ["low_latency"],
            "3": ["low_memory"],
            "4": ["quantized"],
            "0": [],
        }
        if choice in mapping:
            self.answers.constraints = mapping[choice]
        else:
            selected = [c.strip() for c in choice.split(",") if c.strip() in mapping]
            self.answers.constraints = []
            for s in selected:
                self.answers.constraints.extend(mapping.get(s, []))
        console.print()

    def _ask_arch_family(self):
        console.print("[bold]Preferred architecture family?[/bold]")
        console.print("  [1] Transformer (proven, general-purpose)")
        console.print("  [2] Mixture of Experts (efficient scaling)")
        console.print("  [3] Linear / State Space (long context)")
        console.print("  [4] Any — pick the best for my needs")
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"], default="4")
        mapping = {"1": "transformer", "2": "moe", "3": "linear", "4": "any"}
        self.answers.arch_family = mapping[choice]
        console.print()

    def _ask_target_params(self):
        console.print("[bold]Target parameter count?[/bold]")
        console.print("  [1] <1B — lightweight, fast")
        console.print("  [2] 1B–7B — balanced (Recommended)")
        console.print("  [3] 7B–30B — capable, higher cost")
        console.print("  [4] 30B+ — maximum capability, high cost")
        console.print("  [5] Not sure — let the system decide")
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4", "5"], default="2")
        mapping = {"1": "under1b", "2": "1b-7b", "3": "7b-30b", "4": "30b+", "5": "auto"}
        self.answers.target_params = mapping[choice]
        console.print()

    def _ask_context_length(self):
        if "long_context" in self.answers.constraints:
            self.answers.context_length = 32768
            return
        console.print("[bold]Maximum context length?[/bold]")
        console.print("  [1] 2K (standard)")
        console.print("  [2] 8K (recommended for most tasks)")
        console.print("  [3] 32K (long documents)")
        console.print("  [4] 128K+ (very long context)")
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"], default="2")
        mapping = {"1": 2048, "2": 8192, "3": 32768, "4": 131072}
        self.answers.context_length = mapping[choice]
        console.print()

    def _answers_to_requirements(self) -> TaskRequirements:
        constraints = list(set(self.answers.constraints))
        priority = "balance"
        max_params = None
        max_memory_gb = None
        target_device = None

        budget_map = {
            "free": (50, 1_000_000_000, 4.0),
            "mid": (500, 7_000_000_000, 16.0),
            "high": (5000, 30_000_000_000, 80.0),
            "enterprise": (50000, 70_000_000_000, 320.0),
        }
        budget_hours, params_limit, mem_limit = budget_map.get(self.answers.budget_tier, (500, 7_000_000_000, 16.0))
        max_params = params_limit
        max_memory_gb = mem_limit

        param_map = {
            "under1b": (350_000_000, "efficiency"),
            "1b-7b": (7_000_000_000, "balance"),
            "7b-30b": (30_000_000_000, "capability"),
            "30b+": (70_000_000_000, "capability"),
            "auto": None,
        }
        if self.answers.target_params in param_map:
            entry = param_map[self.answers.target_params]
            if entry is not None:
                p_limit, p_priority = entry
                if p_limit:
                    max_params = min(max_params, p_limit) if max_params else p_limit
                if p_priority:
                    priority = p_priority

        if self.answers.use_case == "edge":
            target_device = "mobile"
            max_memory_gb = min(max_memory_gb or 999, 2.0)
            constraints.append("low_memory")

        if self.answers.use_case == "inference":
            priority = "efficiency"
            if "low_latency" not in constraints:
                constraints.append("low_latency")

        if self.answers.context_length >= 32768:
            constraints.append("long_context")

        return TaskRequirements(
            use_case=self.answers.use_case,
            constraints=constraints,
            priority=priority,
            target_device=target_device,
            max_params=max_params,
            max_memory_gb=max_memory_gb,
            context_length=self.answers.context_length if "long_context" in constraints else None,
        )

    def _select_family(self, req: TaskRequirements) -> ArchitectureFamily:
        pref = self.answers.arch_family
        if pref != "any":
            family_map = {
                "transformer": ArchitectureFamily.TRANSFORMER,
                "linear": ArchitectureFamily.MAMBA,
            }
            if pref in family_map:
                return family_map[pref]
        return ArchitectureSelector.select_family(req)

    def _design_architecture(self) -> Gene:
        req = self._answers_to_requirements()
        family = self._select_family(req)
        quant = ArchitectureSelector.select_quantization(req)
        return design_for_requirements(req, family, quant)

    def _show_results(self, gene: Gene):
        req = self._answers_to_requirements()

        v_result = self.verifier.verify_all(gene)
        cost = self.cost_estimator.full_estimate(gene)
        e_result = self.evaluator.evaluate(gene)

        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Architecture Design Complete[/bold cyan]\n\n"
            f"  [bold]Hidden:[/bold] {gene.hidden_dim}  "
            f"[bold]Layers:[/bold] {gene.num_layers}  "
            f"[bold]Heads:[/bold] {gene.num_heads}\n"
            f"  [bold]Params:[/bold] {gene.compute_params():,}  "
            f"[bold]VRAM:[/bold] {cost.vram_gb:.1f} GB\n"
            f"  [bold]Family:[/bold] {gene.arch_family.value}  "
            f"[bold]Z3:[/bold] {'[green]PASS[/green]' if v_result.is_valid else '[red]FAIL[/red]'}\n\n"
            f"[bold]Use case:[/bold] {self.answers.use_case}  "
            f"[bold]Score:[/bold] {e_result.score:.2f}",
            title="Results",
            border_style="green" if v_result.is_valid else "yellow"
        ))

        console.print(Panel.fit(
            f"[bold]Cost Estimates[/bold]\n\n"
            f"  Training (1T tokens): {cost.training_hours:.0f} GPU hours\n"
            f"  Training cost: ${cost.training_cost_per_1k_tokens:.2f}/1K tokens\n"
            f"  Inference: ${cost.inference_cost_per_1m_tokens:.2f}/M tokens\n"
            f"  Latency: {cost.inference_latency_ms:.2f} ms/token",
            title="Costs",
            border_style="blue"
        ))

        reason = generate_reasoning(self.answers.use_case, req, gene)
        console.print(Panel.fit(reason, title="Design Rationale", border_style="dim"))

    def _save_generation(self, gene: Gene):
        cost = self.cost_estimator.full_estimate(gene)
        v_result = self.verifier.verify_all(gene)
        record = GenerationRecord(
            timestamp=datetime.now().isoformat(),
            use_case=self.answers.use_case,
            hidden_dim=gene.hidden_dim,
            num_layers=gene.num_layers,
            num_heads=gene.num_heads,
            params=gene.compute_params(),
            verified=v_result.is_valid,
            vram_gb=cost.vram_gb,
        )
        self.tracker.add(record)
        console.print(f"\n[dim]Generation saved. Total designs: {self.tracker.total()}[/dim]")

    def run_with_answers(self, answers: DesignAnswers) -> dict[str, Any]:
        self.answers = answers
        gene = self._design_architecture()
        req = self._answers_to_requirements()
        v_result = self.verifier.verify_all(gene)
        cost = self.cost_estimator.full_estimate(gene)
        e_result = self.evaluator.evaluate(gene)
        self._save_generation(gene)
        return {
            "gene": gene,
            "verified": v_result.is_valid,
            "vram_gb": cost.vram_gb,
            "training_hours": cost.training_hours,
            "training_cost_per_1k": cost.training_cost_per_1k_tokens,
            "inference_cost_per_1m": cost.inference_cost_per_1m_tokens,
            "latency_ms": cost.inference_latency_ms,
            "score": e_result.score,
            "params": gene.compute_params(),
            "use_case": self.answers.use_case,
            "budget_tier": self.answers.budget_tier,
            "arch_family": gene.arch_family.value,
            "constraints": self.answers.constraints,
            "context_length": self.answers.context_length,
        }


def design_from_answers(
    use_case: str = "research",
    budget_tier: str = "mid",
    constraints: list[str] | None = None,
    arch_family: str = "any",
    target_params: str = "1b-7b",
    context_length: int = 8192,
) -> dict[str, Any]:
    answers = DesignAnswers(
        use_case=use_case,
        budget_tier=budget_tier,
        constraints=constraints or [],
        arch_family=arch_family,
        target_params=target_params,
        context_length=context_length,
    )
    session = DesignSession()
    return session.run_with_answers(answers)
