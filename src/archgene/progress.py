from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import Optional
import sys


class ProgressTracker:
    def __init__(self):
        self.console = Console()
        self.progress = None
        self.tasks = {}
    
    def start(self, description: str = "Processing"):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        self.progress.start()
        return self.progress.add_task(description, total=None)
    
    def update(self, task_id, advance: int = 1, description: Optional[str] = None):
        if self.progress and task_id in self.progress.tasks:
            self.progress.update(task_id, advance=advance)
            if description:
                self.progress.update(task_id, description=description)
    
    def stop(self):
        if self.progress:
            self.progress.stop()
    
    def print_status(self, message: str, style: str = "bold green"):
        self.console.print(f"[{style}]{message}[/{style}]")
    
    def print_error(self, message: str):
        self.console.print(f"[bold red]Error:[/bold red] {message}")
    
    def print_warning(self, message: str):
        self.console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
    
    def print_info(self, message: str):
        self.console.print(f"[bold blue]Info:[/bold blue] {message}")
    
    def print_table(self, title: str, data: list[dict], headers: Optional[list[str]] = None):
        if not data:
            return
        
        table = Table(title=title, show_header=True, header_style="bold magenta")
        
        if headers:
            for h in headers:
                table.add_column(h)
        else:
            first_row = data[0]
            for key in first_row.keys():
                table.add_column(str(key).replace("_", " ").title())
        
        for row in data:
            table.add_row(*[str(v) for v in row.values()])
        
        self.console.print(table)
    
    def print_gene_info(self, gene):
        table = Table(title="Gene Architecture", show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        gene_dict = gene.to_dict() if hasattr(gene, 'to_dict') else gene
        
        for key, value in gene_dict.items():
            key_display = key.replace("_", " ").title()
            if isinstance(value, list):
                value_display = ", ".join(str(v) for v in value)
            else:
                value_display = str(value)
            table.add_row(key_display, value_display)
        
        self.console.print(Panel(table, border_style="blue"))
    
    def print_score(self, score: float, max_score: float = 1.0):
        percentage = (score / max_score) * 100
        bar_length = 20
        filled = int(bar_length * score / max_score)
        bar = "#" * filled + "-" * (bar_length - filled)
        
        color = "green" if percentage >= 70 else "yellow" if percentage >= 40 else "red"
        
        self.console.print(f"Score: [{color}]{bar}[/{color}] {percentage:.1f}%")
    
    def confirm(self, message: str) -> bool:
        response = self.console.input(f"{message} [y/N]: ")
        return response.lower() in ("y", "yes")