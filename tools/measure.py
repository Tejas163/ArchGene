import click
import time
import json
from pathlib import Path
from datetime import datetime


DATA_DIR = Path("~/.archgene").expanduser()
DATA_DIR.mkdir(exist_ok=True)


@click.group()
def measure():
    """Measure DX metrics."""
    pass


@measure.command()
@click.option("--command", "-c", required=True, help="Command to measure")
def tthw(command):
    """Measure Time to Hello World for a command."""
    start = time.perf_counter()
    click.echo(f"Measuring: {command}")
    
    import subprocess
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    
    end = time.perf_counter()
    elapsed = end - start
    
    # Log result
    record = {
        "command": command,
        "elapsed_seconds": elapsed,
        "exit_code": result.returncode,
        "timestamp": datetime.now().isoformat()
    }
    
    log_path = DATA_DIR / "tthw.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    
    # Report
    if result.returncode == 0:
        click.echo(f"[green]Success:[/green] {elapsed:.2f}s")
    else:
        click.echo(f"[red]Failed:[/red] {elapsed:.2f}s")
        if result.stderr:
            click.echo(f"Error: {result.stderr[:200]}")
    
    # Benchmark
    if elapsed < 60:
        tier = "CHAMPION"
    elif elapsed < 300:
        tier = "COMPETITIVE"
    elif elapsed < 600:
        tier = "NEEDS_WORK"
    else:
        tier = "RED_FLAG"
    
    click.echo(f"Tier: {tier}")
    return tier


@measure.command()
def benchmark():
    """Run TTHW benchmark suite."""
    commands = [
        ("python main.py version", "version"),
        ("python main.py evaluate", "evaluate"),
        ("python main.py verify", "verify"),
        ("python main.py visualize", "visualize"),
        ("python main.py export", "export"),
    ]
    
    results = []
    for cmd, name in commands:
        start = time.perf_counter()
        import subprocess
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        elapsed = time.perf_counter() - start
        results.append({
            "command": name,
            "seconds": elapsed,
            "passed": result.returncode == 0
        })
        click.echo(f"{name}: {elapsed:.2f}s {'[green]✓[/green]' if result.returncode == 0 else '[red]✗[/red]'}")
    
    # Save
    log_path = DATA_DIR / "benchmark.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    
    avg = sum(r["seconds"] for r in results) / len(results)
    click.echo(f"\nAverage: {avg:.2f}s")


@measure.command()
def report():
    """Show metrics report."""
    tthw_path = DATA_DIR / "tthw.jsonl"
    bench_path = DATA_DIR / "benchmark.json"
    
    if not tthw_path.exists() and not bench_path.exists():
        click.echo("No metrics yet. Run:")
        click.echo("  archgene measure tthw -c 'python main.py evaluate'")
        click.echo("  archgene measure benchmark")
        return
    
    click.echo("=== DX Metrics ===\n")
    
    if bench_path.exists():
        with open(bench_path) as f:
            bench = json.load(f)
        click.echo("CLI Benchmark:")
        for r in bench:
            status = "✓" if r["passed"] else "✗"
            click.echo(f"  {r['command']}: {r['seconds']:.2f}s {status}")
    
    if tthw_path.exists():
        with open(tthw_path) as f:
            tthw = [json.loads(line) for line in f]
        click.echo(f"\nTTHW Records: {len(tthw)}")
        if tthw:
            avg = sum(r["elapsed_seconds"] for r in tthw) / len(tthw)
            click.echo(f"Average: {avg:.2f}s")


if __name__ == "__main__":
    measure()