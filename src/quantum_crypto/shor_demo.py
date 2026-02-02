"""Unified Shor's algorithm implementation with both classical and quantum methods.

Refactored to use Strategy Pattern and Typer CLI.
"""

from __future__ import annotations

import sys
from typing import Optional
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table

# Import new architecture
try:
    from .core import ShorResult
    from .algorithms import ClassicalShor, QuantumShor
    from .runner import run_shor
except ImportError:
    # Handle standalone execution
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from quantum_crypto.core import ShorResult
    from quantum_crypto.algorithms import ClassicalShor, QuantumShor
    from quantum_crypto.runner import run_shor


app = typer.Typer(help="Quantum RSA Lab - Shor's Algorithm Experiment CLI")
console = Console()


class Method(str, Enum):
    QUANTUM = "quantum"
    CLASSICAL = "classical"


@app.command()
def run(
    number: int = typer.Option(15, "--number", "-n", help="Number to factorize (N)"),
    base: Optional[int] = typer.Option(None, "--base", "-a", help="Base (a)"),
    method: Method = typer.Option(Method.QUANTUM, "--method", "-m", help="Execution method"),
    shots: int = typer.Option(1024, "--shots", "-s", help="Number of shots"),
    n_count: int = typer.Option(8, "--n-count", help="Number of counting qubits (quantum only)"),
):
    """Run Shor's algorithm for a single instance."""
    console.print(f"[bold blue]Running Shor's Algorithm[/bold blue]")
    console.print(f"N = {number}, Method = {method.value}")

    try:
        result = run_shor(
            number=number,
            base=base,
            shots=shots,
            method=method.value,
            n_count=n_count
        )
        
        _print_result(result)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def demo():
    """Run a demonstration with N=15 for both classical and quantum methods."""
    console.print("[bold green]=== Shor's Algorithm Demo (N=15) ===[/bold green]\n")
    
    bases = [7, 11, 2]
    
    # Classical
    console.print("[bold]Classical Implementation[/bold]")
    table_c = Table(show_header=True, header_style="bold magenta")
    table_c.add_column("Base (a)")
    table_c.add_column("Factors")
    table_c.add_column("Period (r)")
    table_c.add_column("Status")
    
    for a in bases:
        res = run_shor(15, a, method="classical")
        status = "[green]Success[/green]" if res.success else "[red]Failed[/red]"
        table_c.add_row(str(a), str(res.factors), str(res.period), status)
    
    console.print(table_c)
    console.print()

    # Quantum
    console.print("[bold]Quantum Implementation (QPE)[/bold]")
    try:
        # Check if Qiskit is available by trying to instantiate QuantumShor
        QuantumShor()
        
        table_q = Table(show_header=True, header_style="bold cyan")
        table_q.add_column("Base (a)")
        table_q.add_column("Factors")
        table_q.add_column("Period (r)")
        table_q.add_column("Phase")
        table_q.add_column("Status")

        for a in bases:
            res = run_shor(15, a, method="quantum", shots=1024)
            status = "[green]Success[/green]" if res.success else "[red]Failed[/red]"
            phase = f"{res.measured_phase:.4f}" if res.measured_phase is not None else "-"
            table_q.add_row(str(a), str(res.factors), str(res.period), phase, status)
        
        console.print(table_q)

    except RuntimeError as e:
        console.print(f"[yellow]Quantum demo skipped:[/yellow] {e}")


def _print_result(result: ShorResult):
    """Print the result in a nice format."""
    if result.success:
        console.print(f"[bold green]Success![/bold green] Factors: {result.factors}")
        console.print(f"Period (r): {result.period}")
        if result.measured_phase is not None:
            console.print(f"Measured Phase: {result.measured_phase:.4f}")
    else:
        console.print(f"[bold red]Failed.[/bold red] Method: {result.method}")
        if result.measured_phase is not None:
            console.print(f"Best Measured Phase: {result.measured_phase:.4f}")


if __name__ == "__main__":
    app()
