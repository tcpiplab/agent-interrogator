"""Output management for the agent interrogator."""

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import OutputMode


class OutputManager:
    """Manages terminal output based on configured output mode."""

    def __init__(self, output_mode: OutputMode):
        self.output_mode = output_mode
        self.console = Console()

    def print(self, *args, **kwargs) -> None:
        """Print output if not in quiet mode."""
        if self.output_mode != OutputMode.QUIET:
            self.console.print(*args, **kwargs)

    def print_verbose(self, *args, **kwargs) -> None:
        """Print output only in verbose mode."""
        if self.output_mode == OutputMode.VERBOSE:
            self.console.print(*args, **kwargs)

    def display_prompt(self, prompt: str, cycle: int, context: str) -> None:
        """Display an interrogation prompt."""
        if self.output_mode == OutputMode.VERBOSE:
            self.console.print(Panel(
                Syntax(prompt, "markdown"),
                title=f"[bold cyan]Analysis Cycle {cycle} - {context}[/bold cyan] - Prompt",
                border_style="cyan"
            ))

    def display_response(self, response: str, cycle: int, context: str) -> None:
        """Display an agent's response."""
        if self.output_mode == OutputMode.VERBOSE:
            self.console.print(Panel(
                response,
                title=f"[bold green]Analysis Cycle {cycle} - {context}[/bold green] - Agent Response",
                border_style="green"
            ))

    def display_process_result(self, result: dict, cycle: int, context: str) -> None:
        """Display processed response results."""
        if self.output_mode == OutputMode.VERBOSE:
            self.console.print(Panel(
                str(result),
                title=f"[bold yellow]Analysis Cycle {cycle} - {context}[/bold yellow] - Processed Result",
                border_style="yellow"
            ))

    def display_status(self, message: str, style: str = "yellow") -> None:
        """Display a status message in standard and verbose modes."""
        if self.output_mode != OutputMode.QUIET:
            self.console.print(f"[{style}]{message}[/{style}]")

    def display_table(self, table: Table) -> None:
        """Display a rich table in standard and verbose modes."""
        if self.output_mode != OutputMode.QUIET:
            self.console.print(table)

    def display_panel(self, panel: Panel) -> None:
        """Display a rich panel in standard and verbose modes."""
        if self.output_mode != OutputMode.QUIET:
            self.console.print(panel)
