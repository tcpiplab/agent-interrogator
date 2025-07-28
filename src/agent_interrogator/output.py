"""Output management for the agent interrogator."""

import json
from typing import Any

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .config import OutputMode


class OutputManager:
    """Manages terminal output based on configured output mode."""

    def __init__(self, output_mode: OutputMode):
        self.output_mode = output_mode
        self.console = Console()

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print output if not in quiet mode."""
        if self.output_mode != OutputMode.QUIET:
            self.console.print(*args, **kwargs)

    def print_verbose(self, *args: Any, **kwargs: Any) -> None:
        """Print output only in verbose mode."""
        if self.output_mode == OutputMode.VERBOSE:
            self.console.print(*args, **kwargs)

    def display_prompt(self, prompt: str, cycle: int, context: str) -> None:
        """Display an interrogation prompt."""
        if self.output_mode == OutputMode.VERBOSE:
            # Use Text for proper word wrapping instead of Syntax
            # Syntax doesn't support automatic word wrapping for long lines
            wrapped_text = Text(prompt, style="cyan", overflow="fold")
            self.console.print(
                Panel(
                    wrapped_text,
                    title=f"[bold cyan]Analysis Cycle {cycle} - {context}[/bold cyan] - Prompt",
                    border_style="cyan",
                    expand=False,
                )
            )

    def display_response(self, response: str, cycle: int, context: str) -> None:
        """Display an agent's response."""
        if self.output_mode == OutputMode.VERBOSE:
            # Use Text for proper word wrapping
            wrapped_text = Text(response, style="green", overflow="fold")
            self.console.print(
                Panel(
                    wrapped_text,
                    title=f"[bold green]Analysis Cycle {cycle} - {context}[/bold green] - Agent Response",
                    border_style="green",
                    expand=False,
                )
            )

    def display_process_result(
        self, result: dict[str, Any], cycle: int, context: str
    ) -> None:
        """Display processed response results."""
        if self.output_mode == OutputMode.VERBOSE:
            # Try to display as formatted JSON for better readability
            try:
                # Use Rich's JSON renderer which handles wrapping automatically
                content = JSON(json.dumps(result, indent=2))
            except (TypeError, ValueError):
                # Fallback to text with wrapping if JSON serialization fails
                content = Text(str(result), style="yellow", overflow="fold")  # type: ignore

            self.console.print(
                Panel(
                    content,
                    title=f"[bold yellow]Analysis Cycle {cycle} - {context}[/bold yellow] - Processed Result",
                    border_style="yellow",
                    expand=False,
                )
            )

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
