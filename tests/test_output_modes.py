"""Tests for output mode functionality."""

import pytest
from rich.console import Console
from rich.table import Table

from agent_interrogator.config import (
    InterrogationConfig,
    LLMConfig,
    ModelProvider,
    OutputMode,
)
from agent_interrogator.output import OutputManager


@pytest.fixture
def mock_console(monkeypatch):
    """Mock console to capture output."""

    class MockConsole:
        def __init__(self):
            self.output = []

        def print(self, *args, **kwargs):
            # Convert any Rich objects to strings
            if args:
                obj = args[0]
                if hasattr(obj, "__rich_console__"):
                    # For Rich objects that have a console representation
                    from rich.console import Console

                    real_console = Console()
                    segments = list(
                        obj.__rich_console__(real_console, real_console.options)
                    )
                    text = "".join(segment.text for segment in segments)
                    self.output.append(text)
                elif hasattr(obj, "__rich__"):
                    # For Rich objects that have a string representation
                    self.output.append(str(obj.__rich__()))
                else:
                    # For plain strings and other objects
                    self.output.append(str(obj))

    mock = MockConsole()
    # Mock Console in all relevant modules
    monkeypatch.setattr("agent_interrogator.output.Console", lambda: mock)
    monkeypatch.setattr("agent_interrogator.interrogator.Console", lambda: mock)
    return mock


@pytest.mark.parametrize(
    "mode", [OutputMode.QUIET, OutputMode.STANDARD, OutputMode.VERBOSE]
)
def test_output_modes(mock_console, mode):
    """Test different output modes."""
    output = OutputManager(mode)

    # Test standard print
    output.print("Standard message")

    # Test verbose print
    output.print_verbose("Verbose message")

    # Test status display
    output.display_status("Status message")

    # Test table display
    table = Table()
    table.add_column("Test")
    table.add_row("Value")
    output.display_table(table)

    # Print statements should appear in stdout, but not in mock
    # because we're mocking the Console instance in OutputManager
    if mode == OutputMode.QUIET:
        assert len(mock_console.output) == 0
    elif mode == OutputMode.STANDARD:
        assert len(mock_console.output) >= 3  # Standard message + status + table
        assert any("Standard message" in out for out in mock_console.output)
        assert any("Status message" in out for out in mock_console.output)
    else:  # VERBOSE
        assert len(mock_console.output) >= 4  # Standard + verbose + status + table
        assert any("Standard message" in out for out in mock_console.output)
        assert any("Verbose message" in out for out in mock_console.output)
        assert any("Status message" in out for out in mock_console.output)


@pytest.mark.asyncio
async def test_interrogator_output_modes(mock_console):
    """Test output modes in AgentInterrogator."""
    from agent_interrogator.interrogator import AgentInterrogator

    # Create test config with each mode
    async def mock_callback(prompt: str) -> str:
        return "Test response"

    for mode in OutputMode:
        config = InterrogationConfig(
            llm=LLMConfig(
                provider=ModelProvider.OPENAI,
                model_name="test-model",
                api_key="test-key",
            ),
            output_mode=mode,
        )

        mock_console.output.clear()
        interrogator = AgentInterrogator(config, mock_callback)

        if mode == OutputMode.QUIET:
            assert (
                len(mock_console.output) == 0
            ), f"Expected no output in quiet mode, got: {mock_console.output}"
        else:
            # Should have startup info (logo, config table, ready message)
            assert (
                len(mock_console.output) >= 3
            ), f"Expected at least 3 outputs, got {len(mock_console.output)}: {mock_console.output}"
            assert any(
                "Agent Interrogator Configuration" in out for out in mock_console.output
            ), "Configuration table not found in output"
            assert any(
                "Ready to begin interrogation" in out for out in mock_console.output
            ), "Ready message not found in output"
