"""Main interrogator implementation."""

from typing import Any, Awaitable, Callable, Dict, List

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import InterrogationConfig, ModelProvider
from .llm import HuggingFaceLLM, LLMInterface, OpenAILLM
from .models import AgentProfile, Capability, Function, Parameter
from .output import OutputManager

# Type alias for the agent interaction callback
AgentCallback = Callable[[str], Awaitable[str]]

LOGO = r"""[bold blue]
    ___                    __       ____      __                                   __            
   /   | ____ ____  ____  / /_     /  _/___  / /____  ______________  ____ _____ _/ /_____  _____
  / /| |/ __ `/ _ \/ __ \/ __/     / // __ \/ __/ _ \/ ___/ ___/ __ \/ __ `/ __ `/ __/ __ \/ ___/
 / ___ / /_/ /  __/ / / / /_     _/ // / / / /_/  __/ /  / /  / /_/ / /_/ / /_/ / /_/ /_/ / /    
/_/  |_\__, /\___/_/ /_/\__/    /___/_/ /_/\__/\___/_/  /_/   \____/\__, /\__,_/\__/\____/_/     
      /____/                                                       /____/                        
[/bold blue]"""


class AgentInterrogator:
    """Main class for interrogating AI agents.

    Args:
        config (InterrogationConfig): Configuration for the interrogator
        agent_callback (AgentCallback): Async callback function that takes a prompt
            string and returns the agent's response string
    """

    def __init__(self, config: InterrogationConfig, agent_callback: AgentCallback):
        self.config = config
        self.agent_callback = agent_callback
        self.profile = AgentProfile()
        self.output = OutputManager(self.config.output_mode)
        self.llm = self._initialize_llm()

        # Display logo and configuration
        self._display_startup_info()

    def _display_startup_info(self) -> None:
        """Display the ASCII art logo and configuration information."""
        # Print logo
        self.output.print(LOGO)

        # Create configuration table
        config_table = Table(
            title="[bold cyan]Agent Interrogator Configuration[/bold cyan]"
        )
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        # Add configuration rows
        config_table.add_row("LLM Provider", str(self.config.llm.provider))
        config_table.add_row("Model Name", self.config.llm.model_name)
        config_table.add_row(
            "API Key", "********" if self.config.llm.api_key else "Not provided"
        )
        config_table.add_row("Max Iterations", str(self.config.max_iterations))
        config_table.add_row("Output Mode", str(self.config.output_mode))

        # Add any model-specific kwargs as a list
        kwargs_str = "\n".join(
            f"{k}: {v}" for k, v in self.config.llm.model_kwargs.items()
        )
        if kwargs_str:
            config_table.add_row("Model Settings", kwargs_str)

        self.output.display_table(config_table)
        self.output.print(
            "\n[bold green]Ready to begin interrogation...\n[/bold green]"
        )

    def _initialize_llm(self) -> LLMInterface:
        """Initialize the appropriate LLM based on configuration."""
        if self.config.llm.provider == ModelProvider.OPENAI:
            return OpenAILLM(self.config, self.output)
        elif self.config.llm.provider == ModelProvider.HUGGINGFACE:
            return HuggingFaceLLM(self.config, self.output)
        elif self.config.llm.provider == ModelProvider.OLLAMA:
            from .llm import OllamaLLM
            return OllamaLLM(self.config, self.output)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")

    def _display_profile(self) -> None:
        """Display the complete agent profile in a structured format."""
        self.output.print(
            "\n[bold magenta]═══ Agent Profile Summary ═══[/bold magenta]\n"
        )

        for capability in self.profile.capabilities:
            # Create a table for each capability
            cap_table = Table(
                title=f"[bold cyan]{capability.name}[/bold cyan]",
                caption=capability.description or "No description provided",
                show_header=True,
                header_style="bold green",
            )
            cap_table.add_column("Function", style="cyan")
            cap_table.add_column("Description", style="white")
            cap_table.add_column("Parameters", style="yellow")
            cap_table.add_column("Return Type", style="magenta")

            # Add rows for each function
            for func in capability.functions:
                params = (
                    "\n".join([f"{p.name}: {p.type}" for p in func.parameters])
                    if func.parameters
                    else "None"
                )
                cap_table.add_row(
                    func.name,
                    func.description or "No description",
                    params,
                    func.return_type or "void",
                )

            self.output.print(cap_table)
            self.output.print("\n")

        # Print summary statistics
        self.output.print(
            f"[bold green]Total Capabilities:[/bold green] {len(self.profile.capabilities)}"
        )
        total_functions = sum(len(cap.functions) for cap in self.profile.capabilities)
        self.output.print(
            f"[bold green]Total Functions:[/bold green] {total_functions}\n"
        )

    async def interrogate(self) -> AgentProfile:
        """Perform the full interrogation process with iterative discovery and analysis."""
        self.output.print("\n[bold cyan]Starting Agent Interrogation...[/bold cyan]\n")

        # Iterative capability discovery
        capabilities = await self._discover_capabilities()
        self.profile.capabilities.extend(capabilities)

        # Iterative analysis of each capability
        for capability in self.profile.capabilities:
            # _analyze_capability modifies the capability in-place and returns discovered functions
            discovered_functions = await self._analyze_capability(capability)
            self.output.print(
                f"[bold cyan]Completed analysis of {capability.name}. Found {len(discovered_functions)} functions.[/bold cyan]"
            )

        # Display the final profile
        self._display_profile()

        return self.profile

    async def _discover_capabilities(self) -> list[Capability]:
        """Discover high-level capabilities of the agent through multiple cycles."""
        discovered_capabilities: List[Capability] = []
        cycle = 0
        previous_responses: List[Dict[str, Any]] = []
        result: Dict[str, Any] = {}

        while cycle < self.config.max_iterations:
            context = {
                "phase": "discovery",
                "cycle": cycle,
                "previous_responses": previous_responses,
                "discovered_capabilities": [
                    cap.model_dump() for cap in discovered_capabilities
                ],
                "next_cycle_focus": (
                    result.get("next_cycle_focus") if cycle > 0 else None
                ),
            }

            # Generate and send prompt
            prompt = await self.llm.generate_prompt(context)

            # Display the prompt being sent
            self.output.print_verbose(
                Panel(
                    Text(prompt, style="cyan", overflow="fold"),
                    title=f"[bold cyan]Discovery Cycle {cycle + 1}[/bold cyan] - Prompt",
                    border_style="cyan",
                    expand=False,
                )
            )

            # Get response from agent
            response = await self.agent_callback(prompt)

            # Display the response received
            self.output.print_verbose(
                Panel(
                    Text(response, style="cyan", overflow="fold"),
                    title=f"[bold cyan]Discovery Cycle {cycle + 1}[/bold cyan] - Agent Response",
                    border_style="cyan",
                    expand=False,
                )
            )

            # Store the prompt/response pair in conversation history
            previous_responses.append(
                {"prompt": prompt, "response": response, "cycle": cycle}
            )

            # Process response
            result = await self.llm.process_response(response, context)

            # Add new capabilities with validation
            new_capabilities = []
            for cap in result.get("capabilities", []):
                if isinstance(cap, dict) and "name" in cap and "description" in cap:
                    try:
                        new_capabilities.append(Capability(**cap))
                    except Exception as e:
                        self.output.print(f"[red]Error creating capability: {e}[/red]")
                else:
                    self.output.print(f"[red]Invalid capability format: {cap}[/red]")
            discovered_capabilities.extend(new_capabilities)

            # Check if we should continue discovery
            should_continue = await self.llm.should_continue_cycle(result)
            self.output.print(
                f"[yellow]Discovery cycle {cycle + 1} complete. is_complete={result.get('is_complete', False)}[/yellow]"
            )
            if not should_continue:
                self.output.print(
                    "[bold yellow]Discovery phase complete![/bold yellow]"
                )
                break

            cycle += 1

        return discovered_capabilities

    async def _analyze_capability(self, capability: Capability) -> List[Function]:
        """Analyze a specific capability in detail through multiple cycles."""
        cycle = 0
        previous_responses: List[Dict[str, Any]] = []
        discovered_functions: List[Function] = []
        result: Dict[str, Any] = {}

        while cycle < self.config.max_iterations:
            # Build context with Pydantic models directly
            context = {
                "phase": "analysis",
                "cycle": cycle,
                "previous_responses": previous_responses,
                "capability": capability,  # Use Pydantic model directly
                "discovered_functions": discovered_functions,  # Use list of Pydantic models
                "next_cycle_focus": (
                    result.get("next_cycle_focus") if cycle > 0 else None
                ),
            }

            # Generate and send prompt
            prompt = await self.llm.generate_prompt(context)

            # Display the prompt being sent
            self.output.display_prompt(prompt, cycle + 1, capability.name)

            # Get response from agent
            response = await self.agent_callback(prompt)

            # Display the response received
            self.output.display_response(response, cycle + 1, capability.name)

            # Store the prompt/response pair in conversation history
            previous_responses.append(
                {"prompt": prompt, "response": response, "cycle": cycle}
            )

            # Process the response
            result = await self.llm.process_response(response, context)

            # Extract discovered functions with validation
            new_functions = []
            for func_data in result.get("functions", []):
                try:
                    # Ensure we have a dictionary copy to modify safely
                    func_dict = (
                        func_data.copy()
                        if isinstance(func_data, dict)
                        else func_data.dict()
                    )

                    # Process parameters - convert all to Parameter objects
                    if "parameters" in func_dict:
                        normalized_params = []
                        for param in func_dict["parameters"]:
                            if isinstance(param, str):
                                # Extract parameter type if specified in format "name: type"
                                if ":" in param:
                                    name, param_type = param.split(":", 1)
                                    normalized_params.append(
                                        Parameter(
                                            name=name.strip(), type=param_type.strip()
                                        )
                                    )
                                else:
                                    normalized_params.append(
                                        Parameter(
                                            name=param.strip(),
                                            type="string",  # Default to string if type not specified
                                        )
                                    )
                            elif isinstance(param, dict):
                                normalized_params.append(Parameter(**param))
                            else:
                                normalized_params.append(
                                    param
                                )  # Already a Parameter object
                        func_dict["parameters"] = normalized_params

                    new_functions.append(Function(**func_dict))
                except Exception as e:
                    self.output.print(f"[red]Error creating function: {e}[/red]")

            # Add new functions to the capability and our tracking list
            capability.functions.extend(new_functions)
            discovered_functions.extend(new_functions)

            # Display progress
            if new_functions:
                self.output.display_status(
                    f"Found {len(new_functions)} new functions in {capability.name}"
                )

            # Display process results in verbose mode
            self.output.display_process_result(result, cycle + 1, capability.name)

            # Check if we should continue analysis
            should_continue = await self.llm.should_continue_cycle(result)
            self.output.display_status(
                f"Analysis cycle {cycle + 1} complete. is_complete={result.get('is_complete', False)}"
            )
            if not should_continue:
                self.output.display_status(
                    f"Analysis of {capability.name} complete!", "bold yellow"
                )
                break

            cycle += 1

        return discovered_functions
