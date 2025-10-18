"""LLM interface implementations."""

import json
import os
import platform
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .output import OutputManager

import httpx
import torch
from openai import AsyncOpenAI
from transformers import pipeline

try:
    import ane_transformers  # type: ignore

    HAS_ANE = platform.processor() in ["arm", "arm64"]
except ImportError:
    HAS_ANE = False

from .config import HuggingFaceConfig, InterrogationConfig, LLMConfig, ModelProvider
from .models import Function, Parameter
from .prompt_templates import (  # Discovery templates; Analysis templates; Processing templates; Schema templates; Formatting templates
    ANALYSIS_JSON_SCHEMA,
    ANALYSIS_PROCESSING_PROMPT_TEMPLATE,
    ANALYSIS_PROCESSING_SYSTEM_PROMPT,
    ANALYSIS_PROMPT_TEMPLATE,
    DISCOVERY_JSON_SCHEMA,
    DISCOVERY_PROCESSING_PROMPT_TEMPLATE,
    DISCOVERY_PROCESSING_SYSTEM_PROMPT,
    DISCOVERY_PROMPT_TEMPLATE,
    INITIAL_ANALYSIS_PROMPT_TEMPLATE,
    INITIAL_DISCOVERY_PROMPT,
    KNOWN_ITEMS_TEMPLATE,
)


class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""

    def __init__(self, config: InterrogationConfig, output_manager: "OutputManager"):
        """Initialize the LLM interface.

        Args:
            config: Full interrogation configuration
            output_manager: OutputManager instance for controlled output
        """
        self.config = config
        self.output = output_manager

    @abstractmethod
    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        """Generate an interrogation prompt based on context.

        Args:
            context: Dictionary containing:
                - phase: str, one of 'discovery', 'analysis'
                - cycle: int, current iteration number
                - previous_responses: list of previous responses
                - discovered_capabilities: list of capabilities found so far
                - capability: dict, current capability being analyzed (for analysis phase)
                - discovered_functions: list of functions found so far (for analysis phase)
        """
        pass

    @abstractmethod
    async def process_discovery_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response during capability discovery phase.

        Args:
            response: Raw response from the agent
            context: Current conversation context

        Returns:
            Dict containing:
            - capabilities: List of discovered capabilities
            - next_cycle_focus: Optional guidance for next discovery cycle
        """
        pass

    @abstractmethod
    async def process_analysis_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response during capability analysis phase.

        Args:
            response: Raw response from the agent
            context: Current conversation context with capability being analyzed

        Returns:
            Dict containing:
            - functions: List of discovered functions
            - next_cycle_focus: Optional guidance for next analysis cycle
        """
        pass

    async def process_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response based on the current phase.

        Args:
            response: Raw response from the agent
            context: Current conversation context

        Returns:
            Structured data extracted from the response
        """
        phase = context.get("phase", "discovery")
        if phase == "discovery":
            return await self.process_discovery_response(response, context)
        else:
            return await self.process_analysis_response(response, context)

    def _format_known_items(self, context: Dict[str, Any]) -> str:
        """Format known capabilities and functions for prompt context."""
        # Format capabilities
        capabilities = context.get("discovered_capabilities", [])
        capabilities_str = (
            "\\n".join(f"- {cap.name}: {cap.description or ''}" for cap in capabilities)
            or "None"
        )

        # Format functions
        functions = context.get("discovered_functions", [])
        functions_str = (
            "\\n".join(f"- {func.name}: {func.description or ''}" for func in functions)
            or "None"
        )

        return KNOWN_ITEMS_TEMPLATE.format(
            capabilities_str=capabilities_str, functions_str=functions_str
        )

    async def should_continue_cycle(self, results: Dict[str, Any]) -> bool:
        """Determine if another cycle should be run based on results."""
        # Stop if marked complete
        if results.get("is_complete", False):
            return False

        return True


class OpenAILLM(LLMInterface):
    """OpenAI-based LLM implementation."""

    def __init__(self, config: InterrogationConfig, output_manager: "OutputManager"):
        super().__init__(config, output_manager)
        if config.llm.provider != ModelProvider.OPENAI:
            raise ValueError("OpenAILLM requires provider to be OPENAI")

        # Detect if proxy is configured and create appropriate http client
        http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
        https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
        use_proxy = bool(http_proxy or https_proxy)

        if use_proxy:
            # Create httpx client with SSL verification disabled for proxy interception
            http_client = httpx.AsyncClient(verify=False)
            self.client = AsyncOpenAI(api_key=config.llm.api_key, http_client=http_client)
            self.output.print_verbose(
                "[yellow]Detected proxy configuration - SSL verification disabled for OpenAI client[/yellow]"
            )
        else:
            self.client = AsyncOpenAI(api_key=config.llm.api_key)

        self.model_kwargs = config.llm.model_kwargs

    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        """Generate an interrogation prompt based on context."""
        phase = context.get("phase", "discovery")
        cycle = context.get("cycle", 0)

        if cycle == 0:
            if phase == "discovery":
                # Initial discovery prompt
                return INITIAL_DISCOVERY_PROMPT
            else:
                # Initial analysis prompt
                return INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(
                    capability_name=context["capability"].name
                )
        else:
            if phase == "discovery":
                system_prompt = DISCOVERY_PROCESSING_SYSTEM_PROMPT
                # Follow-up discovery prompts
                discovered = context.get("discovered_capabilities", [])
                next_focus = context.get("next_cycle_focus")

                capabilities_str = "\n".join(
                    f"- {cap.get('name', '')}: {cap.get('description', '')}"
                    for cap in discovered
                )

                interrogation_prompt_request = DISCOVERY_PROMPT_TEMPLATE.format(
                    capabilities_str=capabilities_str,
                    focus_guidance=next_focus,
                    context=context.get("previous_responses", []),
                )
            else:
                system_prompt = ANALYSIS_PROCESSING_SYSTEM_PROMPT
                # Analysis phase
                capability = context["capability"]
                discovered_functions = context.get("discovered_functions", [])
                next_focus = context.get("next_cycle_focus")

                functions_str = "\n".join(
                    f"- {func.name}: {func.description or ''}"
                    for func in discovered_functions
                )

                interrogation_prompt_request = ANALYSIS_PROMPT_TEMPLATE.format(
                    capability=capability,
                    functions_str=functions_str,
                    focus_guidance=next_focus,
                    context=context.get("previous_responses", []),
                )

        interrogation_prompt = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": interrogation_prompt_request},
            ],
            temperature=0.1,
            **self.model_kwargs,
        )
        content = interrogation_prompt.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI API returned empty content")
        return str(content)

    async def process_discovery_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response during capability discovery phase."""
        discovery_prompt = DISCOVERY_PROCESSING_PROMPT_TEMPLATE.format(
            json_format=DISCOVERY_JSON_SCHEMA, context=context, response=response
        )
        self.output.print_verbose("[bold cyan]Discovery Prompt:[/bold cyan]")
        self.output.print_verbose(discovery_prompt)

        discovery = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=[
                {"role": "system", "content": DISCOVERY_PROCESSING_SYSTEM_PROMPT},
                {"role": "user", "content": discovery_prompt},
            ],
            temperature=0.1,
            **self.model_kwargs,
        )

        try:
            content = discovery.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI API returned empty content")

            # Strip whitespace
            content = content.strip()

            # Try to extract JSON from markdown code block if present
            import re

            # Check for markdown code block
            markdown_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if markdown_match:
                content = markdown_match.group(1).strip()

            result = json.loads(content)
            if isinstance(result, dict):
                return result
            else:
                raise ValueError("Expected dictionary response from JSON")
        except json.JSONDecodeError as e:
            self.output.print_verbose(
                "[red]Failed to parse JSON response. Raw content:[/red]"
            )
            self.output.print_verbose(content)
            raise ValueError(f"Failed to process discovery response: {str(e)}")

    async def process_analysis_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response during capability analysis phase."""
        analysis_prompt = ANALYSIS_PROCESSING_PROMPT_TEMPLATE.format(
            capability_name=context["capability"].name,
            json_format=ANALYSIS_JSON_SCHEMA,
            response=response,
            context=context,
        )

        self.output.print_verbose("[bold cyan]Analysis Prompt:[/bold cyan]")
        self.output.print_verbose(analysis_prompt)

        # First attempt - try to get a clean JSON response
        analysis = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=[
                {"role": "system", "content": ANALYSIS_PROCESSING_SYSTEM_PROMPT},
                {"role": "user", "content": analysis_prompt},
            ],
            temperature=0.1,
            **self.model_kwargs,
        )

        content = analysis.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI API returned empty content")
        content = content.strip()

        try:
            # Try to parse the raw content first
            result = json.loads(content)
            if isinstance(result, dict):
                return result
            else:
                raise ValueError("Expected dictionary response from JSON")
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the content
            import re

            json_matches = re.findall(r"\{[^{}]*\}", content)

            for json_str in json_matches:
                try:
                    # Try to parse each potential JSON object
                    result = json.loads(json_str)
                    # Verify it has the expected structure
                    if isinstance(result, dict) and "functions" in result:
                        return result
                except json.JSONDecodeError:
                    continue

            # If we get here, try one more time with a more explicit prompt
            analysis = await self.client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You MUST respond with ONLY a valid JSON object following the schema exactly. No other text or explanation.",
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.1,
                **self.model_kwargs,
            )

            content = analysis.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI API returned empty content")
            content = content.strip()

            try:
                result = json.loads(content)
                if isinstance(result, dict):
                    return result
                else:
                    raise ValueError("Expected dictionary response from JSON")
            except json.JSONDecodeError as e:
                self.output.print_verbose(
                    "[red]Failed to parse JSON response. Raw content:[/red]"
                )
                self.output.print_verbose(content)
                raise ValueError(
                    f"Failed to process analysis response after multiple attempts: {str(e)}"
                )


class HuggingFaceLLM(LLMInterface):
    """HuggingFace-based LLM implementation."""

    def __init__(self, config: InterrogationConfig, output_manager: "OutputManager"):
        super().__init__(config, output_manager)
        if config.llm.provider != ModelProvider.HUGGINGFACE:
            raise ValueError("HuggingFaceLLM requires provider to be HUGGINGFACE")

        # Get HuggingFace-specific config or use defaults
        hf_config = config.llm.huggingface or HuggingFaceConfig(
            local_model_path=None,
            allow_download=True,
            revision=None,
            device="auto",
            quantization=None,
        )

        # Determine model source (local path or hub)
        model_source = hf_config.local_model_path or config.llm.model_name
        if not model_source:
            raise ValueError("Either model_name or local_model_path must be provided")

        # If using hub model and downloads are disabled, verify model exists locally
        if not hf_config.local_model_path and not hf_config.allow_download:
            from huggingface_hub import snapshot_download

            try:
                # Try to find model in cache without downloading
                snapshot_download(
                    repo_id=config.llm.model_name,
                    revision=hf_config.revision,
                    local_files_only=True,
                )
            except Exception as e:
                raise ValueError(
                    f"Model {config.llm.model_name} not found locally and downloads are disabled. "
                    f"Error: {str(e)}"
                )

        # Handle device placement
        device = hf_config.device
        if device == "auto":
            if HAS_ANE:
                device = "ane"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # Set up pipeline kwargs
        pipeline_kwargs = {
            "model": model_source,
            "device": device if device != "ane" else "cpu",  # ANE setup happens later
            **config.llm.model_kwargs,
        }

        # Add revision if specified
        if hf_config.revision:
            pipeline_kwargs["revision"] = hf_config.revision

        # Configure quantization if specified
        if hf_config.quantization:
            import torch

            if device == "ane":
                # ANE requires FP16
                pipeline_kwargs["torch_dtype"] = torch.float16
            elif device == "cuda":
                from transformers import BitsAndBytesConfig

                if hf_config.quantization == "int8":
                    pipeline_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_4bit_compute_dtype=torch.float16,  # More stable than default fp32
                    )
                elif hf_config.quantization == "fp16":
                    pipeline_kwargs["torch_dtype"] = torch.float16
            else:
                self.output.print_verbose(
                    "[yellow]Warning: Quantization requested but no GPU/ANE available. Falling back to full precision.[/yellow]"
                )

        # Log model loading details in verbose mode
        self.output.print_verbose(f"[bold cyan]Loading HuggingFace model:[/bold cyan]")
        self.output.print_verbose(f"Source: {model_source}")
        self.output.print_verbose(f"Device: {hf_config.device}")
        if hf_config.quantization:
            self.output.print_verbose(f"Quantization: {hf_config.quantization}")

        # Initialize the pipeline
        self.pipe = pipeline("text-generation", **pipeline_kwargs)

        # Set up ANE if requested and available
        if device == "ane":
            if not HAS_ANE:
                self.output.print_verbose(
                    "[yellow]ANE requested but not available. Using CPU instead.[/yellow]"
                )
            else:
                try:
                    from ane_transformers.model import init_ane_model  # type: ignore

                    self.pipe.model = init_ane_model(self.pipe.model)
                    self.output.print_verbose(
                        "[green]Successfully initialized model on Apple Neural Engine[/green]"
                    )
                except Exception as e:
                    self.output.print_verbose(
                        f"[yellow]Failed to initialize ANE model: {str(e)}. Using CPU instead.[/yellow]"
                    )

    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        """Generate an interrogation prompt based on context."""
        phase = context.get("phase", "discovery")
        cycle = context.get("cycle", 0)

        if cycle == 0:
            if phase == "discovery":
                # Initial discovery prompt
                return INITIAL_DISCOVERY_PROMPT
            else:
                # Initial analysis prompt
                return INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(
                    capability_name=context["capability"].name
                )
        else:
            if phase == "discovery":
                system_prompt = DISCOVERY_PROCESSING_SYSTEM_PROMPT
                # Follow-up discovery prompts
                discovered = context.get("discovered_capabilities", [])
                next_focus = context.get("next_cycle_focus")

                capabilities_str = "\n".join(
                    f"- {cap.get('name', '')}: {cap.get('description', '')}"
                    for cap in discovered
                )

                interrogation_prompt_request = DISCOVERY_PROMPT_TEMPLATE.format(
                    capabilities_str=capabilities_str,
                    focus_guidance=next_focus,
                    context=context.get("previous_responses", []),
                )
            else:
                system_prompt = ANALYSIS_PROCESSING_SYSTEM_PROMPT
                # Analysis phase
                capability = context["capability"]
                discovered_functions = context.get("discovered_functions", [])
                next_focus = context.get("next_cycle_focus")

                functions_str = "\n".join(
                    f"- {func.name}: {func.description or ''}"
                    for func in discovered_functions
                )

                interrogation_prompt_request = ANALYSIS_PROMPT_TEMPLATE.format(
                    capability=capability,
                    functions_str=functions_str,
                    focus_guidance=next_focus,
                    context=context.get("previous_responses", []),
                )

        # Construct full prompt with system and user messages
        full_prompt = f"System: {system_prompt}\n\nUser: {interrogation_prompt_request}"

        # Generate response using HuggingFace pipeline
        result = self.pipe(
            full_prompt,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            num_return_sequences=1,
            **{
                k: v
                for k, v in self.config.llm.model_kwargs.items()
                if k
                not in [
                    "max_new_tokens",
                    "temperature",
                    "do_sample",
                    "num_return_sequences",
                ]
            },
        )

        # Extract generated text (remove the input prompt)
        generated_text = result[0]["generated_text"]
        if isinstance(generated_text, str):
            return generated_text[len(full_prompt) :].strip()
        else:
            raise ValueError("Unexpected response format from HuggingFace pipeline")

    async def process_discovery_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response during capability discovery phase."""
        self.output.print_verbose("[bold cyan]Discovery Response:[/bold cyan]")
        self.output.print_verbose(response)

        discovery_prompt = DISCOVERY_PROCESSING_PROMPT_TEMPLATE.format(
            json_format=DISCOVERY_JSON_SCHEMA, response=response, context=context
        )

        def extract_json(text: str) -> Dict[str, Any]:
            """Extract and parse JSON from text, handling various formats."""
            # First try direct JSON parsing
            text = text.strip()
            try:
                result = json.loads(text)
                if isinstance(result, dict):
                    return result
                else:
                    raise ValueError("Expected dictionary response from JSON")
            except json.JSONDecodeError:
                pass

            # Try to find JSON object pattern
            import re

            patterns = [
                # Standard JSON object
                r"\{[^{]*\}",
                # JSON with potential markdown code block
                r"```(?:json)?\s*(\{.*?\})\s*```",
                # Backup: any {...} content
                r"\{.*\}",
            ]

            for pattern in patterns:
                matches = re.finditer(pattern, text, re.DOTALL)
                for match in matches:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    try:
                        result = json.loads(json_str)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        continue

            raise ValueError("No valid JSON found in response")

        # First attempt - try to get a clean JSON response
        full_prompt = (
            f"System: {DISCOVERY_PROCESSING_SYSTEM_PROMPT}\n\nUser: {discovery_prompt}"
        )
        result = self.pipe(
            full_prompt,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            num_return_sequences=1,
            **{
                k: v
                for k, v in self.config.llm.model_kwargs.items()
                if k
                not in [
                    "max_new_tokens",
                    "temperature",
                    "do_sample",
                    "num_return_sequences",
                ]
            },
        )

        try:
            # Try to parse the response after removing the prompt
            generated_text = result[0]["generated_text"]
            if isinstance(generated_text, str):
                return extract_json(generated_text[len(full_prompt) :])
            else:
                raise ValueError("Unexpected response format from HuggingFace pipeline")
        except (json.JSONDecodeError, ValueError) as e:
            # If first attempt fails, try again with more explicit JSON formatting
            self.output.print_verbose(
                "[yellow]First attempt failed, trying again with explicit JSON formatting[/yellow]"
            )

            # Add explicit JSON formatting hints
            discovery_prompt = (
                "You MUST respond with valid JSON. No other text is allowed.\n"
                + discovery_prompt
            )
            full_prompt = (
                f"{DISCOVERY_PROCESSING_SYSTEM_PROMPT}\n\nUser: {discovery_prompt}"
            )

            result = self.pipe(
                full_prompt,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                num_return_sequences=1,
                **{
                    k: v
                    for k, v in self.config.llm.model_kwargs.items()
                    if k
                    not in [
                        "max_new_tokens",
                        "temperature",
                        "do_sample",
                        "num_return_sequences",
                    ]
                },
            )

            try:
                # Try to parse the full response
                generated_text = result[0]["generated_text"]
                if isinstance(generated_text, str):
                    return extract_json(generated_text[len(full_prompt) :])
                else:
                    raise ValueError(
                        "Unexpected response format from HuggingFace pipeline"
                    )
            except (json.JSONDecodeError, ValueError) as e2:
                # If both attempts fail, try one last time with the full generated text
                try:
                    full_text = result[0]["generated_text"]
                    if isinstance(full_text, str):
                        return extract_json(full_text)
                    else:
                        raise ValueError(
                            "Unexpected response format from HuggingFace pipeline"
                        )
                except (json.JSONDecodeError, ValueError) as e3:
                    self.output.print_verbose(
                        f"[red]Failed to extract JSON from response: {str(e3)}[/red]"
                    )
                    raise ValueError(
                        "Failed to extract valid JSON from model response after multiple attempts"
                    )

    async def process_analysis_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's response during capability analysis phase."""
        self.output.print_verbose("[bold cyan]Analysis Response:[/bold cyan]")
        self.output.print_verbose(response)

        capability = context["capability"]

        analysis_prompt = ANALYSIS_PROCESSING_PROMPT_TEMPLATE.format(
            capability_name=capability.name,
            json_format=ANALYSIS_JSON_SCHEMA,
            response=response,
        )

        # HuggingFace API call
        analysis_result = self.pipe(
            analysis_prompt,
            max_new_tokens=1024,
            do_sample=False,
            return_full_text=False,
        )

        try:
            # Extract generated text from result
            generated_text = analysis_result[0]["generated_text"]
            if not isinstance(generated_text, str):
                raise ValueError("Unexpected response format from HuggingFace pipeline")

            # Try to extract JSON from the generated text
            import re

            json_match = re.search(r"(\{.*\})", generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
                else:
                    raise ValueError("Expected dictionary response from JSON")
            else:
                raise ValueError("No JSON found in generated response")
        except (json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Failed to process analysis response: {str(e)}")
