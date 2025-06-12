"""LLM interface implementations."""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Awaitable, Callable

from openai import AsyncOpenAI
from transformers import pipeline

from .config import InterrogationConfig, LLMConfig, ModelProvider
from .models import Function, Parameter
from .prompt_templates import (
    # Discovery templates
    INITIAL_DISCOVERY_PROMPT, DISCOVERY_PROMPT_TEMPLATE,
    # Analysis templates
    INITIAL_ANALYSIS_PROMPT_TEMPLATE, ANALYSIS_PROMPT_TEMPLATE,
    # Processing templates
    DISCOVERY_PROCESSING_SYSTEM_PROMPT, DISCOVERY_PROCESSING_PROMPT_TEMPLATE,
    ANALYSIS_PROCESSING_SYSTEM_PROMPT, ANALYSIS_PROCESSING_PROMPT_TEMPLATE,
    # Schema templates
    DISCOVERY_JSON_SCHEMA, ANALYSIS_JSON_SCHEMA,
    # Formatting templates
    KNOWN_ITEMS_TEMPLATE
)


class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""
    
    def __init__(self, config: InterrogationConfig):
        """Initialize the LLM interface.
        
        Args:
            config: Full interrogation configuration
        """
        self.config = config
    
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
    async def process_discovery_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
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
    async def process_analysis_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
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
        
    async def process_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
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
        capabilities_str = "\\n".join(
            f"- {cap.name}: {cap.description or ''}" 
            for cap in capabilities
        ) or "None"
        
        # Format functions
        functions = context.get("discovered_functions", [])
        functions_str = "\\n".join(
            f"- {func.name}: {func.description or ''}" 
            for func in functions
        ) or "None"
        
        return KNOWN_ITEMS_TEMPLATE.format(
            capabilities_str=capabilities_str,
            functions_str=functions_str
        )
    
    async def should_continue_cycle(self, results: Dict[str, Any]) -> bool:
        """Determine if another cycle should be run based on results."""
        # Stop if marked complete
        if results.get("is_complete", False):
            return False
            
        return True


class OpenAILLM(LLMInterface):
    """OpenAI-based LLM implementation."""
    
    def __init__(self, config: InterrogationConfig):
        super().__init__(config)
        if config.llm.provider != ModelProvider.OPENAI:
            raise ValueError("Invalid provider for OpenAILLM")
            
        # Configure OpenAI client
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
                    context=context.get("previous_responses", [])
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
                    context=context.get("previous_responses", [])
                )
        
        interrogation_prompt = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": interrogation_prompt_request}
            ],
            temperature=0.1,
            **self.model_kwargs
        )
        return interrogation_prompt.choices[0].message.content

    async def process_discovery_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the agent's response during capability discovery phase."""
        discovery_prompt = DISCOVERY_PROCESSING_PROMPT_TEMPLATE.format(
            json_format=DISCOVERY_JSON_SCHEMA,
            context=context,
            response=response
        )
        print("Discovery Prompt:")
        print(discovery_prompt)
        
        discovery = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=[
                {
                    "role": "system",
                    "content": DISCOVERY_PROCESSING_SYSTEM_PROMPT
                },
                {"role": "user", "content": discovery_prompt}
            ],
            temperature=0.1,
            **self.model_kwargs
        )
        
        try:
            return json.loads(discovery.choices[0].message.content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to process discovery response: {str(e)}")

    async def process_analysis_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the agent's response during capability analysis phase."""
        analysis_prompt = ANALYSIS_PROCESSING_PROMPT_TEMPLATE.format(
            capability_name=context["capability"].name,
            json_format=ANALYSIS_JSON_SCHEMA,
            response=response,
            context=context
        )

        print("Analysis Response:")
        print(analysis_prompt)
        
        # First attempt - try to get a clean JSON response
        analysis = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=[
                {
                    "role": "system",
                    "content": ANALYSIS_PROCESSING_SYSTEM_PROMPT
                },
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1,
            **self.model_kwargs
        )
        
        content = analysis.choices[0].message.content.strip()
        
        try:
            # Try to parse the raw content first
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the content
            import re
            json_matches = re.findall(r'\{[^{}]*\}', content)
            
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
                        "content": "You MUST respond with ONLY a valid JSON object following the schema exactly. No other text or explanation."
                    },
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                **self.model_kwargs
            )
            
            content = analysis.choices[0].message.content.strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print("Failed to parse JSON response. Raw content:")
                print(content)
                raise ValueError(f"Failed to process analysis response after multiple attempts: {str(e)}")


class HuggingFaceLLM(LLMInterface):
    """HuggingFace-based LLM implementation."""
    
    def __init__(self, config: InterrogationConfig):
        super().__init__(config)
        if config.llm.provider != ModelProvider.HUGGINGFACE:
            raise ValueError("Invalid provider for HuggingFaceLLM")
            
        # Configure HuggingFace pipeline
        self.pipe = pipeline(
            "text-generation",
            model=config.llm.model_name,
            **config.llm.model_kwargs
        )

    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        """Generate an interrogation prompt based on context."""
        phase = context.get("phase", "discovery")
        cycle = context.get("cycle", 0)
        
        if phase == "discovery":
            if cycle == 0:
                return INITIAL_DISCOVERY_PROMPT
            
            discovered = context.get("discovered_capabilities", [])
            next_focus = context.get("next_cycle_focus")
            
            capabilities_str = "\\n".join(
                f"- {cap.name}: {cap.description or ''}" 
                for cap in discovered
            )
            
            focus_guidance = f'Based on previous exploration, I should focus on: {next_focus}' if next_focus else ''
            
            return DISCOVERY_PROMPT_TEMPLATE.format(
                capabilities_str=capabilities_str,
                focus_guidance=focus_guidance
            )
        else:
            capability = context["capability"]
            discovered_functions = context.get("discovered_functions", [])
            next_focus = context.get("next_cycle_focus")
            
            functions_str = "\\n".join(
                f"- {func.name}: {func.description or ''}" 
                for func in discovered_functions
            )
            
            if cycle == 0:
                return INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(
                    capability_name=capability.name
                )
            else:
                focus_guidance = f'Based on previous analysis, I should focus on: {next_focus}' if next_focus else ''
                
                return ANALYSIS_PROMPT_TEMPLATE.format(
                    capability_name=capability.name,
                    functions_str=functions_str,
                    focus_guidance=focus_guidance
                )

    async def process_discovery_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the agent's response during capability discovery phase."""
        print("Discovery Response:")
        print(response)
        
        discovery_prompt = DISCOVERY_PROCESSING_PROMPT_TEMPLATE.format(
            json_format=DISCOVERY_JSON_SCHEMA,
            response=response
        )
        
        # HuggingFace API call
        discovery_result = self.pipe(
            discovery_prompt,
            max_new_tokens=1024,
            do_sample=False,
            return_full_text=False
        )
        
        try:
            # Extract generated text from result
            generated_text = discovery_result[0]['generated_text']
            # Try to extract JSON from the generated text
            import re
            json_match = re.search(r'(\{.*\})', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in generated response")
        except (json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Failed to process discovery response: {str(e)}")

    async def process_analysis_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the agent's response during capability analysis phase."""
        print("Analysis Response:")
        print(response)
        
        capability = context["capability"]
        
        analysis_prompt = ANALYSIS_PROCESSING_PROMPT_TEMPLATE.format(
            capability_name=capability.name,
            json_format=ANALYSIS_JSON_SCHEMA,
            response=response
        )
        
        # HuggingFace API call
        analysis_result = self.pipe(
            analysis_prompt,
            max_new_tokens=1024,
            do_sample=False,
            return_full_text=False
        )
        
        try:
            # Extract generated text from result
            generated_text = analysis_result[0]['generated_text']
            # Try to extract JSON from the generated text
            import re
            json_match = re.search(r'(\{.*\})', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in generated response")
        except (json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Failed to process analysis response: {str(e)}")
