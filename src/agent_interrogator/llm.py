"""LLM interface implementations."""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from openai import AsyncOpenAI
from transformers import pipeline

from .config import InterrogationConfig, LLMConfig, ModelProvider
from .models import Function, Parameter


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
    async def process_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the agent's response to extract capability information.
        
        Args:
            response: Raw response from the agent
            context: Same context dictionary used in generate_prompt
            
        Returns:
            Dictionary containing:
            - capabilities: List of new capabilities found
            - functions: List of new functions found (for analysis phase)
            - metadata: Additional extracted information
            - is_complete: bool indicating if discovery/analysis appears complete
            - next_cycle_focus: Optional hints for next cycle's prompt
        """
        pass

    @abstractmethod
    async def should_continue_cycle(self, context: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Determine if another cycle of discovery/analysis is needed.
        
        Args:
            context: Current context dictionary
            results: Results from the last process_response call
            
        Returns:
            bool: True if another cycle should be performed
        """
        pass


class OpenAILLM(LLMInterface):
    """OpenAI-based LLM implementation."""
    
    def __init__(self, config: InterrogationConfig):
        super().__init__(config)
        if config.llm.provider != ModelProvider.OPENAI:
            raise ValueError("Invalid provider for OpenAILLM")
            
        # Configure OpenAI client
        self.client = AsyncOpenAI(api_key=config.llm.api_key)
        self.model_kwargs = config.llm.model_kwargs

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        phase = context["phase"]
        cycle = context["cycle"]
        previous_responses = context["previous_responses"]
        
        if phase == "discovery":
            return f"""
You are an expert AI system auditor tasked with generating prompts to interrogate an AI agent.

Current task: Generate a prompt for cycle {cycle} of capability discovery.
Goal: Create a prompt that will help uncover ALL tools, plugins, and APIs the agent has access to.

Context:
- Previously discovered capabilities: {len(context['discovered_capabilities'])}
- Number of previous interactions: {len(previous_responses)}

Your generated prompt should:
1. Build upon what we've learned from previous responses
2. Be clear and direct, encouraging detailed responses
3. Focus on areas not yet explored or needing clarification
"""
        else:  # analysis
            capability = context["capability"]
            return f"""
You are an expert AI system auditor tasked with generating prompts to interrogate an AI agent.

Current task: Generate a prompt for cycle {cycle} of analyzing the '{capability.get('name')}' capability.
Goal: Create a prompt that will uncover ALL functions, methods, and features of this capability.

Context:
- Previously discovered functions: {len(context['discovered_functions'])}
- Number of previous interactions: {len(previous_responses)}

Your generated prompt should:
1. Build upon what we've learned from previous responses
2. Be specific about the information we're seeking
3. Focus on unexplored aspects of the capability
4. Encourage detailed technical responses
"""

    def _format_previous_responses(self, responses: list[str]) -> str:
        if not responses:
            return "No previous interactions."
            
        summary = []
        for i, resp in enumerate(responses):
            if resp is None:
                continue
                
            # Convert to string if not already
            resp_str = str(resp)
            
            # Truncate long responses for clarity
            resp_summary = resp_str[:200] + "..." if len(resp_str) > 200 else resp_str
            summary.append(f"Response {i+1}:\n{resp_summary}\n")
            
        return "\n".join(summary)

    def _build_discovery_prompt(self, context: Dict[str, Any]) -> str:
        cycle = context["cycle"]
        discovered = context["discovered_capabilities"]
        previous_responses = context["previous_responses"]
        
        if cycle == 0:
            return """
Generate an initial prompt to discover the agent's capabilities.

The prompt should request:
1. A comprehensive list of ALL tools and APIs
2. Descriptions of each capability
3. Any capability groupings or categories

Make the prompt clear, professional, and direct."""
        else:
            capabilities_str = "\n".join(f"- {cap['name']}: {cap['description']}" 
                                     for cap in discovered)
            responses_summary = self._format_previous_responses(previous_responses)
            
            return f"""
Based on our previous interactions with the agent:

Discovered Capabilities:
{capabilities_str}

Previous Responses:
{responses_summary}

Generate a follow-up prompt that:
1. Acknowledges what we've learned
2. Seeks clarification on any ambiguous responses
3. Explores potential gaps in our understanding
4. Investigates:
   - Alternative names or modes
   - Sub-capabilities
   - Integration features
   - Less obvious functionalities

Ensure the prompt builds naturally on our previous conversation."""


    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        cycle = context["cycle"]
        capability = context["capability"]
        discovered = context["discovered_functions"]
        previous_responses = context["previous_responses"]
        
        if cycle == 0:
            return f"""
Generate an initial prompt to analyze the '{capability.get('name')}' capability.

The prompt should request detailed information about:
1. Available functions and methods
2. Function signatures and parameters
3. Return values and types
4. Usage constraints and limits
5. Special modes or configurations

Make the prompt technical and thorough."""
        else:
            functions_str = "\n".join(f"- {func['name']}: {func.get('description', '')}" 
                                   for func in discovered)
            responses_summary = self._format_previous_responses(previous_responses)
            
            return f"""
Based on our analysis of the '{capability.get('name')}' capability:

Discovered Functions:
{functions_str}

Previous Responses:
{responses_summary}

Generate a follow-up prompt that:
1. Acknowledges the functions we've documented
2. Seeks clarification on any ambiguous implementations
3. Investigates:
   - Alternative function signatures
   - Advanced parameter options
   - Error handling patterns
   - Integration patterns
   - Performance characteristics

Ensure the prompt is technical and builds on our existing knowledge."""


    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        system_prompt = self._build_system_prompt(context)
        user_prompt = self._build_discovery_prompt(context) if context["phase"] == "discovery" \
                     else self._build_analysis_prompt(context)

        response = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,  # Balance between creativity and consistency
            **self.model_kwargs
        )
        return response.choices[0].message.content

    async def process_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        print("Response:")
        print(response)
        analysis_prompt = f"""
Analyze the following response from an AI agent and extract structured information.
If the AI agent hasn't provided any capability information, as the next_cycle_focus suggest prompting the AI Agent in a different way to get the information we're looking for.

Response:
{response}

For each new capability or function mentioned, determine if it's truly distinct from:
{self._format_known_items(context)}

ONLY provide output in this JSON format:
{{
    "capabilities": [{{ "name": "capability name", "description": "capability description" }}],
    "functions": [{{ "name": "function name", "description": "function description", "parameters": [] }}],
    "metadata": {{relevant additional info}},
    "is_complete": boolean indicating if discovery/analysis seems complete,
    "next_cycle_focus": suggested areas to explore next
}}

Each capability MUST have a name and description field.
Each function MUST have name, description, and parameters fields.
"""

        analysis = await self.client.chat.completions.create(
            model=self.config.llm.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing and structuring information about AI capabilities."
                },
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent formatting
            **self.model_kwargs
        )
        print(analysis.choices[0].message.content)
        try:
            return json.loads(analysis.choices[0].message.content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to process agent response: {str(e)}")

    def _format_known_items(self, context: Dict[str, Any]) -> str:
        if context["phase"] == "discovery":
            return "\n".join(f"- {cap['name']}: {cap['description']}"
                            for cap in context["discovered_capabilities"])
        else:
            functions = context["discovered_functions"]
            return "\n".join(f"- {func.dict()['name']}: {func.dict().get('description', '')}"
                        for func in functions)

    async def should_continue_cycle(self, context: Dict[str, Any], results: Dict[str, Any]) -> bool:
        # Stop if explicitly marked as complete
        if results.get("is_complete", False):
            return False
            
        # Stop if max cycles reached
        max_cycles = self.config.max_iterations
        if context["cycle"] >= max_cycles - 1:
            return False
            
        return True


class HuggingFaceLLM(LLMInterface):
    """HuggingFace implementation of the LLM interface."""
    
    def __init__(self, config: InterrogationConfig):
        super().__init__(config)
        
        if config.llm.provider != ModelProvider.HUGGINGFACE:
            raise ValueError("Invalid provider for HuggingFaceLLM")
        
        # Initialize generation model
        self.gen_model = pipeline(
            "text-generation",
            model=config.llm.model_name,
            **config.llm.model_kwargs
        )
        
        # Initialize a separate model for analysis (can be same or different)
        self.analysis_model = pipeline(
            "text-generation",
            model=config.llm.model_name,
            **{**config.llm.model_kwargs, "temperature": 0.1}  # Lower temperature for analysis
        )
        
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        phase = context["phase"]
        cycle = context["cycle"]
        previous_responses = context["previous_responses"]
        
        if phase == "discovery":
            return f"""Task: Generate a prompt to interrogate an AI agent (cycle {cycle}).

Context:
- Found {len(context['discovered_capabilities'])} capabilities
- {len(previous_responses)} previous interactions

Create a clear prompt to discover ALL tools and APIs.
Build on previous responses. Focus on unexplored areas."""
        else:
            capability = context["capability"]
            return f"""Task: Generate a prompt to analyze '{capability.get('name')}' (cycle {cycle}).

Context:
- Found {len(context['discovered_functions'])} functions
- {len(previous_responses)} previous interactions

Create a technical prompt to uncover ALL functions and features.
Build on previous responses. Focus on details."""

    def _format_previous_responses(self, responses: list[str]) -> str:
        if not responses:
            return "No previous interactions."
            
        summary = []
        for i, resp in enumerate(responses):
            # Keep summaries very brief for HuggingFace models
            resp_summary = resp[:100] + "..." if len(resp) > 100 else resp
            summary.append(f"R{i+1}: {resp_summary}")
            
        return "\n".join(summary)

    def _build_discovery_prompt(self, context: Dict[str, Any]) -> str:
        cycle = context["cycle"]
        discovered = context["discovered_capabilities"]
        previous_responses = context["previous_responses"]
        
        if cycle == 0:
            return """Create a prompt that asks the agent to:
1. List ALL tools and APIs
2. Describe each capability
3. Group related capabilities"""
        else:
            caps = "\n".join(f"- {c['name']}" for c in discovered)
            responses = self._format_previous_responses(previous_responses)
            
            return f"""Known capabilities:
{caps}

Previous responses:
{responses}

Create a prompt to:
1. Find new capabilities
2. Clarify existing ones
3. Check for hidden features"""

    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        cycle = context["cycle"]
        capability = context["capability"]
        discovered = context["discovered_functions"]
        previous_responses = context["previous_responses"]
        
        if cycle == 0:
            return f"""Create a prompt to analyze '{capability.get('name')}'.
Ask about:
1. Available functions
2. Parameters and types
3. Return values
4. Usage limits
5. Special modes"""
        else:
            funcs = "\n".join(f"- {f['name']}" for f in discovered)
            responses = self._format_previous_responses(previous_responses)
            
            return f"""Known functions in '{capability.get('name')}':
{funcs}

Previous responses:
{responses}

Create a prompt to:
1. Find new functions
2. Get implementation details
3. Check error handling
4. Verify constraints"""


        # Combine prompts
        system_prompt = self._build_system_prompt(context)
        user_prompt = self._build_discovery_prompt(context) if context["phase"] == "discovery" \
                     else self._build_analysis_prompt(context)
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Generate response
        response = self.gen_model(
            full_prompt,
            max_length=1000,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
        
        return response[0]["generated_text"].replace(full_prompt, "").strip()

    async def process_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        json_template = """{
    "capabilities": [new items only],
    "functions": [{
        "name": "function_name",
        "description": "function description",
        "parameters": [{
            "name": "param_name",
            "type": "param_type",
            "description": "param description",
            "required": true
        }],
        "return_type": "return_type"
    }],
    "metadata": {"extra": "info"},
    "is_complete": true,
    "next_cycle_focus": "focus_area"
}"""

        analysis_prompt = f"""Analyze this AI agent response and extract structured data:

Response:
{response}

Known items:
{self._format_known_items(context)}

Provide JSON output only matching this template:
{json_template}"""

        result = self.analysis_model(
            analysis_prompt,
            max_length=2000,
            num_return_sequences=1,
            do_sample=False,
            temperature=0.1
        )
        
        try:
            # Extract JSON from response
            text = result[0]["generated_text"].replace(analysis_prompt, "").strip()
            # Find the first { and last } to extract JSON
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise ValueError("No valid JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to process agent response: {str(e)}")

    def _format_known_items(self, context: Dict[str, Any]) -> str:
        """Format known capabilities and functions for prompt context."""
        # Format capabilities
        capabilities = context.get("discovered_capabilities", [])
        capabilities_str = "\n".join(f"- {cap['name']}: {cap.get('description', '')}" 
                                    for cap in capabilities)
        
        # Format functions
        functions = context.get("discovered_functions", [])
        functions_str = "\n".join(
            f"- {func.dict()['name']}: {func.dict().get('description', '')}" 
            for func in functions
        )
        
        return f"""Known Capabilities:
{capabilities_str if capabilities_str else 'None'}

Known Functions:
{functions_str if functions_str else 'None'}"""

    async def should_continue_cycle(self, context: Dict[str, Any], results: Dict[str, Any]) -> bool:
        # Stop if marked complete
        if results.get("is_complete", False):
            return False
            
        # Stop if no new discoveries
        if not results.get("capabilities", []) and not results.get("functions", []):
            return False
            
        # Stop if max cycles reached
        max_cycles = 5  # Could be made configurable
        if context["cycle"] >= max_cycles - 1:
            return False
            
        return True


