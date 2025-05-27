"""Main interrogator implementation."""

from typing import Any, Callable, Dict, List, Optional, Awaitable

from .config import InterrogationConfig, ModelProvider
from .llm import LLMInterface, OpenAILLM, HuggingFaceLLM
from .models import AgentProfile, Capability, Function

# Type alias for the agent interaction callback
AgentCallback = Callable[[str], Awaitable[str]]

class AgentInterrogator:
    """Main class for interrogating AI agents.
    
    Args:
        config (InterrogationConfig): Configuration for the interrogator
        agent_callback (AgentCallback): Async callback function that takes a prompt
            string and returns the agent's response string
    """
    
    def __init__(
        self,
        config: InterrogationConfig,
        agent_callback: AgentCallback
    ):
        self.config = config
        self.agent_callback = agent_callback
        self.llm = self._initialize_llm()
        self.profile = AgentProfile()

    def _initialize_llm(self) -> LLMInterface:
        """Initialize the appropriate LLM based on configuration."""
        if self.config.llm.provider == ModelProvider.OPENAI:
            return OpenAILLM(self.config)
        elif self.config.llm.provider == ModelProvider.HUGGINGFACE:
            return HuggingFaceLLM(self.config)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")

    async def interrogate(self) -> AgentProfile:
        """Perform the full interrogation process with iterative discovery and analysis."""
        # Iterative capability discovery
        capabilities = await self._discover_capabilities()
        self.profile.capabilities.extend(capabilities)

        # Iterative analysis of each capability
        for capability in self.profile.capabilities:
            await self._analyze_capability(capability)

        return self.profile

    async def _discover_capabilities(self) -> list[Capability]:
        """Discover high-level capabilities of the agent through multiple cycles."""
        discovered_capabilities = []
        cycle = 0
        previous_responses = []
        
        while True:
            context = {
                "phase": "discovery",
                "cycle": cycle,
                "previous_responses": previous_responses,
                "discovered_capabilities": [cap.dict() for cap in discovered_capabilities]
            }
            
            # Generate and send prompt
            prompt = await self.llm.generate_prompt(context)
            response = await self.agent_callback(prompt)
            previous_responses.append(response)
            
            # Process response
            result = await self.llm.process_response(response, context)
            
            # Add new capabilities
            new_capabilities = [Capability(**cap) for cap in result.get("capabilities", [])]
            discovered_capabilities.extend(new_capabilities)
            
            # Check if we should continue discovery
            if not await self.llm.should_continue_cycle(context, result):
                break
                
            cycle += 1
        
        return discovered_capabilities

    async def _analyze_capability(self, capability: Capability) -> None:
        """Analyze a specific capability in detail through multiple cycles."""
        cycle = 0
        previous_responses = []
        discovered_functions = []
        
        while True:
            context = {
                "phase": "analysis",
                "cycle": cycle,
                "previous_responses": previous_responses,
                "capability": capability.dict(),
                "discovered_functions": discovered_functions
            }
            
            # Generate and send prompt
            prompt = await self.llm.generate_prompt(context)
            response = await self.agent_callback(prompt)
            previous_responses.append(response)
            
            # Process response
            result = await self.llm.process_response(response, context)
            
            # Update capability with new information
            new_functions = [Function(**func) for func in result.get("functions", [])]
            discovered_functions.extend(new_functions)
            capability.functions.extend(new_functions)
            capability.metadata.update(result.get("metadata", {}))
            
            # Check if we should continue analysis
            if not await self.llm.should_continue_cycle(context, result):
                break
                
            cycle += 1
