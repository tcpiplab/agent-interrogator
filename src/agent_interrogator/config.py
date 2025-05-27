"""Configuration schema for the agent interrogator."""

from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class LLMConfig(BaseModel):
    """Configuration for the LLM to be used for interrogation."""
    provider: ModelProvider = Field(
        ...,
        description="The provider of the LLM (OpenAI or HuggingFace)"
    )
    model_name: str = Field(
        ...,
        description="Name of the model to use"
    )
    api_key: Optional[str] = Field(
        None,
        description="API key for the provider (if required)"
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model-specific parameters"
    )


class InterrogationConfig(BaseModel):
    """Main configuration for the agent interrogator."""
    llm: LLMConfig = Field(
        ...,
        description="Configuration for the LLM to use for interrogation"
    )
    max_iterations: int = Field(
        default=5,
        description="Maximum number of iterations for capability discovery"
    )
    output_format: str = Field(
        default="json",
        description="Format for the output (json/yaml)"
    )
