"""Configuration schema for the agent interrogator."""

from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class OutputMode(str, Enum):
    """Terminal output modes for the interrogator."""
    QUIET = "quiet"      # No terminal output
    STANDARD = "standard"  # Basic info and status indicators
    VERBOSE = "verbose"   # Detailed prompts and responses


class HuggingFaceConfig(BaseModel):
    """HuggingFace-specific configuration options."""
    local_model_path: Optional[str] = Field(
        None,
        description="Path to local model directory. If set, this will be used instead of model_name"
    )
    allow_download: bool = Field(
        True,
        description="Whether to allow downloading models from HuggingFace Hub"
    )
    revision: Optional[str] = Field(
        None,
        description="Model revision/tag to use (e.g., 'main')"
    )
    device: str = Field(
        "auto",
        description="Device to place model on ('cpu', 'cuda', 'auto', 'ane'). 'ane' uses Apple Neural Engine on M1/M2/M3 Macs if available"
    )
    quantization: Optional[str] = Field(
        None,
        description="Quantization method to use (e.g., 'int8', 'fp16', None for no quantization)"
    )


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
    huggingface: Optional[HuggingFaceConfig] = Field(
        None,
        description="HuggingFace-specific configuration options"
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
    # TODO: Implement support for different output formats (json/yaml)
    # This will allow users to control how the agent profile and capabilities
    # are serialized in the final output
    output_format: str = Field(
        default="json",
        description="Format for the output (json/yaml)"
    )
    output_mode: OutputMode = Field(
        default=OutputMode.STANDARD,
        description="Controls the level of terminal output during interrogation"
    )
