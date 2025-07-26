"""Data models for representing agent capabilities and functions."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Parameter(BaseModel):
    """Represents a parameter for a function."""

    name: str
    type: str
    description: Optional[str] = None
    required: bool = True
    default: Optional[Any] = None


class Function(BaseModel):
    """Represents a function within a capability."""

    name: str
    description: Optional[str] = None
    parameters: List[Parameter] = Field(default_factory=list)
    return_type: Optional[str] = None


class Capability(BaseModel):
    """Represents a capability of the agent."""

    name: str
    description: Optional[str] = None
    functions: List[Function] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentProfile(BaseModel):
    """Complete profile of an agent's capabilities."""

    capabilities: List[Capability] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
