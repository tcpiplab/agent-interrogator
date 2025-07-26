"""Tests for data models."""

import json
from datetime import datetime

import pytest

from agent_interrogator.models import AgentProfile, Capability, Function, Parameter


class TestParameter:
    """Test Parameter model."""

    def test_parameter_required(self):
        """Test creating a required parameter."""
        param = Parameter(
            name="file_path",
            type="string",
            description="Path to the file",
            required=True,
        )
        assert param.name == "file_path"
        assert param.type == "string"
        assert param.description == "Path to the file"
        assert param.required is True
        assert param.default is None

    def test_parameter_optional_with_default(self):
        """Test creating an optional parameter with default."""
        param = Parameter(
            name="max_results",
            type="integer",
            description="Maximum number of results",
            required=False,
            default=10,
        )
        assert param.name == "max_results"
        assert param.required is False
        assert param.default == 10

    def test_parameter_json_serialization(self):
        """Test parameter JSON serialization."""
        param = Parameter(
            name="query", type="string", description="Search query", required=True
        )
        json_str = param.model_dump_json()
        data = json.loads(json_str)
        assert data["name"] == "query"
        assert data["type"] == "string"
        assert data["required"] is True


class TestFunction:
    """Test Function model."""

    def test_function_basic(self):
        """Test creating a basic function."""
        func = Function(name="search_web", description="Search the web for information")
        assert func.name == "search_web"
        assert func.description == "Search the web for information"
        assert func.parameters == []
        assert func.return_type is None

    def test_function_with_parameters(self):
        """Test creating a function with parameters."""
        params = [
            Parameter(name="query", type="string", required=True),
            Parameter(name="limit", type="integer", required=False, default=10),
        ]
        func = Function(
            name="search",
            description="Perform a search",
            parameters=params,
            return_type="list[SearchResult]",
        )
        assert len(func.parameters) == 2
        assert func.parameters[0].name == "query"
        assert func.parameters[1].name == "limit"
        assert func.return_type == "list[SearchResult]"

    def test_function_equality(self):
        """Test function equality comparison."""
        func1 = Function(name="test", description="Test function")
        func2 = Function(name="test", description="Test function")
        func3 = Function(name="other", description="Other function")

        assert func1.model_dump() == func2.model_dump()
        assert func1.model_dump() != func3.model_dump()


class TestCapability:
    """Test Capability model."""

    def test_capability_basic(self):
        """Test creating a basic capability."""
        cap = Capability(name="file_operations", description="File system operations")
        assert cap.name == "file_operations"
        assert cap.description == "File system operations"
        assert cap.functions == []
        assert cap.metadata == {}

    def test_capability_with_functions(self):
        """Test creating a capability with functions."""
        functions = [
            Function(name="read_file", description="Read a file"),
            Function(name="write_file", description="Write to a file"),
        ]
        cap = Capability(
            name="file_ops",
            description="File operations",
            functions=functions,
            metadata={"version": "1.0"},
        )
        assert len(cap.functions) == 2
        assert cap.functions[0].name == "read_file"
        assert cap.metadata["version"] == "1.0"


class TestAgentProfile:
    """Test AgentProfile model."""

    def test_profile_empty(self):
        """Test creating an empty profile."""
        profile = AgentProfile()
        assert profile.capabilities == []
        assert isinstance(profile.metadata, dict)
        # The model doesn't automatically set timestamp

    def test_profile_with_capabilities(self):
        """Test creating a profile with capabilities."""
        cap1 = Capability(
            name="web_search",
            functions=[Function(name="search", description="Search web")],
        )
        cap2 = Capability(
            name="file_ops", functions=[Function(name="read", description="Read file")]
        )

        profile = AgentProfile(
            capabilities=[cap1, cap2], metadata={"agent_name": "TestBot"}
        )

        assert len(profile.capabilities) == 2
        assert profile.capabilities[0].name == "web_search"
        assert profile.metadata["agent_name"] == "TestBot"

    def test_profile_json_export(self):
        """Test exporting profile to JSON."""
        param = Parameter(name="path", type="string", required=True)
        func = Function(name="read", parameters=[param])
        cap = Capability(name="file_ops", functions=[func])
        profile = AgentProfile(capabilities=[cap])

        json_str = profile.model_dump_json(indent=2)
        data = json.loads(json_str)

        assert len(data["capabilities"]) == 1
        assert data["capabilities"][0]["name"] == "file_ops"
        assert len(data["capabilities"][0]["functions"]) == 1
        assert data["capabilities"][0]["functions"][0]["name"] == "read"

    def test_profile_timestamp(self):
        """Test profile can have timestamp in metadata."""
        from datetime import datetime

        timestamp = datetime.now().isoformat()
        profile = AgentProfile(metadata={"interrogation_timestamp": timestamp})

        # Verify we can set and retrieve timestamp
        assert profile.metadata["interrogation_timestamp"] == timestamp
