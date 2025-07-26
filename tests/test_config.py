"""Tests for configuration module."""

import pytest
from pydantic import ValidationError

from agent_interrogator.config import (
    HuggingFaceConfig,
    InterrogationConfig,
    LLMConfig,
    ModelProvider,
    OutputMode,
)


class TestLLMConfig:
    """Test LLMConfig validation and behavior."""

    def test_openai_config_valid(self):
        """Test valid OpenAI configuration."""
        config = LLMConfig(
            provider=ModelProvider.OPENAI, model_name="gpt-4", api_key="test-key"
        )
        assert config.provider == ModelProvider.OPENAI
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"

    def test_openai_config_missing_api_key(self):
        """Test that OpenAI config works without API key (optional field)."""
        # API key is optional, so this should work
        config = LLMConfig(
            provider=ModelProvider.OPENAI, model_name="gpt-4", api_key=None
        )
        assert config.api_key is None

    def test_huggingface_config_valid(self):
        """Test valid HuggingFace configuration."""
        config = LLMConfig(
            provider=ModelProvider.HUGGINGFACE, model_name="mistralai/Mistral-7B-v0.1"
        )
        assert config.provider == ModelProvider.HUGGINGFACE
        assert config.model_name == "mistralai/Mistral-7B-v0.1"
        assert config.api_key is None  # Not required for HF

    def test_huggingface_with_options(self):
        """Test HuggingFace configuration with additional options."""
        hf_config = HuggingFaceConfig(
            device="cuda", quantization="fp16", allow_download=False
        )
        config = LLMConfig(
            provider=ModelProvider.HUGGINGFACE,
            model_name="test-model",
            huggingface=hf_config,
        )
        assert config.huggingface.device == "cuda"
        assert config.huggingface.quantization == "fp16"
        assert config.huggingface.allow_download is False

    def test_model_kwargs(self):
        """Test model kwargs are properly stored."""
        config = LLMConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            api_key="test-key",
            model_kwargs={"temperature": 0.7, "max_tokens": 2000},
        )
        assert config.model_kwargs["temperature"] == 0.7
        assert config.model_kwargs["max_tokens"] == 2000


class TestInterrogationConfig:
    """Test InterrogationConfig validation and behavior."""

    def test_default_values(self):
        """Test default configuration values."""
        llm_config = LLMConfig(
            provider=ModelProvider.OPENAI, model_name="gpt-4", api_key="test-key"
        )
        config = InterrogationConfig(llm=llm_config)

        assert config.max_iterations == 5
        assert config.output_mode == OutputMode.STANDARD

    def test_custom_values(self):
        """Test custom configuration values."""
        llm_config = LLMConfig(
            provider=ModelProvider.OPENAI, model_name="gpt-4", api_key="test-key"
        )
        config = InterrogationConfig(
            llm=llm_config, max_iterations=10, output_mode=OutputMode.VERBOSE
        )

        assert config.max_iterations == 10
        assert config.output_mode == OutputMode.VERBOSE

    def test_parse_obj_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "llm": {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key": "test-key",
                "model_kwargs": {"temperature": 0.5},
            },
            "max_iterations": 3,
            "output_mode": "quiet",
        }

        config = InterrogationConfig.model_validate(config_dict)
        assert config.llm.provider == ModelProvider.OPENAI
        assert config.llm.model_name == "gpt-4"
        assert config.llm.model_kwargs["temperature"] == 0.5
        assert config.max_iterations == 3
        assert config.output_mode == OutputMode.QUIET


class TestOutputMode:
    """Test OutputMode enum."""

    def test_enum_values(self):
        """Test all output mode values exist."""
        assert OutputMode.QUIET.value == "quiet"
        assert OutputMode.STANDARD.value == "standard"
        assert OutputMode.VERBOSE.value == "verbose"

    def test_from_string(self):
        """Test creating output mode from string."""
        assert OutputMode("quiet") == OutputMode.QUIET
        assert OutputMode("standard") == OutputMode.STANDARD
        assert OutputMode("verbose") == OutputMode.VERBOSE
