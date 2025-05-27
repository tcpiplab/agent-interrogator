"""Tests for agent callback functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import aiohttp
from typing import AsyncGenerator

from agent_interrogator import AgentInterrogator, InterrogationConfig, LLMConfig
from agent_interrogator.models import AgentProfile, Capability

# Fixtures
@pytest.fixture
async def mock_llm_config() -> LLMConfig:
    """Create a mock LLM configuration."""
    return LLMConfig(
        provider="openai",
        model_name="gpt-4",
        api_key="test-key"
    )

@pytest.fixture
async def mock_config(mock_llm_config) -> InterrogationConfig:
    """Create a mock interrogator configuration."""
    return InterrogationConfig(
        llm=mock_llm_config,
        max_iterations=2
    )

@pytest.fixture
async def mock_callback() -> AsyncMock:
    """Create a mock callback function."""
    return AsyncMock(return_value="Mock agent response")

@pytest.fixture
async def interrogator(mock_config, mock_callback) -> AgentInterrogator:
    """Create an interrogator instance with mock components."""
    return AgentInterrogator(mock_config, mock_callback)

# Basic callback tests
async def test_callback_invocation(interrogator, mock_callback):
    """Test that the callback is invoked correctly."""
    await interrogator.interrogate()
    assert mock_callback.called
    assert mock_callback.call_count >= 1  # Should be called at least once

async def test_callback_args(interrogator, mock_callback):
    """Test that the callback receives correct argument types."""
    await interrogator.interrogate()
    for call in mock_callback.call_args_list:
        args, _ = call
        assert isinstance(args[0], str)  # First arg should be prompt string

# Error handling tests
async def test_callback_error():
    """Test handling of callback errors."""
    async def failing_callback(prompt: str) -> str:
        raise RuntimeError("Simulated callback failure")
    
    config = InterrogationConfig(
        llm=LLMConfig(provider="openai", model_name="gpt-4", api_key="test-key")
    )
    interrogator = AgentInterrogator(config, failing_callback)
    
    with pytest.raises(RuntimeError):
        await interrogator.interrogate()

# Integration tests with different callback types
async def test_http_callback():
    """Test HTTP callback integration."""
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={"response": "HTTP response"}
        )
        
        async def http_callback(prompt: str) -> str:
            async with aiohttp.ClientSession() as session:
                async with session.post("http://test") as response:
                    result = await response.json()
                    return result["response"]
        
        config = InterrogationConfig(
            llm=LLMConfig(provider="openai", model_name="gpt-4", api_key="test-key")
        )
        interrogator = AgentInterrogator(config, http_callback)
        profile = await interrogator.interrogate()
        
        assert isinstance(profile, AgentProfile)
        assert mock_post.called

async def test_queue_callback():
    """Test queue-based callback."""
    request_queue = asyncio.Queue()
    response_queue = asyncio.Queue()
    
    async def queue_processor():
        while True:
            prompt = await request_queue.get()
            await response_queue.put(f"Processed: {prompt}")
            request_queue.task_done()
    
    async def queue_callback(prompt: str) -> str:
        await request_queue.put(prompt)
        return await response_queue.get()
    
    # Start queue processor
    processor_task = asyncio.create_task(queue_processor())
    
    config = InterrogationConfig(
        llm=LLMConfig(provider="openai", model_name="gpt-4", api_key="test-key")
    )
    interrogator = AgentInterrogator(config, queue_callback)
    
    try:
        profile = await interrogator.interrogate()
        assert isinstance(profile, AgentProfile)
    finally:
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass

# Retry wrapper tests
async def test_retry_wrapper():
    """Test retry wrapper functionality."""
    call_count = 0
    
    async def failing_callback(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("Temporary failure")
        return "Success after retries"
    
    from examples.callbacks import with_retry
    retrying_callback = with_retry(failing_callback, max_retries=3, delay=0.1)
    
    config = InterrogationConfig(
        llm=LLMConfig(provider="openai", model_name="gpt-4", api_key="test-key")
    )
    interrogator = AgentInterrogator(config, retrying_callback)
    
    profile = await interrogator.interrogate()
    assert isinstance(profile, AgentProfile)
    assert call_count == 3  # Should succeed on third try

# Playwright callback tests
async def test_playwright_callback():
    """Test Playwright-based callback."""
    from examples.callbacks import PlaywrightCallback
    from unittest.mock import AsyncMock, patch
    
    # Mock Playwright's async_playwright
    with patch("playwright.async_api.async_playwright") as mock_playwright:
        # Setup mock browser and page
        mock_page = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.wait_for_selector.return_value.text_content = AsyncMock(
            return_value="Agent response"
        )
        
        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        
        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        
        mock_browser_type = AsyncMock()
        mock_browser_type.launch = AsyncMock(return_value=mock_browser)
        
        mock_playwright_instance = AsyncMock()
        mock_playwright_instance.chromium = mock_browser_type
        mock_playwright.return_value.start = AsyncMock(
            return_value=mock_playwright_instance
        )
        
        # Create callback instance
        callback = PlaywrightCallback(
            url="https://test.com",
            prompt_selector="#prompt",
            submit_selector="#submit",
            response_selector="#response"
        )
        
        # Create and run interrogator
        config = InterrogationConfig(
            llm=LLMConfig(provider="openai", model_name="gpt-4", api_key="test-key")
        )
        interrogator = AgentInterrogator(config, callback)
        
        # Run interrogation
        profile = await interrogator.interrogate()
        
        # Verify interactions
        assert isinstance(profile, AgentProfile)
        assert mock_page.fill.called
        assert mock_page.click.called
        assert mock_page.wait_for_selector.called
