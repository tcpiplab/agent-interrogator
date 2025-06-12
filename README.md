# Agent Interrogator
<img src="assets/logo.webp" alt="Project Logo" width="200" style="vertical-align:middle; margin-right:10px;" />
A Python framework for systematically identifying and analyzing AI agent capabilities through automated interrogation. It supports iterative discovery and analysis cycles to exhaustively uncover all supported capabilities and function arguments.

## Features

- Automated discovery of agent capabilities and functions
- Iterative analysis with smart prompt adaptation
- Support for both OpenAI and HuggingFace models
- Multiple callback implementations for different agent types
- Browser automation support via Playwright
- Structured output for security tool integration
- Async-first design with robust error handling

## Installation

For basic installation:
```bash
pip install .
```

For development installation (includes testing and linting tools):
```bash
pip install .[dev]
```

## Quick Start

```python
import asyncio
from agent_interrogator import AgentInterrogator, InterrogationConfig, LLMConfig, ModelProvider

# Configure the interrogator
config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OPENAI,  # or ModelProvider.HUGGINGFACE
        model_name="gpt-4",
        api_key="your-api-key"
    ),
    max_iterations=5  # Maximum cycles for capability discovery
)

# Simple callback example
async def api_callback(prompt: str) -> str:
    # Implement your agent interaction logic here
    # response = await call_agent_api(prompt)
    return "Example response"

# Main async function
async def main():
    # Create and run the interrogator
    interrogator = AgentInterrogator(config, api_callback)
    profile = await interrogator.interrogate()

    # Access the discovered capabilities
    for capability in profile.capabilities:
        print(f"Capability: {capability.name}")
        print(f"Description: {capability.description}")
        for function in capability.functions:
            print(f"  Function: {function.name}")
            for param in function.parameters:
                print(f"    Parameter: {param.name} ({param.type})")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
```

## Callback Examples

The framework provides several callback implementations for different integration scenarios:

### Playwright Browser Automation
```python
from agent_interrogator import AgentInterrogator, InterrogationConfig
from agent_interrogator.callbacks import PlaywrightCallback

# Create a callback for web-based agents
callback = PlaywrightCallback(
    url="http://localhost:8501/",
    # CSS selectors for key elements
    prompt_selector='textarea[type="textarea"]',
    submit_selector='button[data-testid="stChatInputSubmitButton"]',
    response_selector='div[data-testid="stMarkdownPre"] code',
    # Browser configuration
    browser_type="chromium",  # or "firefox", "webkit"
    headless=True,
    # Optional settings
    timeout=30000,  # milliseconds
    wait_for_network_idle=True
)

# Use with interrogator
interrogator = AgentInterrogator(config, callback)
profile = await interrogator.interrogate()
```

### HTTP API Integration
```python
import aiohttp
from typing import Optional

class HTTPCallback:
    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __call__(self, prompt: str) -> str:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        async with self.session.post(
            self.endpoint,
            json={"prompt": prompt},
            headers=self.headers
        ) as response:
            result = await response.json()
            return result["response"]
    
    async def cleanup(self):
        if self.session:
            await self.session.close()
            self.session = None
```

### WebSocket Real-time Communication
```python
import json
import aiohttp
from typing import Optional

class WebSocketCallback:
    def __init__(self, ws_url: str):
        self.url = ws_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
    
    async def __call__(self, prompt: str) -> str:
        if not self.ws:
            self.session = aiohttp.ClientSession()
            self.ws = await self.session.ws_connect(self.url)
        
        await self.ws.send_json({"type": "prompt", "content": prompt})
        response = await self.ws.receive_json()
        return response["content"]
    
    async def cleanup(self):
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
            self.session = self.ws = None
```

### Retry Wrapper with Backoff
```python
from functools import wraps
import asyncio
from typing import Any, Callable, TypeVar

T = TypeVar('T')

def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
            
            assert last_exception is not None
            raise last_exception
        return wrapper
    return decorator

# Usage example
@with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
async def api_callback(prompt: str) -> str:
    return await make_api_call(prompt)

interrogator = AgentInterrogator(config, api_callback)
```

## Configuration

The framework can be configured via a YAML file or directly in code:

```yaml
# config.yaml
llm:
  provider: openai
  model_name: gpt-4
  api_key: ${OPENAI_API_KEY}  # Uses environment variable
  model_kwargs:
    temperature: 0.7
    max_tokens: 2000

max_iterations: 5  # Maximum discovery cycles
```

```python
# Or in code
from agent_interrogator import InterrogationConfig, LLMConfig, ModelProvider

config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 2000
        }
    ),
    max_iterations=5
)
```

## Development

### Setup
```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run type checking
mypy src/

# Run formatters
black src/ tests/
isort src/ tests/
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
