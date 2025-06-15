# Agent Interrogator

<p align="center">
  <img src="assets/logo.webp" alt="Project Logo" width="400" />
</p>
A Python framework for systematically identifying and analyzing AI agent capabilities through automated interrogation. It supports iterative discovery and analysis cycles to exhaustively uncover all supported capabilities and function arguments.

## Features

- Automated discovery of agent capabilities and functions
- Iterative analysis with smart prompt adaptation
- Support for both OpenAI and HuggingFace models
- Multiple callback implementations for different agent types
- Browser automation support via Playwright
- Structured output for security tool integration
- Async-first design with robust error handling
- Configurable terminal output with quiet, standard, and verbose modes

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

The framework can be configured via a YAML file or directly in code. It supports the following output modes:

- `quiet`: No terminal output
- `standard`: Shows startup logo, configuration info, and high-level status (default)
- `verbose`: Includes all standard output plus detailed logs of prompts, responses, and process results


```yaml
# Configuration Reference

The Agent Interrogator can be configured either through a YAML file or programmatically. Here's a complete reference of all available configuration options:

```yaml
# config.yaml
llm:
  # Required: LLM provider (openai or huggingface)
  provider: openai
  
  # Required: Model name or path
  model_name: gpt-4  # For OpenAI
  # model_name: mistralai/Mistral-7B-v0.1  # For HuggingFace
  
  # Required for OpenAI: API key
  api_key: ${OPENAI_API_KEY}  # Uses environment variable
  
  # Optional: Provider-specific settings
  huggingface:
    # Optional: Path to local model (alternative to model_name)
    local_model_path: /path/to/model
    
    # Optional: Device placement (auto, cpu, cuda, mps, ane)
    device: auto
    
    # Optional: Model quantization (fp16, int8)
    quantization: fp16
    
    # Optional: Allow downloading models from HF Hub
    allow_download: true
    
    # Optional: Model revision/tag
    revision: main
  
  # Optional: Model-specific parameters
  model_kwargs:
    temperature: 0.7
    max_tokens: 2000
    # ... any other model-specific parameters

# Optional: Maximum discovery cycles (default: 5)
max_iterations: 5

# Optional: Terminal output mode (quiet, standard, verbose)
output_mode: standard
```

## Using Configuration in Code

```python
from agent_interrogator import (
    InterrogationConfig, LLMConfig, ModelProvider,
    OutputMode, HuggingFaceConfig
)

# OpenAI Configuration
config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key",
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 2000
        }
    ),
    max_iterations=5,
    output_mode=OutputMode.STANDARD
)

# HuggingFace Configuration
config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.HUGGINGFACE,
        model_name="mistralai/Mistral-7B-v0.1",
        huggingface=HuggingFaceConfig(
            device="ane",  # Use Apple Neural Engine on M1/M2/M3 Macs
            quantization="fp16",
            allow_download=True,
            revision="main"
        )
    ),
    max_iterations=5,
    output_mode=OutputMode.VERBOSE
)

# Create and run the interrogator
interrogator = AgentInterrogator(config, callback)
profile = await interrogator.interrogate()
```

## Configuration Options

### LLM Provider Settings

#### OpenAI
- `provider`: Must be `openai`
- `model_name`: Name of the OpenAI model (e.g., `gpt-4`, `gpt-3.5-turbo`)
- `api_key`: Your OpenAI API key
- `model_kwargs`: Additional parameters for the OpenAI API

#### HuggingFace
- `provider`: Must be `huggingface`
- `model_name`: Model name from HuggingFace Hub or path to local model
- `huggingface`: Provider-specific settings:
  - `local_model_path`: Optional path to locally downloaded model
  - `device`: Model device placement (`auto`, `cpu`, `cuda`, `mps`, `ane`)
  - `quantization`: Model quantization (`fp16`, `int8`)
  - `allow_download`: Whether to allow downloading models from HF Hub
  - `revision`: Model revision/tag to use

### General Settings

#### Output Mode
- `quiet`: No terminal output
- `standard`: Shows startup logo, configuration info, and high-level status (default)
- `verbose`: Includes all standard output plus detailed logs

#### Interrogation Settings
- `max_iterations`: Maximum number of discovery cycles (default: 5)
```

## Loading Configuration from YAML

```python
from agent_interrogator import InterrogationConfig
from pathlib import Path
import yaml

# Load config from YAML file
with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)

# Create config from dictionary
config = InterrogationConfig.parse_obj(config_dict)

# Create and run the interrogator
interrogator = AgentInterrogator(config, callback)
profile = await interrogator.interrogate()
```
```

## Callback Configuration

The Agent Interrogator uses a callback-based approach for agent interaction. The callback function must be async and follow this signature:

```python
Callable[[str], Awaitable[str]]  # Takes a prompt string, returns a response string
```

If your callback implementation requires cleanup (e.g., closing connections), implement the `cleanup` method:

```python
async def cleanup(self) -> None:
    # Close connections, cleanup resources, etc.
    pass
```

The framework provides several built-in callback implementations:

### HTTP API Integration

```python
from agent_interrogator.callbacks import HTTPCallback

callback = HTTPCallback(
    endpoint="https://api.youragent.com/chat",
    api_key="your-api-key",  # Optional
    headers={  # Optional custom headers
        "Content-Type": "application/json",
        "User-Agent": "AgentInterrogator/1.0"
    }
)

interrogator = AgentInterrogator(config, callback)
```

### WebSocket Real-time Communication

```python
from agent_interrogator.callbacks import WebSocketCallback

callback = WebSocketCallback(
    ws_url="wss://youragent.com/ws",
    message_format={  # Optional message format
        "type": "prompt",
        "content": "{prompt}"  # {prompt} is replaced with actual prompt
    }
)

interrogator = AgentInterrogator(config, callback)
```

### Browser Automation (Playwright)

```python
from agent_interrogator.callbacks import PlaywrightCallback

callback = PlaywrightCallback(
    url="http://localhost:8501/",
    prompt_selector='textarea[type="textarea"]',
    submit_selector='button[type="submit"]',
    response_selector='.response-text',
    browser_type="chromium",  # or "firefox", "webkit"
    headless=True,
    timeout=30000  # milliseconds
)

interrogator = AgentInterrogator(config, callback)
```

### Queue-based Integration

```python
from agent_interrogator.callbacks import QueueCallback
from asyncio import Queue

# Create request/response queues
request_queue = Queue()
response_queue = Queue()

callback = QueueCallback(
    request_queue=request_queue,
    response_queue=response_queue
)

interrogator = AgentInterrogator(config, callback)
```

### Adding Retry Logic

You can wrap any callback with retry logic using the `with_retry` decorator:

```python
from agent_interrogator.callbacks import with_retry

@with_retry(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    exceptions=(ConnectionError, TimeoutError)
)
async def my_callback(prompt: str) -> str:
    # Your callback implementation
    return await make_api_call(prompt)

interrogator = AgentInterrogator(config, my_callback)
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
