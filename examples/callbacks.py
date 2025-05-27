"""Example implementations of agent callbacks for different scenarios."""

import aiohttp
import asyncio
from typing import Dict, Any, Optional
import json

# Example 1: Simple HTTP API callback
async def http_api_callback(prompt: str) -> str:
    """Callback for interacting with an agent via HTTP API."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.example.com/agent",
            json={"prompt": prompt},
            headers={"Authorization": "Bearer your-token"}
        ) as response:
            result = await response.json()
            return result["response"]

# Example 2: Local process callback
async def local_process_callback(prompt: str) -> str:
    """Callback for interacting with an agent running as a local process."""
    # Simulating interaction with a local process via stdin/stdout
    process = await asyncio.create_subprocess_exec(
        "agent-cli",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate(prompt.encode())
    return stdout.decode().strip()

# Example 3: WebSocket callback
class WebSocketCallback:
    """Callback for interacting with an agent via WebSocket."""
    
    def __init__(self, websocket_url: str):
        self.url = websocket_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
    
    async def connect(self):
        """Establish WebSocket connection."""
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(self.url)
    
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
    
    async def __call__(self, prompt: str) -> str:
        """Make the class callable as a callback."""
        if not self.ws:
            await self.connect()
        
        await self.ws.send_str(json.dumps({"type": "prompt", "content": prompt}))
        response = await self.ws.receive_json()
        return response["content"]

# Example 4: Queue-based callback for distributed systems
class QueueCallback:
    """Callback for interacting with an agent via message queues."""
    
    def __init__(self, request_queue: asyncio.Queue, response_queue: asyncio.Queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
    
    async def __call__(self, prompt: str) -> str:
        """Send prompt to request queue and wait for response."""
        await self.request_queue.put(prompt)
        return await self.response_queue.get()

# Example 5: Retry wrapper for any callback
def with_retry(
    callback,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> callable:
    """Wrap any callback with retry logic."""
    
    async def retry_wrapper(prompt: str) -> str:
        last_error = None
        current_delay = delay
        
        for attempt in range(max_retries):
            try:
                return await callback(prompt)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        
        raise last_error
    
    return retry_wrapper

# Example usage of retry wrapper
retrying_http_callback = with_retry(
    http_api_callback,
    max_retries=3,
    delay=1.0,
    backoff=2.0
)

# Example 6: Playwright-based callback for browser automation
class PlaywrightCallback:
    """Callback for interacting with agents through web interfaces using Playwright.
    
    This callback allows interaction with agents that are accessible through web UIs,
    such as chat interfaces or web-based playgrounds. It supports multiple browsers
    and can handle complex UI interactions.
    """
    
    def __init__(
        self,
        url: str,
        prompt_selector: str,
        submit_selector: str,
        response_selector: str,
        browser_type: str = "chromium",
        headless: bool = True
    ):
        """Initialize the Playwright callback.
        
        Args:
            url: The URL of the web interface
            prompt_selector: CSS selector for the prompt input element
            submit_selector: CSS selector for the submit button
            response_selector: CSS selector for the response element
            browser_type: Browser to use (chromium, firefox, or webkit)
            headless: Whether to run browser in headless mode
        """
        self.url = url
        self.prompt_selector = prompt_selector
        self.submit_selector = submit_selector
        self.response_selector = response_selector
        self.browser_type = browser_type
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
    
    async def initialize(self):
        """Initialize the browser and page."""
        from playwright.async_api import async_playwright
        
        self.playwright = await async_playwright().start()
        browser_client = getattr(self.playwright, self.browser_type)
        self.browser = await browser_client.launch(headless=self.headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        await self.page.goto(self.url)
        
        # Wait for the page to be fully loaded
        await self.page.wait_for_load_state("networkidle")
    
    async def cleanup(self):
        """Clean up browser resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def __call__(self, prompt: str) -> str:
        """Send prompt through web interface and get response."""
        if not self.page:
            await self.initialize()
        
        try:
            # Type the prompt
            await self.page.fill(self.prompt_selector, prompt)
            
            # Click submit and wait for response
            await self.page.click(self.submit_selector)
            
            # Wait for and get the response
            response_element = await self.page.wait_for_selector(
                self.response_selector,
                state="visible",
                timeout=30000  # 30 seconds timeout
            )
            response = await response_element.text_content()
            
            # Clean up the response (remove any UI artifacts)
            return response.strip()
            
        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to interact with web interface: {str(e)}")

# Example usage of Playwright callback
playwright_callback = PlaywrightCallback(
    url="https://example.com/agent-chat",
    prompt_selector="#prompt-textarea",
    submit_selector="#submit-button",
    response_selector="#response-div",
    browser_type="chromium",
    headless=True
)
