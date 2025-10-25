"""Example implementation of MCP (Model Context Protocol) callback using official MCP SDK.

This implementation uses the official MCP Python SDK to connect to MCP servers via SSE transport.
It properly handles the MCP protocol handshake, authentication, and provides capability discovery.

FEATURES:
- Official MCP SDK integration
- SSE (Server-Sent Events) transport
- Bearer token authentication for HuggingFace
- Full MCP handshake and capability discovery
- Tool and resource enumeration

USAGE:
1. Install dependencies: pip install -e .
2. Set environment variables:
   - OPENAI_API_KEY: Your OpenAI API key
   - HF_TOKEN or HUGGINGFACE_TOKEN: Your HuggingFace access token
3. Run: python examples/callbacks_HuggingFace_MCP_example.py

NOTE ON AUTHENTICATION AND PROXY:
The official MCP Python SDK's sse_client does not currently support:
1. Custom HTTP headers (for Bearer token authentication)
2. HTTP proxy configuration

These are limitations of the current SDK implementation. For HuggingFace MCP server access,
you may need to use their official client libraries or wait for SDK updates.
"""
import traceback
import os
from agent_interrogator.config import InterrogationConfig, LLMConfig, ModelProvider
import asyncio
from typing import Optional, Dict, List, Any

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import httpx
from urllib.parse import urlparse

from agent_interrogator import AgentInterrogator

# Load environment variables from .env file if present
load_dotenv(override=True)


def should_use_hf_token(url: str) -> bool:
    """Check if the target URL is a HuggingFace domain.

    Only HuggingFace domains should receive the HuggingFace authentication token
    to prevent credential leaking to arbitrary MCP servers during security testing.

    Args:
        url: The target URL to check

    Returns:
        True if the URL is a huggingface.co domain, False otherwise
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        # Check if hostname is exactly huggingface.co or a subdomain of it
        return hostname == "huggingface.co" or hostname.endswith(".huggingface.co")
    except Exception:
        return False


class MCPCallback:
    """Callback for interacting with a Model Context Protocol (MCP) server using official SDK.

    This implementation uses the official MCP Python SDK to properly connect to MCP servers
    via SSE (Server-Sent Events) transport. It handles capability discovery and tool/resource
    enumeration.

    The callback handles:
    - Official MCP SDK integration via SSE transport
    - Connection initialization and proper handshake
    - Discovery of available tools and resources
    - Managing session state and cleanup

    Note: The current SDK does not support Bearer token auth or HTTP proxies in sse_client.
    """

    def __init__(
        self,
        server_url: str,
        client_name: str = "agent-interrogator",
        client_version: str = "1.0.0",
        auth_token: Optional[str] = None,
        proxy_url: Optional[str] = None,
        timeout: int = 30,
        insecure: bool = False
    ):
        """Initialize the MCP callback.

        Args:
            server_url: Base URL of the MCP server (e.g., https://huggingface.co/mcp)
            client_name: Name of this client for MCP handshake
            client_version: Version of this client
            auth_token: Optional authentication token (currently not supported by SDK)
            proxy_url: Optional proxy URL (currently not supported by SDK)
            timeout: Request timeout in seconds
            insecure: Disable SSL certificate verification (useful for proxy interception)
        """
        self.server_url = server_url.rstrip("/")
        self.client_name = client_name
        self.client_version = client_version
        self.auth_token = auth_token
        self.proxy_url = proxy_url
        self.timeout = timeout
        self.insecure = insecure

        # Warn about configuration
        if self.insecure:
            print("Warning: SSL certificate verification disabled (insecure mode enabled)")
        if self.auth_token:
            print(f"Using Bearer token authentication for HuggingFace")
        if self.proxy_url:
            print(f"Note: HTTP proxy set via environment variables (http_proxy/https_proxy) is automatically used by httpx")

        # MCP SDK session management
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self.initialized: bool = False

        # MCP server capabilities
        self.server_info: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []
        self.resources: List[Dict[str, Any]] = []

    async def initialize(self):
        """Initialize connection to the MCP server using official SDK.

        Performs the MCP handshake:
        1. Establishes HTTP connection
        2. Creates ClientSession
        3. Initializes session with server
        4. Discovers available tools and resources
        """
        if self.initialized:
            return

        print(f"Connecting to MCP server at {self.server_url}...")

        try:
            # Prepare headers for Bearer token authentication
            # SECURITY: Only include HuggingFace token when targeting huggingface.co domains
            # This prevents credential leaking to arbitrary MCP servers during pentesting
            headers = {}
            if self.auth_token and should_use_hf_token(self.server_url):
                headers["Authorization"] = f"Bearer {self.auth_token}"
                print(f"Using HuggingFace authentication token (target domain verified)")
            elif self.auth_token and not should_use_hf_token(self.server_url):
                print(f"Warning: HuggingFace token provided but target is not huggingface.co domain")
                print(f"Token will NOT be sent to protect against credential leaking")

            # Create httpx client factory with optional SSL verification bypass
            client_factory = None
            if self.insecure:
                # Create custom httpx client factory with SSL verification disabled
                # This is useful when routing through intercepting proxies like Burp Suite
                def insecure_client_factory(
                    headers: Optional[Dict[str, str]] = None,
                    timeout: Optional[httpx.Timeout] = None,
                    auth: Optional[httpx.Auth] = None
                ) -> httpx.AsyncClient:
                    # Use provided timeout or fall back to our configured timeout
                    if timeout is None:
                        timeout = httpx.Timeout(self.timeout)
                    return httpx.AsyncClient(
                        verify=False,
                        headers=headers,
                        timeout=timeout,
                        auth=auth
                    )
                client_factory = insecure_client_factory

            # Establish HTTP connection using streamable HTTP transport
            if client_factory:
                self._streams_context = streamablehttp_client(
                    url=self.server_url,
                    headers=headers,
                    timeout=self.timeout,
                    httpx_client_factory=client_factory
                )
            else:
                self._streams_context = streamablehttp_client(
                    url=self.server_url,
                    headers=headers,
                    timeout=self.timeout
                )

            streams = await self._streams_context.__aenter__()

            # Unpack streams: (read_stream, write_stream, get_session_id_callback)
            read_stream, write_stream, get_session_id = streams

            # Create and initialize ClientSession with just the read/write streams
            self._session_context = ClientSession(read_stream, write_stream)
            self.session = await self._session_context.__aenter__()

            # Initialize the session with the server
            init_result = await self.session.initialize()

            # Extract server info
            if hasattr(init_result, 'serverInfo'):
                server_info_obj = init_result.serverInfo
                self.server_info = {
                    'name': getattr(server_info_obj, 'name', 'Unknown'),
                    'version': getattr(server_info_obj, 'version', 'Unknown')
                }
            else:
                self.server_info = {'name': 'Unknown', 'version': 'Unknown'}

            # Extract capabilities
            capabilities = {}
            if hasattr(init_result, 'capabilities'):
                cap_obj = init_result.capabilities
                if hasattr(cap_obj, '__dict__'):
                    capabilities = cap_obj.__dict__

            print(f"Connected to MCP server: {self.server_info.get('name')} "
                  f"v{self.server_info.get('version')}")
            print(f"Server capabilities: {list(capabilities.keys())}")

            # Discover tools
            try:
                tools_result = await self.session.list_tools()
                if hasattr(tools_result, 'tools'):
                    self.tools = []
                    for tool in tools_result.tools:
                        tool_dict = {
                            'name': getattr(tool, 'name', 'Unknown'),
                            'description': getattr(tool, 'description', 'No description'),
                            'inputSchema': getattr(tool, 'inputSchema', {})
                        }
                        self.tools.append(tool_dict)

                    print(f"Discovered {len(self.tools)} tools")
                    for tool in self.tools:
                        print(f"  - {tool['name']}: {tool['description']}")
                else:
                    print("No tools available")
            except Exception as e:
                print(f"Warning: Could not list tools: {e}")

            # Discover resources
            try:
                resources_result = await self.session.list_resources()
                if hasattr(resources_result, 'resources'):
                    self.resources = []
                    for resource in resources_result.resources:
                        resource_dict = {
                            'uri': getattr(resource, 'uri', 'Unknown'),
                            'name': getattr(resource, 'name', 'No name'),
                            'description': getattr(resource, 'description', '')
                        }
                        self.resources.append(resource_dict)

                    print(f"Discovered {len(self.resources)} resources")
                    for resource in self.resources:
                        print(f"  - {resource['uri']}: {resource['name']}")
                else:
                    print("No resources available")
            except Exception as e:
                print(f"Warning: Could not list resources: {e}")

            self.initialized = True

        except Exception as e:
            print(f"Failed to initialize MCP connection: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Clean up resources and close connection."""
        if self._session_context and self.session:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing session: {e}")
            self.session = None
            self._session_context = None

        if self._streams_context:
            try:
                await self._streams_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing streams: {e}")
            self._streams_context = None

        if self._http_client:
            try:
                await self._http_client.aclose()
            except Exception as e:
                print(f"Error closing HTTP client: {e}")
            self._http_client = None

        self.initialized = False
        print("MCP connection closed")

    async def __call__(self, prompt: str) -> str:
        """Handle a prompt by interacting with the MCP server.

        This method returns information about available tools and resources.
        Future versions will implement intelligent tool routing based on prompt analysis.

        Args:
            prompt: The input prompt to send

        Returns:
            Response from the MCP server with tool/resource information
        """
        if not self.initialized:
            await self.initialize()

        try:
            response_parts = []
            response_parts.append(f"Prompt received: {prompt}")
            response_parts.append(f"\nMCP Server: {self.server_info.get('name', 'Unknown')}")
            response_parts.append(f"Available tools: {len(self.tools)}")

            if self.tools:
                response_parts.append("\nTools:")
                for tool in self.tools[:10]:
                    response_parts.append(f"  - {tool['name']}: {tool['description']}")

            response_parts.append(f"\nAvailable resources: {len(self.resources)}")

            if self.resources:
                response_parts.append("\nResources:")
                for resource in self.resources[:10]:
                    response_parts.append(f"  - {resource['uri']}")

            return "\n".join(response_parts)

        except Exception as e:
            return f"Error communicating with MCP server: {str(e)}"


# Configuration for AgentInterrogator
config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    max_iterations=5
)

# Initialize MCP Callback for HuggingFace MCP server
# Get HuggingFace token from environment variables
# You can get your token from https://huggingface.co/settings/tokens
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

if not hf_token:
    print("Warning: No HuggingFace token found. Set HF_TOKEN or HUGGINGFACE_TOKEN environment variable.")
    print("You can get a token from https://huggingface.co/settings/tokens")
    print("Note: The current MCP SDK does not support Bearer token auth, so this may not work with HuggingFace.")

# Detect if proxy is configured via environment variables
# If proxy is detected, enable insecure mode to bypass SSL verification
# This is common when using intercepting proxies like Burp Suite
http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
use_insecure_mode = bool(http_proxy or https_proxy)

if use_insecure_mode:
    print(f"Detected proxy configuration: http_proxy={http_proxy}, https_proxy={https_proxy}")
    print("Enabling insecure mode to bypass SSL verification for proxy interception")

callback = MCPCallback(
    server_url="https://huggingface.co/mcp",
    client_name="agent-interrogator",
    client_version="1.0.0",
    auth_token=hf_token,
    proxy_url=None,
    timeout=30,
    insecure=use_insecure_mode
)


async def main():
    """Run the MCP callback to discover capabilities of HuggingFace MCP server."""
    print("=" * 60)
    print("MCP Server Capability Discovery")
    print("=" * 60)
    print()

    try:
        # Initialize the MCP connection
        await callback.initialize()
        print()

        # Run the interrogation to discover agent capabilities
        print("Starting agent interrogation...")
        print()
        interrogator = AgentInterrogator(config, callback)
        profile = await interrogator.interrogate()

        # Display results
        print()
        print("=" * 60)
        print("INTERROGATION RESULTS")
        print("=" * 60)
        print(profile)

    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
    finally:
        # Clean up
        print("\nCleaning up...")
        await callback.cleanup()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())