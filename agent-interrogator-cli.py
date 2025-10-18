#!/usr/bin/env python3
"""Agent Interrogator CLI - Pentesting tool for MCP server reconnaissance.

This CLI tool performs automated reconnaissance of MCP (Model Context Protocol) servers
by systematically discovering their capabilities, tools, and functions through iterative
LLM-based interrogation.

FEATURES:
- Generic MCP server support (any MCP endpoint)
- Proxy support with automatic SSL bypass
- Bearer token authentication via environment variables
- JSON and Markdown output formats
- Colored terminal output for easy reading

USAGE:
    # Basic usage
    python agent-interrogator-cli.py --target https://example.com/mcp

    # With proxy (requires http_proxy/https_proxy env vars)
    python agent-interrogator-cli.py --target https://example.com/mcp --proxy

    # Custom model and timeout
    python agent-interrogator-cli.py --target https://example.com/mcp --openai-model-name gpt-4o --timeout 60

    # Save output to files
    python agent-interrogator-cli.py --target https://example.com/mcp --output json,markdown

ENVIRONMENT VARIABLES:
    OPENAI_API_KEY:     Required. Your OpenAI API key
    MCP_AUTH_TOKEN:     Optional. Bearer token for MCP server authentication
    HF_TOKEN:           Optional. HuggingFace token (alternative to MCP_AUTH_TOKEN)
    http_proxy:         Optional. HTTP proxy URL (e.g., http://127.0.0.1:8080)
    https_proxy:        Optional. HTTPS proxy URL (e.g., http://127.0.0.1:8080)

EXIT CODES:
    0:  Success
    1:  General error
    2:  Missing required environment variable
    3:  Invalid proxy configuration
    4:  MCP connection failed
    5:  Interrogation failed
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_interrogator import AgentInterrogator
from agent_interrogator.config import InterrogationConfig, LLMConfig, ModelProvider

# Load environment variables
load_dotenv(override=True)

# Rich console for colored output
console = Console()

# Exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_MISSING_ENV_VAR = 2
EXIT_INVALID_PROXY_CONFIG = 3
EXIT_MCP_CONNECTION_FAILED = 4
EXIT_INTERROGATION_FAILED = 5


class MCPCallback:
    """Callback for interacting with a Model Context Protocol (MCP) server.

    This implementation uses the official MCP Python SDK with streamable HTTP transport
    to connect to generic MCP servers. It supports Bearer token authentication and
    automatic SSL bypass when using intercepting proxies.
    """

    def __init__(
        self,
        server_url: str,
        client_name: str = "agent-interrogator-cli",
        client_version: str = "1.0.0",
        auth_token: Optional[str] = None,
        timeout: int = 30,
        insecure: bool = False,
    ):
        """Initialize the MCP callback.

        Args:
            server_url: Base URL of the MCP server
            client_name: Name of this client for MCP handshake
            client_version: Version of this client
            auth_token: Optional Bearer authentication token
            timeout: Request timeout in seconds
            insecure: Disable SSL certificate verification for proxy interception
        """
        self.server_url = server_url.rstrip("/")
        self.client_name = client_name
        self.client_version = client_version
        self.auth_token = auth_token
        self.timeout = timeout
        self.insecure = insecure

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
        """Initialize connection to the MCP server.

        Performs the MCP handshake:
        1. Establishes HTTP connection
        2. Creates ClientSession
        3. Initializes session with server
        4. Discovers available tools and resources
        """
        if self.initialized:
            return

        console.print(f"[cyan]Connecting to MCP server at {self.server_url}...[/cyan]")

        try:
            # Prepare headers for Bearer token authentication
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
                console.print("[green]Using Bearer token authentication[/green]")

            # Create httpx client factory with optional SSL verification bypass
            client_factory = None
            if self.insecure:
                console.print(
                    "[yellow]SSL verification disabled for proxy interception[/yellow]"
                )

                # Get proxy configuration from environment
                http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
                https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")

                # Build proxy dict for httpx
                proxies = {}
                if http_proxy:
                    proxies["http://"] = http_proxy
                if https_proxy:
                    proxies["https://"] = https_proxy

                def insecure_client_factory(
                    headers: Optional[Dict[str, str]] = None,
                    timeout: Optional[httpx.Timeout] = None,
                    auth: Optional[httpx.Auth] = None,
                ) -> httpx.AsyncClient:
                    if timeout is None:
                        timeout = httpx.Timeout(self.timeout)
                    # Force HTTP/1.1 for proxy compatibility
                    # Some proxies don't handle HTTP/2 properly, so we explicitly disable it
                    return httpx.AsyncClient(
                        verify=False,
                        headers=headers,
                        timeout=timeout,
                        auth=auth,
                        http1=True,   # Explicitly enable HTTP/1.1
                        http2=False,  # Explicitly disable HTTP/2
                        proxies=proxies if proxies else None  # Explicitly set proxy
                    )

                client_factory = insecure_client_factory

            # Establish HTTP connection using streamable HTTP transport
            if client_factory:
                self._streams_context = streamablehttp_client(
                    url=self.server_url,
                    headers=headers,
                    timeout=self.timeout,
                    httpx_client_factory=client_factory,
                )
            else:
                self._streams_context = streamablehttp_client(
                    url=self.server_url, headers=headers, timeout=self.timeout
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
            if hasattr(init_result, "serverInfo"):
                server_info_obj = init_result.serverInfo
                self.server_info = {
                    "name": getattr(server_info_obj, "name", "Unknown"),
                    "version": getattr(server_info_obj, "version", "Unknown"),
                }
            else:
                self.server_info = {"name": "Unknown", "version": "Unknown"}

            # Extract capabilities
            capabilities = {}
            if hasattr(init_result, "capabilities"):
                cap_obj = init_result.capabilities
                if hasattr(cap_obj, "__dict__"):
                    capabilities = cap_obj.__dict__

            console.print(
                f"[green]Connected to MCP server: {self.server_info.get('name')} "
                f"v{self.server_info.get('version')}[/green]"
            )
            console.print(f"[cyan]Server capabilities: {list(capabilities.keys())}[/cyan]")

            # Discover tools
            try:
                tools_result = await self.session.list_tools()
                if hasattr(tools_result, "tools"):
                    self.tools = []
                    for tool in tools_result.tools:
                        tool_dict = {
                            "name": getattr(tool, "name", "Unknown"),
                            "description": getattr(tool, "description", "No description"),
                            "inputSchema": getattr(tool, "inputSchema", {}),
                        }
                        self.tools.append(tool_dict)

                    console.print(f"[green]Discovered {len(self.tools)} tools[/green]")
                    if self.tools:
                        table = Table(title="Available Tools")
                        table.add_column("Tool Name", style="cyan")
                        table.add_column("Description", style="white")
                        for tool in self.tools[:10]:
                            table.add_row(tool["name"], tool["description"])
                        if len(self.tools) > 10:
                            table.add_row(
                                "...", f"(and {len(self.tools) - 10} more)"
                            )
                        console.print(table)
                else:
                    console.print("[yellow]No tools available[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not list tools: {e}[/yellow]")

            # Discover resources
            try:
                resources_result = await self.session.list_resources()
                if hasattr(resources_result, "resources"):
                    self.resources = []
                    for resource in resources_result.resources:
                        resource_dict = {
                            "uri": getattr(resource, "uri", "Unknown"),
                            "name": getattr(resource, "name", "No name"),
                            "description": getattr(resource, "description", ""),
                        }
                        self.resources.append(resource_dict)

                    console.print(
                        f"[green]Discovered {len(self.resources)} resources[/green]"
                    )
                    if self.resources:
                        table = Table(title="Available Resources")
                        table.add_column("URI", style="cyan")
                        table.add_column("Name", style="white")
                        for resource in self.resources[:10]:
                            table.add_row(resource["uri"], resource["name"])
                        if len(self.resources) > 10:
                            table.add_row(
                                "...", f"(and {len(self.resources) - 10} more)"
                            )
                        console.print(table)
                else:
                    console.print("[yellow]No resources available[/yellow]")
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not list resources: {e}[/yellow]"
                )

            self.initialized = True

        except Exception as e:
            console.print(f"[red]Failed to initialize MCP connection: {str(e)}[/red]")
            console.print(f"[red]Error type: {type(e).__name__}[/red]")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Clean up resources and close connection."""
        if self._session_context and self.session:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                console.print(f"[yellow]Error closing session: {e}[/yellow]")
            self.session = None
            self._session_context = None

        if self._streams_context:
            try:
                await self._streams_context.__aexit__(None, None, None)
            except Exception as e:
                console.print(f"[yellow]Error closing streams: {e}[/yellow]")
            self._streams_context = None

        if self._http_client:
            try:
                await self._http_client.aclose()
            except Exception as e:
                console.print(f"[yellow]Error closing HTTP client: {e}[/yellow]")
            self._http_client = None

        self.initialized = False
        console.print("[cyan]MCP connection closed[/cyan]")

    async def __call__(self, prompt: str) -> str:
        """Handle a prompt by interacting with the MCP server.

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
            response_parts.append(
                f"\nMCP Server: {self.server_info.get('name', 'Unknown')}"
            )
            response_parts.append(f"Available tools: {len(self.tools)}")

            if self.tools:
                response_parts.append("\nTools:")
                for tool in self.tools[:10]:
                    response_parts.append(
                        f"  - {tool['name']}: {tool['description']}"
                    )

            response_parts.append(f"\nAvailable resources: {len(self.resources)}")

            if self.resources:
                response_parts.append("\nResources:")
                for resource in self.resources[:10]:
                    response_parts.append(f"  - {resource['uri']}")

            return "\n".join(response_parts)

        except Exception as e:
            return f"Error communicating with MCP server: {str(e)}"


def check_proxy_configuration(use_proxy: bool) -> bool:
    """Validate proxy configuration based on user intent and environment.

    Args:
        use_proxy: Whether user specified --proxy flag

    Returns:
        True if proxy should be used, False otherwise

    Exits:
        EXIT_INVALID_PROXY_CONFIG if configuration is invalid
    """
    http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    proxy_configured = bool(http_proxy or https_proxy)

    if use_proxy and not proxy_configured:
        console.print(
            Panel(
                "[red]Proxy requested but environment variables not set![/red]\n\n"
                "To use a proxy, set these environment variables:\n\n"
                "[cyan]export http_proxy=http://127.0.0.1:8080\n"
                "export https_proxy=http://127.0.0.1:8080[/cyan]\n\n"
                "Then run the tool again with --proxy flag.",
                title="Proxy Configuration Error",
                border_style="red",
            )
        )
        sys.exit(EXIT_INVALID_PROXY_CONFIG)

    if not use_proxy and proxy_configured:
        console.print(
            Panel(
                "[yellow]Warning: Proxy environment variables are set![/yellow]\n\n"
                f"http_proxy: {http_proxy}\n"
                f"https_proxy: {https_proxy}\n\n"
                "If you want to proxy traffic, add the --proxy flag.\n"
                "If you do NOT want to proxy traffic, unset these variables:\n\n"
                "[cyan]unset http_proxy https_proxy[/cyan]",
                title="Proxy Configuration Warning",
                border_style="yellow",
            )
        )
        sys.exit(EXIT_INVALID_PROXY_CONFIG)

    if use_proxy and proxy_configured:
        console.print(
            Panel(
                f"[green]Proxy enabled[/green]\n\n"
                f"http_proxy: {http_proxy}\n"
                f"https_proxy: {https_proxy}\n\n"
                "SSL verification will be automatically disabled.",
                title="Proxy Configuration",
                border_style="green",
            )
        )
        return True

    return False


def check_environment_variables(no_auth: bool = False) -> Dict[str, Optional[str]]:
    """Check for required and optional environment variables.

    Args:
        no_auth: If True, disable authentication and show warning

    Returns:
        Dict with environment variable values

    Exits:
        EXIT_MISSING_ENV_VAR if required variables are missing
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    mcp_auth_token = os.getenv("MCP_AUTH_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if not openai_api_key:
        console.print(
            Panel(
                "[red]Missing required environment variable: OPENAI_API_KEY[/red]\n\n"
                "Set your OpenAI API key:\n\n"
                "[cyan]export OPENAI_API_KEY='your-api-key-here'[/cyan]\n\n"
                "Get your API key from: https://platform.openai.com/api-keys",
                title="Missing API Key",
                border_style="red",
            )
        )
        sys.exit(EXIT_MISSING_ENV_VAR)

    # Handle --no-auth flag
    if no_auth:
        console.print(
            Panel(
                "[yellow]WARNING: Authentication disabled (--no-auth)[/yellow]\n\n"
                "No Authorization header will be sent to the target MCP server.\n\n"
                "[bold]Security Note:[/bold]\n"
                "If the target server accepts requests without authentication,\n"
                "this indicates a potential security issue with the target server.\n\n"
                "This mode is intended for:\n"
                "  - Development/testing environments\n"
                "  - Secret URL endpoints (e.g., Zapier MCP servers)\n"
                "  - Servers that error when receiving unexpected auth headers",
                title="No Authentication Mode",
                border_style="yellow",
            )
        )
        # Force auth token to None when --no-auth is specified
        mcp_auth_token = None
    else:
        # Normal auth token detection
        if mcp_auth_token:
            console.print("[green]Found MCP authentication token[/green]")
        else:
            console.print(
                "[yellow]No MCP authentication token found (MCP_AUTH_TOKEN, HF_TOKEN, or HUGGINGFACE_TOKEN)[/yellow]"
            )

    env_vars = {
        "openai_api_key": openai_api_key,
        "mcp_auth_token": mcp_auth_token,
    }

    return env_vars


def save_output(profile: str, output_formats: List[str], target_url: str):
    """Save interrogation results to file(s).

    Args:
        profile: The agent profile output string
        output_formats: List of formats to save ('json', 'markdown')
        target_url: Target MCP server URL for filename generation
    """
    # Generate base filename from target URL
    from urllib.parse import urlparse

    parsed = urlparse(target_url)
    hostname = parsed.hostname or "unknown"
    base_filename = f"agent-profile-{hostname}"

    for fmt in output_formats:
        if fmt == "json":
            # Try to parse profile as JSON if possible
            try:
                # Profile string may contain JSON-like content
                # For now, save as wrapped JSON
                output_data = {
                    "target": target_url,
                    "profile": profile,
                }
                filename = f"{base_filename}.json"
                with open(filename, "w") as f:
                    json.dump(output_data, f, indent=2)
                console.print(f"[green]Saved JSON output to: {filename}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to save JSON: {e}[/red]")

        elif fmt == "markdown":
            filename = f"{base_filename}.md"
            with open(filename, "w") as f:
                f.write(f"# Agent Profile: {target_url}\n\n")
                f.write(profile)
                f.write("\n")
            console.print(f"[green]Saved Markdown output to: {filename}[/green]")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent Interrogator - MCP Server Reconnaissance Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --target https://example.com/mcp
  %(prog)s --target https://huggingface.co/mcp --proxy
  %(prog)s --target https://example.com/mcp --openai-model-name gpt-4o --timeout 60
  %(prog)s --target https://example.com/mcp --output json,markdown

Environment Variables:
  OPENAI_API_KEY    Required. Your OpenAI API key
  MCP_AUTH_TOKEN    Optional. Bearer token for MCP authentication
  HF_TOKEN          Optional. HuggingFace token (alternative to MCP_AUTH_TOKEN)
  http_proxy        Optional. HTTP proxy URL
  https_proxy       Optional. HTTPS proxy URL
        """,
    )

    parser.add_argument(
        "--target",
        required=True,
        help="MCP server endpoint URL (e.g., https://example.com/mcp)",
    )

    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Enable proxy support (requires http_proxy/https_proxy env vars)",
    )

    parser.add_argument(
        "--openai-model-name",
        default="gpt-4o-mini",
        help="OpenAI model to use for interrogation (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="MCP connection timeout in seconds (default: 30)",
    )

    parser.add_argument(
        "--output",
        help="Output format(s): json, markdown, or both (comma-separated). Example: json,markdown",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum interrogation iterations (default: 5)",
    )

    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable authentication (do not send Authorization header). Use for development servers or secret URLs.",
    )

    return parser.parse_args()


async def main():
    """Main CLI entry point."""
    args = parse_arguments()

    # Display banner
    console.print(
        Panel(
            "[bold cyan]Agent Interrogator CLI[/bold cyan]\n"
            "Automated MCP Server Reconnaissance",
            border_style="cyan",
        )
    )
    console.print()

    # Check proxy configuration
    use_proxy = check_proxy_configuration(args.proxy)

    # Check environment variables (pass --no-auth flag)
    env_vars = check_environment_variables(no_auth=args.no_auth)

    console.print()
    console.print(f"[cyan]Target MCP Server:[/cyan] {args.target}")
    console.print(f"[cyan]OpenAI Model:[/cyan] {args.openai_model_name}")
    console.print(f"[cyan]Timeout:[/cyan] {args.timeout} seconds")
    console.print(f"[cyan]Max Iterations:[/cyan] {args.max_iterations}")
    console.print()

    # Initialize configuration
    config = InterrogationConfig(
        llm=LLMConfig(
            provider=ModelProvider.OPENAI,
            model_name=args.openai_model_name,
            api_key=env_vars["openai_api_key"],
        ),
        max_iterations=args.max_iterations,
    )

    # Initialize MCP callback
    callback = MCPCallback(
        server_url=args.target,
        client_name="agent-interrogator-cli",
        client_version="1.0.0",
        auth_token=env_vars["mcp_auth_token"],
        timeout=args.timeout,
        insecure=use_proxy,
    )

    try:
        # Initialize MCP connection
        await callback.initialize()
        console.print()

        # Run interrogation
        console.print(
            Panel(
                "[bold yellow]Starting agent interrogation...[/bold yellow]",
                border_style="yellow",
            )
        )
        console.print()

        interrogator = AgentInterrogator(config, callback)
        profile = await interrogator.interrogate()

        # Display results
        console.print()
        console.print(
            Panel(
                "[bold green]INTERROGATION COMPLETE[/bold green]",
                border_style="green",
            )
        )
        console.print()
        console.print(profile)
        console.print()

        # Save output if requested
        if args.output:
            output_formats = [fmt.strip().lower() for fmt in args.output.split(",")]
            valid_formats = [fmt for fmt in output_formats if fmt in ["json", "markdown"]]
            if valid_formats:
                save_output(str(profile), valid_formats, args.target)
            else:
                console.print(
                    "[yellow]Warning: Invalid output format(s). Use 'json', 'markdown', or both.[/yellow]"
                )

        return EXIT_SUCCESS

    except Exception as e:
        console.print()
        console.print(
            Panel(
                f"[bold red]INTERROGATION FAILED[/bold red]\n\n{str(e)}",
                border_style="red",
            )
        )
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return EXIT_INTERROGATION_FAILED

    finally:
        # Cleanup
        console.print()
        console.print("[cyan]Cleaning up...[/cyan]")
        await callback.cleanup()
        console.print("[green]Done![/green]")


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(EXIT_GENERAL_ERROR)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        traceback.print_exc()
        sys.exit(EXIT_GENERAL_ERROR)