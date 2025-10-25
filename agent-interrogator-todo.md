# Agent Interrogator TODO List

## High Priority - Core Functionality

### 1. Fix HuggingFace Token Behavior
- Remove default HuggingFace authentication token from all requests
- Only include HF token when target URL is explicitly a `huggingface.co` domain
- This prevents leaking credentials to arbitrary MCP servers during pentesting
- **Risk Level**: HIGH - Current behavior leaks credentials to untrusted targets

### 2. Implement Target Rate Limiting
- Add `--rate-limit-target` flag (boolean)
- When enabled, enforce 5 second delay between requests to target MCP server
- Critical for avoiding detection/blocking during authorized testing
- Helps maintain stealth during reconnaissance phase

### 3. Implement LLM Rate Limiting
- Add `--rate-limit-llm` flag (boolean)
- When enabled, enforce 5 second delay between OpenAI API calls
- Helps avoid hitting OpenAI rate limits during extended interrogation sessions
- Prevents 429 errors and service disruption

### 4. Add Support for Localhost Ollama LLM
- Add `--ollama` flag (boolean) to enable Ollama mode
- Add `--ollama-model-name` flag (required string argument when `--ollama` is specified)
  - Example: `--ollama --ollama-model-name llama2`
  - Example: `--ollama --ollama-model-name mistral`
  - Example: `--ollama --ollama-model-name codellama:13b`
- When `--ollama` is specified, the tool should use a local Ollama instance instead of OpenAI
- Default Ollama endpoint: `http://localhost:11434` (standard Ollama API port)
- Optional: Consider adding `--ollama-endpoint` flag to override default localhost:11434
- Ollama uses an OpenAI-compatible API, so implementation can leverage existing patterns
- **Key Requirements**:
  - Validate that `--ollama-model-name` is provided when `--ollama` is used
  - Display clear error if Ollama service is not running or unreachable
  - Show which Ollama model is being used in the startup configuration display
  - Support Ollama's model naming conventions (model:tag format)
  - Handle Ollama-specific error responses gracefully
  - OPENAI_API_KEY should not be required when using Ollama mode
- **User Experience**:
  - Example command: `./agent-interrogator-cli.py --target https://example.com/mcp --ollama --ollama-model-name llama2`
  - Show "Using Ollama model: llama2 at http://localhost:11434" in startup output
  - If Ollama is unreachable, show helpful message like "Cannot connect to Ollama at localhost:11434. Is Ollama running? Try: ollama serve"
- **Benefits for Pentesters**:
  - No API costs for extended interrogation sessions
  - Complete privacy - no data sent to external services
  - Works in air-gapped or restricted network environments
  - Can use specialized models optimized for security tasks
  - Full control over model parameters and behavior

## Medium Priority - Enhanced Pentesting Features

### 4. Custom HTTP Headers Support
- Add `--header` flag (can be specified multiple times)
- Format validation: must match pattern `Header-Name: value`
- Examples:
  - `--header "X-Pentest: Company_Name"`
  - `--header "X-Vendor: tcpiplab.com"`
- Store headers and apply to all target MCP server requests (not LLM API calls)
- Useful for identifying traffic during authorized testing

### 5. Improve File Output Format
- Currently outputs bare JSON to file
- Add Markdown output option that matches terminal output exactly
- Should capture the complete interrogation session including:
  - Configuration used
  - All discovery cycles
  - All analysis cycles
  - Final capability summary
  - Timestamps for each phase
- Consider adding `--output-format` flag with options: `json`, `markdown`, `both`

## Lower Priority - Documentation & Advanced Features

### 6. Documentation Updates
- Document the authentication guessing feature
- Document all new CLI flags and their behavior
- Add examples showing common pentesting scenarios
- Clarify what gets proxied through Burp vs what doesn't
- Add troubleshooting section for common issues
- Update README.md with pentesting use cases

### 7. Azure OpenAI Support
- Add ability to specify custom OpenAI-compatible API endpoint
- Support Azure OpenAI deployments with custom URLs
- Add flags like:
  - `--llm-endpoint` for custom API base URL
  - `--llm-deployment-name` for Azure deployment names
  - `--llm-api-version` for Azure API versioning
- Maintain compatibility with standard OpenAI API
- Consider supporting other OpenAI-compatible APIs (e.g., local LLMs)

## Notes

### Implementation Considerations
- All rate limiting should be implemented with async sleep to avoid blocking
- Custom headers should be validated before any requests are made
- File output should include option to append timestamp to filename
- Azure OpenAI implementation should reuse existing OpenAI client with custom base URL
- Ollama implementation should use OpenAI-compatible client with base_url="http://localhost:11434/v1"
- Ollama mode should bypass OPENAI_API_KEY requirement check
- Consider connection timeout and retry logic for Ollama connectivity issues
- Ollama responses may be slower than OpenAI - adjust timeouts accordingly

### Security Considerations
- Never proxy LLM API calls through Burp (current behavior is correct)
- Always proxy target MCP server calls through Burp when configured
- Ensure sensitive tokens are never logged in verbose output
- Consider adding `--redact-tokens` flag for safe log sharing
- Ollama runs on localhost and should never be proxied through Burp
- Ollama mode keeps all LLM data local - no external API calls for complete privacy

### Testing Requirements
- Test rate limiting with actual MCP servers
- Verify HF token is only sent to huggingface.co domains
- Test custom header injection
- Validate Markdown output matches terminal output exactly
- Test Azure OpenAI integration with real Azure deployment
- Test Ollama integration with multiple models (llama2, mistral, codellama)
- Verify Ollama works without OPENAI_API_KEY set
- Test Ollama error handling when service is not running
- Verify Ollama calls are never proxied through Burp
- Test Ollama with rate limiting enabled
