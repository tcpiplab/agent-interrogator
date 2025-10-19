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

### Security Considerations
- Never proxy LLM API calls through Burp (current behavior is correct)
- Always proxy target MCP server calls through Burp when configured
- Ensure sensitive tokens are never logged in verbose output
- Consider adding `--redact-tokens` flag for safe log sharing

### Testing Requirements
- Test rate limiting with actual MCP servers
- Verify HF token is only sent to huggingface.co domains
- Test custom header injection
- Validate Markdown output matches terminal output exactly
- Test Azure OpenAI integration with real Azure deployment
