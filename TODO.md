# Agent Interrogator CLI - TODO List

This document tracks pending development tasks for the Agent Interrogator CLI tool.

**Note**: Item numbers do not necessarily reflect priority or implementation sequence.

---

## 1. Rate Limiting - OpenAI API

**Priority**: TBD

**Description**: Add optional `--rate-limit-llm` flag to throttle OpenAI API calls.

**Details**:
- Add `--rate-limit-llm` boolean flag (action="store_true")
- When enabled, introduce 5-second delay between OpenAI API calls
- Useful when user suspects rate limiting or wants to avoid hitting API quotas
- Implementation location: `src/agent_interrogator/llm.py` (OpenAILLM class methods)

**Example Usage**:
```bash
./agent-interrogator-cli.py --target https://example.com/mcp --rate-limit-llm
```

---

## 2. Rate Limiting - Target MCP Server

**Priority**: TBD

**Description**: Add optional `--rate-limit-target` flag to throttle target MCP server calls.

**Details**:
- Add `--rate-limit-target` boolean flag (action="store_true")
- When enabled, introduce 5-second delay between calls to target MCP server
- Useful for respecting server rate limits or avoiding detection during pentesting
- Implementation location: `agent-interrogator-cli.py` (MCPCallback.__call__ method)

**Example Usage**:
```bash
./agent-interrogator-cli.py --target https://example.com/mcp --rate-limit-target
```

---

## 3. Custom HTTP Headers for Target Server

**Priority**: TBD

**Description**: Allow users to specify custom HTTP headers sent to target MCP server.

**Details**:
- Add `--header` flag (action="append" to allow multiple headers)
- Format validation: Must be `Header-Name: Header Value`
- Use case: Adding pentesting identification headers, custom vendor headers, etc.
- Implementation location: `agent-interrogator-cli.py` (MCPCallback.initialize method)
- Should validate format before attempting connection

**Example Usage**:
```bash
./agent-interrogator-cli.py --target https://example.com/mcp \
  --header "X-Pentest: Pentest_Company_Name" \
  --header "X-Vendor: example.com"
```

**Technical Notes**:
- Parse header string: split on first `: ` (colon + space)
- Validate header name doesn't contain spaces or special chars
- Merge custom headers with existing auth headers (don't override Authorization)

---

## 4. Remove Default HuggingFace Auth Token Behavior

**Priority**: HIGH (Security/Privacy)

**Description**: Stop sending HuggingFace auth tokens by default to non-HuggingFace servers.

**Background**:
- Legacy functionality from when tool was hardcoded for `huggingface.co/mcp`
- Currently tool accepts HF_TOKEN environment variable for any target server
- Security concern: Users' HF tokens being sent to arbitrary servers

**Details**:
- Only use `HF_TOKEN` / `HUGGINGFACE_TOKEN` env vars when target URL contains `huggingface.co`
- For other targets, require explicit `--oauth-token` or `MCP_AUTH_TOKEN` env var
- Update `check_environment_variables()` function in `agent-interrogator-cli.py`
- Add logic to detect HuggingFace domain from target URL

**Implementation Location**:
- `agent-interrogator-cli.py:542-617` (check_environment_variables function)

---

## 5. Documentation Updates

**Priority**: TBD

**Description**: Document new features added in recent development sessions.

**Features to Document**:
- `--guess-auth-type` flag for OAuth detection
- `--oauth-token` flag for pre-authenticated tokens
- `--no-auth` flag for disabling authentication
- OAuth discovery mechanism (WWW-Authenticate headers, .well-known endpoints)
- Enhanced error handling and ExceptionGroup support

**Files to Update**:
- `README.md` - Main usage documentation
- `CLAUDE.md` - Architecture and development notes (if needed)
- Docstrings in `agent-interrogator-cli.py`

---

## 6. Improve File Output Functionality

**Priority**: TBD

**Description**: Enhance JSON and Markdown output to capture full tool execution details.

**Current State**:
- JSON output: Minimal wrapper around profile string
- Markdown output: Basic profile output only
- Missing: Configuration, auth detection results, tool execution metadata

**Desired State**:
- **Markdown output**: Should mirror complete terminal output from entire run
  - Configuration details (target, model, timeout, etc.)
  - Auth detection results (if `--guess-auth-type` used)
  - MCP server info and capabilities
  - Discovery/analysis phase outputs
  - Full agent profile results
  - Timestamps and execution metadata

- **JSON output**: Structured data including
  - All configuration parameters
  - Auth detection results (structured)
  - Server capabilities
  - Discovered tools and resources (full schemas)
  - Interrogation profile (parsed if possible)
  - Execution metadata (timestamps, iteration counts, etc.)

**Implementation Location**:
- `agent-interrogator-cli.py:918-956` (save_output function)

**Technical Considerations**:
- May need to refactor main() to collect all output data
- Consider using rich.Console file output for Markdown
- Ensure sensitive data (tokens) are not written to files

---

## 7. Azure OpenAI Support

**Priority**: LOW (Future Enhancement)

**Description**: Add support for Azure-hosted OpenAI API deployments.

**Details**:
- Allow users to specify custom OpenAI API endpoint URL
- Support Azure OpenAI authentication (API key + deployment name)
- Add CLI flags: `--openai-endpoint`, `--azure-deployment`
- Maintain backward compatibility with standard OpenAI API

**Example Usage**:
```bash
./agent-interrogator-cli.py --target https://example.com/mcp \
  --openai-endpoint https://your-resource.openai.azure.com \
  --azure-deployment your-deployment-name
```

**Implementation Location**:
- `src/agent_interrogator/config.py` (LLMConfig class)
- `src/agent_interrogator/llm.py` (OpenAILLM class initialization)
- `agent-interrogator-cli.py` (argument parsing)

**Technical Notes**:
- Azure OpenAI uses different base URL structure
- May require api-version parameter
- Authentication may differ (check Azure OpenAI SDK docs)

---

## Implementation Notes

- All TODO items should maintain backward compatibility
- Add tests for new features where applicable
- Update help text and examples for new CLI flags
- Follow existing code style and patterns (see CODING_STYLE_RULES.md if available)
- Consider rate limiting implications for professional pentesting use

---

**Last Updated**: 2025-01-19
**Branch**: feature/add-oauth-support