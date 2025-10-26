"""Fuzzing engine for MCP server security validation.

This module provides functionality to systematically fuzz MCP server functions
discovered through the Agent Interrogator. It consumes AgentProfile data and
generates parameter-aware security test cases to identify validation flaws,
logic errors, and Excessive Agency vulnerabilities.

All fuzzing operations are non-destructive by default and designed for
authorized penetration testing engagements.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from mcp import ClientSession

from .models import AgentProfile, Capability, Function, Parameter
from .payloads import PayloadGenerator


class FuzzResultStatus(Enum):
    """Status codes for fuzzing test results."""

    SUCCESS = "success"              # 2xx response
    CLIENT_ERROR = "client_error"    # 4xx response
    SERVER_ERROR = "server_error"    # 5xx response
    UNEXPECTED = "unexpected"        # Unexpected behavior
    ERROR = "error"                  # Test execution error


@dataclass
class FuzzResult:
    """Result of a single fuzzing test."""

    function_name: str
    parameter_name: str
    payload: Any
    payload_type: str
    status: FuzzResultStatus
    http_status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    vulnerability_flags: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def is_potential_vulnerability(self) -> bool:
        """Check if this result indicates a potential vulnerability.

        Returns:
            True if the result shows unexpected behavior suggesting a vulnerability
        """
        # Success with malformed input suggests validation bypass
        if self.status == FuzzResultStatus.SUCCESS and self.payload_type != "normal":
            return True

        # Server errors may indicate injection or logic flaws
        if self.status == FuzzResultStatus.SERVER_ERROR:
            return True

        # Flagged vulnerabilities
        if self.vulnerability_flags:
            return True

        return False


@dataclass
class FuzzTarget:
    """Represents a function to be fuzzed with its input schema."""

    capability_name: str
    function: Function
    test_count: int = 0
    vulnerability_count: int = 0

    def get_full_name(self) -> str:
        """Get the full qualified name of the function.

        Returns:
            String in format 'capability.function'
        """
        return f"{self.capability_name}.{self.function.name}"


class FuzzingEngine:
    """Core fuzzing engine for MCP server security testing."""

    def __init__(
        self,
        agent_profile: AgentProfile,
        mcp_session: ClientSession,
        rate_limit_delay: float = 0.0,
        non_destructive: bool = True,
        max_payloads_per_param: Optional[int] = None,
    ):
        """Initialize the fuzzing engine.

        Args:
            agent_profile: AgentProfile object containing discovered capabilities
            mcp_session: Active MCP ClientSession for invoking functions
            rate_limit_delay: Delay in seconds between MCP requests (default: 0)
            non_destructive: If True, use safe payloads for high-risk functions
            max_payloads_per_param: Limit payloads per parameter (None for all)
        """
        self.agent_profile = agent_profile
        self.mcp_session = mcp_session
        self.rate_limit_delay = rate_limit_delay
        self.non_destructive = non_destructive
        self.max_payloads_per_param = max_payloads_per_param

        self.payload_generator = PayloadGenerator(non_destructive=non_destructive)
        self.fuzz_targets: List[FuzzTarget] = []
        self.results: List[FuzzResult] = []

        self._last_request_time: Optional[float] = None

    def map_targets(self) -> List[FuzzTarget]:
        """Parse AgentProfile and generate fuzzing targets.

        Implements FR-1: Fuzzing Target Identification from SRD.

        Returns:
            List of FuzzTarget objects
        """
        targets = []

        for capability in self.agent_profile.capabilities:
            for function in capability.functions:
                target = FuzzTarget(
                    capability_name=capability.name,
                    function=function,
                )
                targets.append(target)

        self.fuzz_targets = targets
        return targets

    async def fuzz_all_targets(self) -> List[FuzzResult]:
        """Execute fuzzing tests against all mapped targets.

        Returns:
            List of FuzzResult objects
        """
        if not self.fuzz_targets:
            self.map_targets()

        for target in self.fuzz_targets:
            await self.fuzz_target(target)

        return self.results

    async def fuzz_target(self, target: FuzzTarget) -> List[FuzzResult]:
        """Execute fuzzing tests for a specific target function.

        Implements FR-2: Parameter-Aware Payload Generation from SRD.

        Args:
            target: FuzzTarget to fuzz

        Returns:
            List of FuzzResult objects for this target
        """
        target_results = []

        # Check if function is high-risk
        is_high_risk = self.payload_generator.is_high_risk_function(target.function.name)

        # If no parameters, test with empty invocation
        if not target.function.parameters:
            result = await self._execute_fuzz_test(
                target=target,
                param_name="<no_params>",
                payload=None,
                payload_type="empty"
            )
            target_results.append(result)
            target.test_count += 1
            if result.is_potential_vulnerability():
                target.vulnerability_count += 1
        else:
            # Fuzz each parameter
            for param in target.function.parameters:
                param_results = await self._fuzz_parameter(
                    target=target,
                    param=param,
                    is_high_risk=is_high_risk
                )
                target_results.extend(param_results)

        self.results.extend(target_results)
        return target_results

    async def _fuzz_parameter(
        self,
        target: FuzzTarget,
        param: Parameter,
        is_high_risk: bool
    ) -> List[FuzzResult]:
        """Fuzz a specific parameter with generated payloads.

        Args:
            target: FuzzTarget being tested
            param: Parameter to fuzz
            is_high_risk: Whether the function is high-risk

        Returns:
            List of FuzzResult objects
        """
        results = []

        # Generate payloads for this parameter
        if is_high_risk and self.non_destructive:
            # Use safe canary values for high-risk functions
            payloads = [self.payload_generator.get_safe_test_value(param.type)]
            payload_type = "canary"
        else:
            # Generate full payload set
            payloads = self.payload_generator.generate_for_parameter(
                param_name=param.name,
                param_type=param.type,
                is_required=param.required
            )
            payload_type = "fuzzing"

        # Limit payloads if configured
        if self.max_payloads_per_param:
            payloads = payloads[:self.max_payloads_per_param]

        # Execute test for each payload
        for payload in payloads:
            result = await self._execute_fuzz_test(
                target=target,
                param_name=param.name,
                payload=payload,
                payload_type=payload_type
            )
            results.append(result)
            target.test_count += 1
            if result.is_potential_vulnerability():
                target.vulnerability_count += 1

        return results

    async def _execute_fuzz_test(
        self,
        target: FuzzTarget,
        param_name: str,
        payload: Any,
        payload_type: str
    ) -> FuzzResult:
        """Execute a single fuzzing test.

        Implements FR-3: MCP Function Invocation from SRD.

        Args:
            target: FuzzTarget being tested
            param_name: Name of parameter being fuzzed
            payload: Fuzz payload to send
            payload_type: Type of payload ('fuzzing', 'canary', 'empty')

        Returns:
            FuzzResult object
        """
        # Apply rate limiting
        if self.rate_limit_delay > 0 and self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)

        start_time = time.time()

        try:
            # Build arguments dict
            if param_name == "<no_params>":
                arguments = {}
            else:
                # Build minimal argument set with just the fuzzed parameter
                # In real scenarios, you may need to provide required params
                arguments = {param_name: payload}

            # Invoke MCP function
            # Implements FR-3: Serialization and Execution
            response = await self.mcp_session.call_tool(
                name=target.function.name,
                arguments=arguments
            )

            execution_time = time.time() - start_time
            self._last_request_time = time.time()

            # Implements FR-4: Response Analysis
            result = self._analyze_response(
                target=target,
                param_name=param_name,
                payload=payload,
                payload_type=payload_type,
                response=response,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self._last_request_time = time.time()

            result = FuzzResult(
                function_name=target.get_full_name(),
                parameter_name=param_name,
                payload=payload,
                payload_type=payload_type,
                status=FuzzResultStatus.ERROR,
                error_message=str(e),
                execution_time=execution_time
            )

        return result

    def _analyze_response(
        self,
        target: FuzzTarget,
        param_name: str,
        payload: Any,
        payload_type: str,
        response: Any,
        execution_time: float
    ) -> FuzzResult:
        """Analyze MCP response for security issues.

        Implements FR-4: Vulnerability Detection from SRD.

        Args:
            target: FuzzTarget being tested
            param_name: Parameter name
            payload: Payload sent
            payload_type: Type of payload
            response: MCP response object
            execution_time: Time taken for request

        Returns:
            FuzzResult with analysis
        """
        vulnerability_flags = []

        # Extract response data
        # MCP responses have 'content' attribute with list of content items
        response_body = ""
        if hasattr(response, "content"):
            for content_item in response.content:
                if hasattr(content_item, "text"):
                    response_body += content_item.text

        # Check for vulnerability indicators
        # FR-4: Flag responses that reveal internal function details or stack traces
        if any(indicator in response_body.lower() for indicator in ["traceback", "stack trace", "exception", "error:"]):
            vulnerability_flags.append("ERROR_DISCLOSURE")

        # Success with malformed input suggests validation bypass
        if payload_type == "fuzzing" and not response.isError:
            vulnerability_flags.append("VALIDATION_BYPASS")

        # Determine status
        if hasattr(response, "isError") and response.isError:
            status = FuzzResultStatus.CLIENT_ERROR
        else:
            status = FuzzResultStatus.SUCCESS

        result = FuzzResult(
            function_name=target.get_full_name(),
            parameter_name=param_name,
            payload=payload,
            payload_type=payload_type,
            status=status,
            response_body=response_body[:1000],  # Truncate for storage
            vulnerability_flags=vulnerability_flags,
            execution_time=execution_time
        )

        return result

    def get_vulnerability_summary(self) -> Dict[str, Any]:
        """Generate summary of discovered vulnerabilities.

        Returns:
            Dictionary with vulnerability statistics and findings
        """
        vulnerabilities = [r for r in self.results if r.is_potential_vulnerability()]

        summary = {
            "total_tests": len(self.results),
            "total_vulnerabilities": len(vulnerabilities),
            "targets_tested": len(self.fuzz_targets),
            "vulnerable_functions": len([t for t in self.fuzz_targets if t.vulnerability_count > 0]),
            "vulnerability_breakdown": {},
            "high_priority_findings": [],
        }

        # Breakdown by vulnerability type
        for vuln in vulnerabilities:
            for flag in vuln.vulnerability_flags:
                if flag not in summary["vulnerability_breakdown"]:
                    summary["vulnerability_breakdown"][flag] = 0
                summary["vulnerability_breakdown"][flag] += 1

        # High priority findings (multiple flags or server errors)
        for vuln in vulnerabilities:
            if len(vuln.vulnerability_flags) > 1 or vuln.status == FuzzResultStatus.SERVER_ERROR:
                summary["high_priority_findings"].append({
                    "function": vuln.function_name,
                    "parameter": vuln.parameter_name,
                    "payload": str(vuln.payload)[:100],
                    "flags": vuln.vulnerability_flags,
                    "status": vuln.status.value,
                })

        return summary

    def export_results(self, filepath: str, format: str = "json") -> None:
        """Export fuzzing results to file.

        Implements FR-5: Recursive Interrogation Preparation from SRD.

        Args:
            filepath: Path to output file
            format: Output format ('json' or 'markdown')
        """
        if format == "json":
            self._export_json(filepath)
        elif format == "markdown":
            self._export_markdown(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, filepath: str) -> None:
        """Export results as JSON."""
        data = {
            "summary": self.get_vulnerability_summary(),
            "targets": [
                {
                    "function": t.get_full_name(),
                    "test_count": t.test_count,
                    "vulnerability_count": t.vulnerability_count,
                }
                for t in self.fuzz_targets
            ],
            "results": [
                {
                    "function": r.function_name,
                    "parameter": r.parameter_name,
                    "payload": str(r.payload)[:200],  # Truncate for readability
                    "payload_type": r.payload_type,
                    "status": r.status.value,
                    "http_status": r.http_status_code,
                    "response": r.response_body[:500] if r.response_body else None,
                    "error": r.error_message,
                    "vulnerability_flags": r.vulnerability_flags,
                    "execution_time": r.execution_time,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _export_markdown(self, filepath: str) -> None:
        """Export results as Markdown."""
        summary = self.get_vulnerability_summary()

        with open(filepath, "w") as f:
            f.write("# MCP Fuzzing Results\n\n")

            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- Total Tests: {summary['total_tests']}\n")
            f.write(f"- Total Vulnerabilities: {summary['total_vulnerabilities']}\n")
            f.write(f"- Targets Tested: {summary['targets_tested']}\n")
            f.write(f"- Vulnerable Functions: {summary['vulnerable_functions']}\n\n")

            # Vulnerability breakdown
            if summary['vulnerability_breakdown']:
                f.write("## Vulnerability Breakdown\n\n")
                for vuln_type, count in summary['vulnerability_breakdown'].items():
                    f.write(f"- {vuln_type}: {count}\n")
                f.write("\n")

            # High priority findings
            if summary['high_priority_findings']:
                f.write("## High Priority Findings\n\n")
                for finding in summary['high_priority_findings']:
                    f.write(f"### {finding['function']}\n\n")
                    f.write(f"- Parameter: {finding['parameter']}\n")
                    f.write(f"- Payload: `{finding['payload']}`\n")
                    f.write(f"- Flags: {', '.join(finding['flags'])}\n")
                    f.write(f"- Status: {finding['status']}\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            for target in self.fuzz_targets:
                if target.vulnerability_count > 0:
                    f.write(f"### {target.get_full_name()}\n\n")
                    f.write(f"- Tests: {target.test_count}\n")
                    f.write(f"- Vulnerabilities: {target.vulnerability_count}\n\n")

                    # Get results for this target
                    target_results = [r for r in self.results if r.function_name == target.get_full_name() and r.is_potential_vulnerability()]
                    for result in target_results:
                        f.write(f"#### {result.parameter_name}\n\n")
                        f.write(f"- Payload: `{str(result.payload)[:100]}`\n")
                        f.write(f"- Status: {result.status.value}\n")
                        if result.vulnerability_flags:
                            f.write(f"- Flags: {', '.join(result.vulnerability_flags)}\n")
                        f.write("\n")