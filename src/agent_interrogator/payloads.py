"""Non-destructive fuzzing payloads for security testing.

This module provides a collection of non-destructive fuzzing payloads designed
to identify validation flaws, logic errors, and security vulnerabilities without
causing actual harm to target systems.

All payloads are designed for authorized penetration testing and security research.
"""

from typing import Any, Dict, List

# String-based fuzzing payloads
STRING_PAYLOADS: List[str] = [
    # Empty and boundary tests
    "",
    " ",
    "   ",

    # Extremely long strings (boundary testing)
    "A" * 1000,
    "A" * 10000,
    "A" * 100000,

    # XSS canaries (non-destructive)
    "<script>alert('XSS_TEST')</script>",
    "<img src=x onerror=alert('XSS_TEST')>",
    "javascript:alert('XSS_TEST')",
    "<svg onload=alert('XSS_TEST')>",

    # SQL injection canaries (non-destructive read-only tests)
    "' OR '1'='1",
    "1' OR '1'='1' --",
    "admin' --",
    "' UNION SELECT NULL--",
    "1' AND '1'='1",

    # Command injection canaries (non-destructive)
    "; echo FUZZ_TEST_COMMAND",
    "| echo FUZZ_TEST_COMMAND",
    "`echo FUZZ_TEST_COMMAND`",
    "$(echo FUZZ_TEST_COMMAND)",
    "&& echo FUZZ_TEST_COMMAND",

    # Path traversal payloads
    "../",
    "../../",
    "../../../",
    "..\\",
    "..\\..\\",
    "....//",
    "....\\\\",

    # JSON structure payloads
    '{"key": "value"}',
    '{"nested": {"key": "value"}}',
    '[{"array": "item"}]',

    # XML structure payloads
    "<root><node>value</node></root>",
    "<?xml version='1.0'?><root/>",

    # Special characters and encoding
    "\x00",
    "\n\r\n",
    "\t\t\t",
    "\\x00\\x01\\x02",
    "%00%0A%0D",

    # Unicode and internationalization
    "\u0000",
    "\u202e",
    "\ufeff",

    # Format string vulnerabilities
    "%s%s%s%s%s",
    "%x%x%x%x%x",
    "%n%n%n%n%n",
]

# Integer/number fuzzing payloads
INTEGER_PAYLOADS: List[int] = [
    # Boundary values
    0,
    -1,
    1,

    # Integer limits (32-bit)
    2147483647,      # MAX_INT32
    -2147483648,     # MIN_INT32
    2147483648,      # MAX_INT32 + 1
    -2147483649,     # MIN_INT32 - 1

    # Integer limits (64-bit)
    9223372036854775807,   # MAX_INT64
    -9223372036854775808,  # MIN_INT64
]

# Float/number fuzzing payloads (for type confusion)
FLOAT_PAYLOADS: List[float] = [
    0.0,
    -0.0,
    0.1,
    -0.1,
    1.1,
    -1.1,
    999999999.999999,
    -999999999.999999,
]

# String representations of numbers (type confusion)
NUMBER_STRING_PAYLOADS: List[str] = [
    "0",
    "-1",
    "1.5",
    "-1.5",
    "999999999",
    "2147483648",
    "NaN",
    "Infinity",
    "-Infinity",
]

# Boolean fuzzing payloads
BOOLEAN_PAYLOADS: List[Any] = [
    True,
    False,
]

# Boolean-like string payloads (case sensitivity and type confusion)
BOOLEAN_STRING_PAYLOADS: List[str] = [
    "true",
    "false",
    "True",
    "False",
    "TRUE",
    "FALSE",
    "tRuE",
    "fAlSe",
    "1",
    "0",
    "yes",
    "no",
    "on",
    "off",
]

# Array/list fuzzing payloads
ARRAY_PAYLOADS: List[List[Any]] = [
    # Empty array
    [],

    # Single element
    ["single"],
    [1],

    # Multiple elements
    ["a", "b", "c"],
    [1, 2, 3],

    # Excessive elements (boundary testing)
    list(range(1000)),

    # Mixed types (type confusion)
    [1, "string", True, None, 3.14],

    # Nested structures
    [["nested"], ["array"]],
    [{"key": "value"}],
]

# Object/dict fuzzing payloads
OBJECT_PAYLOADS: List[Dict[str, Any]] = [
    # Empty object
    {},

    # Simple object
    {"key": "value"},

    # Multiple keys
    {"key1": "value1", "key2": "value2", "key3": "value3"},

    # Nested objects
    {"nested": {"key": "value"}},

    # Mixed types
    {"string": "value", "int": 123, "bool": True, "null": None},

    # Array values
    {"array": [1, 2, 3]},

    # Special characters in keys
    {"key with spaces": "value"},
    {"key.with.dots": "value"},
    {"key/with/slashes": "value"},
]

# IDOR (Insecure Direct Object Reference) test payloads
# These are used for testing authorization boundaries
IDOR_PAYLOADS: List[Any] = [
    # Sequential IDs
    "1",
    "2",
    "3",
    "100",
    "999",
    "1000",

    # Common test IDs
    "0",
    "-1",
    "admin",
    "test",
    "user",
    "guest",

    # UUID-like formats
    "00000000-0000-0000-0000-000000000000",
    "11111111-1111-1111-1111-111111111111",
]

# Null and undefined payloads
NULL_PAYLOADS: List[Any] = [
    None,
    "null",
    "NULL",
    "undefined",
    "UNDEFINED",
]


class PayloadGenerator:
    """Generate parameter-specific fuzzing payloads based on type and context."""

    def __init__(self, non_destructive: bool = True):
        """Initialize the payload generator.

        Args:
            non_destructive: If True, filter out potentially destructive payloads
        """
        self.non_destructive = non_destructive

    def generate_for_parameter(
        self,
        param_name: str,
        param_type: str,
        is_required: bool = True
    ) -> List[Any]:
        """Generate fuzzing payloads for a specific parameter.

        Args:
            param_name: Name of the parameter (e.g., 'userId', 'email')
            param_type: Expected type (e.g., 'string', 'integer', 'boolean')
            is_required: Whether the parameter is required

        Returns:
            List of fuzzing payloads appropriate for this parameter
        """
        payloads = []
        param_type_lower = param_type.lower()
        param_name_lower = param_name.lower()

        # String type fuzzing
        if "string" in param_type_lower or "str" in param_type_lower:
            payloads.extend(STRING_PAYLOADS)

            # Add IDOR tests if parameter name suggests ID context
            if any(id_hint in param_name_lower for id_hint in ["id", "user", "account", "customer"]):
                payloads.extend(IDOR_PAYLOADS)

            # Add type confusion payloads
            payloads.extend(NUMBER_STRING_PAYLOADS)
            payloads.extend(BOOLEAN_STRING_PAYLOADS)

        # Integer/number type fuzzing
        elif any(num_type in param_type_lower for num_type in ["int", "integer", "number", "long"]):
            payloads.extend(INTEGER_PAYLOADS)
            payloads.extend(FLOAT_PAYLOADS)  # Type confusion

            # String representations for type confusion
            payloads.extend(NUMBER_STRING_PAYLOADS)

            # IDOR tests for numeric IDs
            if "id" in param_name_lower:
                payloads.extend([int(x) if x.isdigit() else x for x in IDOR_PAYLOADS if isinstance(x, str) and (x.isdigit() or x == "-1")])

        # Boolean type fuzzing
        elif "bool" in param_type_lower:
            payloads.extend(BOOLEAN_PAYLOADS)
            payloads.extend(BOOLEAN_STRING_PAYLOADS)  # Type confusion
            payloads.extend([0, 1])  # Numeric representation

        # Array/list type fuzzing
        elif any(arr_type in param_type_lower for arr_type in ["array", "list", "[]"]):
            payloads.extend(ARRAY_PAYLOADS)

        # Object/dict type fuzzing
        elif any(obj_type in param_type_lower for obj_type in ["object", "dict", "map", "{}"]):
            payloads.extend(OBJECT_PAYLOADS)

        # Default: treat as string if type is unknown
        else:
            payloads.extend(STRING_PAYLOADS)

        # Add null/undefined tests if parameter is not required
        if not is_required:
            payloads.extend(NULL_PAYLOADS)

        return payloads

    def is_high_risk_function(self, function_name: str) -> bool:
        """Check if a function name suggests high-risk operations.

        High-risk functions should use canary data instead of real operations.

        Args:
            function_name: Name of the function to check

        Returns:
            True if function is high-risk, False otherwise
        """
        high_risk_keywords = [
            "delete",
            "remove",
            "drop",
            "destroy",
            "modify",
            "update",
            "execute",
            "exec",
            "run",
            "shutdown",
            "stop",
            "kill",
            "create_user",
            "add_user",
            "grant",
            "revoke",
            "chmod",
            "chown",
        ]

        function_name_lower = function_name.lower()
        return any(keyword in function_name_lower for keyword in high_risk_keywords)

    def get_safe_test_value(self, param_type: str) -> Any:
        """Get a safe canary value for high-risk function testing.

        Args:
            param_type: Expected parameter type

        Returns:
            A safe test value that won't cause actual harm
        """
        param_type_lower = param_type.lower()

        if "string" in param_type_lower or "str" in param_type_lower:
            return "FUZZ_TEST_CANARY_VALUE"
        elif any(num_type in param_type_lower for num_type in ["int", "integer", "number"]):
            return -1
        elif "bool" in param_type_lower:
            return False
        elif any(arr_type in param_type_lower for arr_type in ["array", "list"]):
            return ["FUZZ_TEST_CANARY"]
        elif any(obj_type in param_type_lower for obj_type in ["object", "dict"]):
            return {"fuzz_test": "canary"}
        else:
            return "FUZZ_TEST_CANARY_VALUE"