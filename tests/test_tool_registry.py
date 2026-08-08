import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.tool_registry import ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def test_register_and_execute_tool(self):
        registry = ToolRegistry()

        @registry.register
        def multiply(a: float, b: float) -> float:
            """Multiplies two numbers together."""
            return a * b

        schemas = registry.get_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "multiply")
        self.assertIn("parameters", schemas[0]["function"])

        result = registry.execute("multiply", {"a": 12, "b": 4})
        self.assertEqual(result, "48")

    def test_execute_unknown_tool_returns_error(self):
        registry = ToolRegistry()
        result = registry.execute("missing_tool", {})
        self.assertIn("not registered", result)


if __name__ == "__main__":
    unittest.main()
