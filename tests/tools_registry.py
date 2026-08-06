from core.tool_registry import ToolRegistry

registry = ToolRegistry()

# 1. Register a test function using the decorator syntax
@registry.register
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together and returns the product."""
    return a * b

# 2. Inspect generated JSON schema
print("--- Generated Schema ---")
import json
print(json.dumps(registry.get_schemas(), indent=2))

# 3. Test dynamic execution
print("\n--- Test Tool Execution ---")
output = registry.execute("multiply", {"a": 12, "b": 4})
print("Result of multiply(12, 4):", output)