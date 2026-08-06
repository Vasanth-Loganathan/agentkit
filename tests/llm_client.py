from core.llm_client import LLMClient
from core.tool_registry import ToolRegistry

# Initialize registry and register our test tool
registry = ToolRegistry()

@registry.register
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together and returns the product."""
    return a * b

# Initialize LLM Client
client = LLMClient()

# Test 1: Standard conversation message
print("--- Test 1: Plain Text Prompt ---")
messages = [{"role": "user", "content": "What is the capital of France?"}]
response = client.generate(messages)
print("Response Text:", response.content)

# Test 2: Prompt requiring tool call
print("\n--- Test 2: Tool Call Prompt ---")
messages = [{"role": "user", "content": "What is 18 multiplied by 6?"}]
schemas = registry.get_schemas()

response = client.generate(messages, tools=schemas)

# Verify if Groq returned a tool call request instead of plain text
if response.tool_calls:
    print("Model requested tool call!")
    for tool_call in response.tool_calls:
        print("  Function Name:", tool_call.function.name)
        print("  Arguments JSON:", tool_call.function.arguments)
else:
    print("Response Text:", response.content)