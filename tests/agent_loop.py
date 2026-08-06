from core.llm_client import LLMClient
from core.tool_registry import ToolRegistry
from core.agent_loop import AgentLoop

# Initialize components
registry = ToolRegistry()

# Register test tools
@registry.register
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b

@registry.register
def add(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b

client = LLMClient()
agent = AgentLoop(llm_client=client, tool_registry=registry)

# Test 1: Single tool call
print("================ TEST 1: Single Tool ================")
answer1 = agent.run("What is 15 multiplied by 4?")
print(f"\nFINAL ANSWER: {answer1}\n")

# Test 2: Chained tool calls (Requires multiply then add)
print("================ TEST 2: Chained Tools ================")
answer2 = agent.run("First multiply 12 by 5, then add 10 to that result.")
print(f"\nFINAL ANSWER: {answer2}")