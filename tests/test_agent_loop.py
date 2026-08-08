import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agent_loop import AgentLoop
from core.tool_registry import ToolRegistry


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        # Mock short-term memory to return the messages list unmodified during testing
        self.mock_memory = MagicMock()
        self.mock_memory.compact_if_needed.side_effect = lambda msgs: msgs

    def test_run_returns_success_when_llm_returns_text(self):
        registry = ToolRegistry()

        @registry.register
        def multiply(a: float, b: float) -> float:
            """Multiplies two numbers together."""
            return a * b

        llm_client = MagicMock()
        llm_client.generate.return_value = MagicMock(content="42", tool_calls=[])

        agent = AgentLoop(
            llm_client=llm_client,
            tool_registry=registry,
            short_term_memory=self.mock_memory,
            max_steps=1,
        )
        status, response = agent.run("What is 6 times 7?")

        self.assertEqual(status, "SUCCESS")
        self.assertEqual(response, "42")
        self.mock_memory.compact_if_needed.assert_called()

    def test_run_returns_stuck_when_loop_detected(self):
        registry = ToolRegistry()

        @registry.register
        def get_weather(city: str) -> str:
            """Returns weather information for a city."""
            return f"Weather in {city} is sunny."

        llm_client = MagicMock()
        tool_call_mock = MagicMock()
        tool_call_mock.id = "call-1"
        tool_call_mock.function.name = "get_weather"
        tool_call_mock.function.arguments = '{"city": "Paris"}'
        llm_client.generate.return_value = MagicMock(content=None, tool_calls=[tool_call_mock])

        agent = AgentLoop(
            llm_client=llm_client,
            tool_registry=registry,
            short_term_memory=self.mock_memory,
            max_steps=5,
            loop_threshold=1,
        )
        status, response = agent.run("Check the weather in Paris.")

        self.assertEqual(status, "STUCK_IN_LOOP")
        self.assertIn("Loop detected", response)

    def test_run_recovers_from_tool_exception(self):
        registry = ToolRegistry()

        @registry.register
        def divide(a: float, b: float) -> float:
            """Divides a by b."""
            if b == 0:
                raise ValueError("Cannot divide by zero!")
            return a / b

        llm_client = MagicMock()

        # Step 1: LLM triggers division by zero
        tool_call = MagicMock()
        tool_call.id = "call-1"
        tool_call.function.name = "divide"
        tool_call.function.arguments = '{"a": 10, "b": 0}'
        response_tool = MagicMock(content=None, tool_calls=[tool_call])

        # Step 2: LLM receives the runtime error string and provides a self-corrected text response
        response_final = MagicMock(content="Division by zero is undefined.", tool_calls=[])
        llm_client.generate.side_effect = [response_tool, response_final]

        agent = AgentLoop(
            llm_client=llm_client,
            tool_registry=registry,
            short_term_memory=self.mock_memory,
            max_steps=5,
        )
        status, response = agent.run("Divide 10 by 0.")

        self.assertEqual(status, "SUCCESS")
        self.assertEqual(response, "Division by zero is undefined.")

    def test_run_returns_max_steps_reached(self):
        registry = ToolRegistry()

        @registry.register
        def dummy_action(step_num: int) -> str:
            """Dummy action."""
            return f"Done step {step_num}"

        llm_client = MagicMock()

        # Dynamically generate unique tool calls to avoid state-hash loop detection
        def generate_response(messages, tools=None):
            step_count = len(messages)
            tool_call = MagicMock()
            tool_call.id = f"call-{step_count}"
            tool_call.function.name = "dummy_action"
            tool_call.function.arguments = f'{{"step_num": {step_count}}}'
            return MagicMock(content=None, tool_calls=[tool_call])

        llm_client.generate.side_effect = generate_response

        agent = AgentLoop(
            llm_client=llm_client,
            tool_registry=registry,
            short_term_memory=self.mock_memory,
            max_steps=2,
            loop_threshold=5,
        )
        status, response = agent.run("Perform continuous steps.")

        self.assertEqual(status, "MAX_STEPS_REACHED")
        self.assertIn("Reached step limit", response)


if __name__ == "__main__":
    unittest.main()