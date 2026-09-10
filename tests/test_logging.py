import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agent_loop import AgentLoop
from core.tool_registry import ToolRegistry


class LoggingTests(unittest.TestCase):
    def test_agent_loop_logs_loop_detection(self):
        registry = ToolRegistry()

        @registry.register
        def get_weather(city: str) -> str:
            """Returns weather for a city."""
            return f"Weather in {city} is sunny."

        llm_client = MagicMock()
        tool_call_mock = MagicMock()
        tool_call_mock.id = "call-1"
        tool_call_mock.function.name = "get_weather"
        tool_call_mock.function.arguments = '{"city": "Paris"}'

        llm_client.generate.return_value = MagicMock(
            content=None,
            tool_calls=[tool_call_mock],
        )

        agent = AgentLoop(
            llm_client=llm_client,
            tool_registry=registry,
            max_steps=1,
            loop_threshold=1,
        )

        with self.assertLogs(level="WARNING") as captured:
            status, message = agent.run("Check the weather in Paris.")

        self.assertEqual(status, "STUCK_IN_LOOP")
        self.assertTrue(any("Loop detected" in entry for entry in captured.output))


if __name__ == "__main__":
    unittest.main()
