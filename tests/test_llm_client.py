import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm_client import LLMClient


class LLMClientTests(unittest.TestCase):
    @patch("core.llm_client.Groq")
    def test_generate_builds_expected_payload(self, groq_cls):
        client_mock = MagicMock()
        response_mock = MagicMock()
        response_mock.choices = [MagicMock(message=MagicMock(content="hello"))]
        client_mock.chat.completions.create.return_value = response_mock
        groq_cls.return_value = client_mock

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
            llm_client = LLMClient(model_name="test-model")

        messages = [{"role": "user", "content": "hi"}]
        llm_client.generate(messages)

        kwargs = client_mock.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["messages"], messages)

    @patch("core.llm_client.Groq")
    def test_generate_attaches_tool_schemas(self, groq_cls):
        client_mock = MagicMock()
        response_mock = MagicMock()
        response_mock.choices = [MagicMock(message=MagicMock(content="ok"))]
        client_mock.chat.completions.create.return_value = response_mock
        groq_cls.return_value = client_mock

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
            llm_client = LLMClient(model_name="test-model")

        tools = [{"type": "function", "function": {"name": "demo"}}]
        llm_client.generate([{"role": "user", "content": "hi"}], tools=tools)

        kwargs = client_mock.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tools"], tools)
        self.assertEqual(kwargs["tool_choice"], "auto")


if __name__ == "__main__":
    unittest.main()
