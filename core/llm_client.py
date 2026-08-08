import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from core.logging import get_logger

try:
    from groq import Groq
except ImportError:  # pragma: no cover - handled for test environments
    Groq = None

load_dotenv()
logger = get_logger("llm_client")


class LLMClient:
    """Raw API client wrapper around Groq's chat completions for LLM execution."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        if Groq is None:
            raise ImportError("groq package is required to use LLMClient")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        logger.debug("Initialized LLMClient with model %s", model_name)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Sends structured conversation history and optional tool schemas to Groq.
        
        Returns the raw model message object.
        """
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }

        # Only attach tools parameter if tool schemas exist
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("Generating LLM response for %s messages", len(messages))
        response = self.client.chat.completions.create(**kwargs)
        logger.debug("Received LLM response")
        return response.choices[0].message