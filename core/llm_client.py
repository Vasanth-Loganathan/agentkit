import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLMClient:
    """Raw API client wrapper around Groq's chat completions for LLM execution."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

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

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message