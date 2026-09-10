import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from core.logging import get_logger
from openai import OpenAI


load_dotenv()
logger = get_logger("llm_client")


class LLMClient:
    def __init__(self, model_name: str = "meta/muse-glimmer-30b"):
        api_key = os.getenv("NVIDIA_API_KEY") 
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
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
            "temperature": 0.3,
        }

        # Only attach tools parameter if tool schemas exist
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("Generating LLM response for %s messages", len(messages))
        response = self.client.chat.completions.create(**kwargs)
        logger.debug("Received LLM response")
        return response.choices[0].message