import os
import sys
from typing import List, Dict, Any, Optional
from core.llm_client import LLMClient
from core.logging import get_logger


logger = get_logger("short_term_memory")


class ShortTermMemory:
    """Manages active conversation history and compacts middle turns when context boundaries are exceeded."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_messages: int = 6,
        tail_keep: int = 2,
    ):
        self.llm_client = llm_client
        self.max_messages = max_messages
        self.tail_keep = tail_keep

    def compact_if_needed(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compacts the middle portion of conversation history if total message count exceeds max_messages."""
        # Don't compact if history is under limit
        if len(messages) <= self.max_messages:
            return messages

        # 1. Identify Head (System Prompt + Original Goal)
        head = messages[:2]  # messages[0] = system, messages[1] = initial user request

        # 2. Identify Tail (Recent active turns)
        tail = messages[-self.tail_keep:]

        # 3. Identify Middle turns to compress
        middle = messages[2:-self.tail_keep]

        if not middle:
            return messages

        logger.info("Compacting %s intermediate history messages", len(middle))

        # Construct summarization prompt for LLM
        summary_prompt = [
            {
                "role": "system",
                "content": "You are a context compactor. Summarize the key facts, tool calls, and progress made in the provided conversation transcript into a concise paragraph. Omit unnecessary details.",
            },
            {
                "role": "user",
                "content": f"Summarize these intermediate conversation steps:\n{str(middle)}",
            },
        ]

        # Generate summary using our existing LLMClient
        summary_response = self.llm_client.generate(summary_prompt)
        summary_text = summary_response.content if hasattr(summary_response, "content") else str(summary_response)

        compacted_summary_turn = {
            "role": "user",
            "content": f"[PREVIOUS CONTEXT SUMMARY]: {summary_text}",
        }

        # Reconstruct messages array: Head + Compacted Middle + Tail
        reconstructed = head + [compacted_summary_turn] + tail
        logger.info("History reduced from %s to %s turns", len(messages), len(reconstructed))
        return reconstructed