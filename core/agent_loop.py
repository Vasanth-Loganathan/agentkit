import json
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from core.llm_client import LLMClient
from core.tool_registry import ToolRegistry
from core.logging import get_logger
from memory.short_term import ShortTermMemory


logger = get_logger("agent_loop")


class AgentLoop:
    """Robust Think-Act-Observe control loop with state-hash loop detection,
    exception recovery, and short-term memory compaction."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        short_term_memory: Optional[ShortTermMemory] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 5,
        loop_threshold: int = 2,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.short_term_memory = short_term_memory or ShortTermMemory(llm_client=llm_client)
        self.system_prompt = system_prompt or "You are a helpful AI assistant with access to tools."
        self.max_steps = max_steps
        self.loop_threshold = loop_threshold
        logger.debug("Initialized AgentLoop with max_steps=%s loop_threshold=%s", max_steps, loop_threshold)

    def _compute_call_hash(self, func_name: str, args: Dict[str, Any]) -> str:
        """Generates a unique SHA256 hash for a tool call to track duplicate executions."""
        canonical_str = f"{func_name}:{json.dumps(args, sort_keys=True)}"
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def run(self, user_input) -> Tuple[str, str]:
        """Executes the agent loop for a prompt using SQLite Episodic Memory."""
        
        # 1. THE SYSTEM PROMPT CHECK
        # If this is a brand new chat, inject the system prompt as the first message.
        if len(self.short_term_memory.messages) == 0:
            self.short_term_memory.add_message({"role": "system", "content": self.system_prompt})

        # 2. COMMIT USER INPUT TO SQLITE
        # Check if it's already a dictionary (multimodal payload) or a plain string
        if isinstance(user_input, dict):
            formatted_message = user_input
        else:
            formatted_message = {"role": "user", "content": user_input}

        self.short_term_memory.add_message(formatted_message)

        seen_hashes: Dict[str, int] = {}
        step = 0

        while step < self.max_steps:
            step += 1
            logger.info("Starting loop step %s/%s", step, self.max_steps)

            # 3. HYDRATE CONTEXT (Sticky Anchor + Sliding Window)
            messages_for_llm = self.short_term_memory.get_messages()
            
            # NEW: Strip internal AgentKit keys before sending to Groq
            clean_messages = []
            for msg in messages_for_llm:
                clean_msg = msg.copy()
                clean_msg.pop('local_path', None)
                clean_messages.append(clean_msg)

            schemas = self.tool_registry.get_schemas()
            logger.debug("Sending request to LLM with %s tool schemas", len(schemas))
            
            # Send the CLEANED context to the LLM
            response = self.llm_client.generate(clean_messages, tools=schemas if schemas else None)
            
            # 4. EVALUATE: Check for requested tool calls
            if response.tool_calls:
                # Save the LLM's tool request to SQLite
                # (Assuming response can be cast to dict or has a method to get the message format)
                tool_request_msg = {"role": "assistant", "tool_calls": [t.model_dump() if hasattr(t, 'model_dump') else t for t in response.tool_calls]}
                # Note: Adjust the above line based on how your llm_client structures the response object natively
                self.short_term_memory.add_message(tool_request_msg)

                for tool_call in response.tool_calls:
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments

                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError as e:
                        tool_output = f"Error: Malformed JSON arguments: {str(e)}"
                        logger.warning("Tool call %s had malformed JSON arguments", func_name)
                        
                        self.short_term_memory.add_message({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output,
                        })
                        continue

                    # Loop detection
                    call_hash = self._compute_call_hash(func_name, args)
                    seen_hashes[call_hash] = seen_hashes.get(call_hash, 0) + 1

                    if seen_hashes[call_hash] >= self.loop_threshold:
                        warning_msg = f"Loop detected! Breaking execution."
                        logger.warning(warning_msg)
                        return "STUCK_IN_LOOP", warning_msg

                    # Execute Tool
                    logger.info("Executing tool %s with args %s", func_name, args)
                    try:
                        tool_output = self.tool_registry.execute(func_name, args)
                    except Exception as exc:
                        tool_output = f"Runtime Error executing tool '{func_name}': {str(exc)}"

                    logger.info("Tool %s returned: %s", func_name, tool_output)

                    # Save the tool's observation back into SQLite
                    self.short_term_memory.add_message({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    })
            else:
                # Model returned a final text response
                self.short_term_memory.add_message({"role": "assistant", "content": response.content})
                logger.info("Agent completed successfully")
                return "SUCCESS", response.content

        logger.warning("Agent reached max steps without completion")
        return "MAX_STEPS_REACHED", f"Agent stopped: Reached step limit ({self.max_steps})."