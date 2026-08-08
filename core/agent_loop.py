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

    def run(self, user_prompt: str) -> Tuple[str, str]:
        """Executes the agent loop for a prompt.

        Returns:
            Tuple[status, final_response_text]
            where status is one of: 'SUCCESS', 'STUCK_IN_LOOP', 'MAX_STEPS_REACHED'
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        seen_hashes: Dict[str, int] = {}
        step = 0

        while step < self.max_steps:
            step += 1
            logger.info("Starting loop step %s/%s", step, self.max_steps)

            # 0. COMPACT: Automatically manage short-term context history
            messages = self.short_term_memory.compact_if_needed(messages)

            schemas = self.tool_registry.get_schemas()
            logger.debug("Sending request to LLM with %s tool schemas", len(schemas))
            response = self.llm_client.generate(messages, tools=schemas if schemas else None)

            # 1. EVALUATE: Check for requested tool calls
            if response.tool_calls:
                messages.append(response)

                for tool_call in response.tool_calls:
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments

                    # Safely parse JSON arguments from the model
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError as e:
                        tool_output = f"Error: Malformed JSON arguments provided by model: {str(e)}"
                        logger.warning("Tool call %s had malformed JSON arguments: %s", func_name, str(e))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output,
                        })
                        continue

                    # Compute state-hash for duplicate detection
                    call_hash = self._compute_call_hash(func_name, args)
                    seen_hashes[call_hash] = seen_hashes.get(call_hash, 0) + 1

                    # 2. LOOP DETECTION CHECK
                    if seen_hashes[call_hash] >= self.loop_threshold:
                        warning_msg = (
                            f"Loop detected! Tool '{func_name}' with args {args} "
                            f"has been called {seen_hashes[call_hash]} times. Breaking execution."
                        )
                        logger.warning("Loop detected for tool %s with args %s", func_name, args)
                        return "STUCK_IN_LOOP", warning_msg

                    # 3. ACT & OBSERVE (With Exception Recovery)
                    logger.info("Executing tool %s with args %s", func_name, args)
                    try:
                        tool_output = self.tool_registry.execute(func_name, args)
                    except Exception as exc:
                        # Catch unexpected tool runtime crashes safely
                        tool_output = f"Runtime Error executing tool '{func_name}': {str(exc)}"

                    logger.info("Tool %s returned: %s", func_name, tool_output)

                    # Feed observation back into history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    })
            else:
                # Model returned a final text response
                messages.append({"role": "assistant", "content": response.content})
                logger.info("Agent completed successfully with final response")
                return "SUCCESS", response.content

        logger.warning("Agent reached max steps without completion")
        return "MAX_STEPS_REACHED", f"Agent stopped: Reached step limit ({self.max_steps})."