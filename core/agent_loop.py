import json
from typing import Optional, List, Dict, Any
from core.llm_client import LLMClient
from core.tool_registry import ToolRegistry


class AgentLoop:
    """The core Think-Act-Observe control loop for driving agent reasoning and tool execution."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        max_steps: int = 5,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt or "You are a helpful AI assistant with access to tools."
        self.max_steps = max_steps

    def run(self, user_prompt: str) -> str:
        """Executes the agent loop for a given user prompt."""
        # Initialize conversation state
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        step = 0
        while step < self.max_steps:
            step += 1
            print(f"\n--- [Loop Step {step}/{self.max_steps}] Thinking... ---")

            # 1. THINK: Call model with available schemas
            schemas = self.tool_registry.get_schemas()
            response = self.llm_client.generate(messages, tools=schemas if schemas else None)

            # 2. EVALUATE: Check for tool calls vs final text
            if response.tool_calls:
                # Append assistant's tool-call response to context
                messages.append(response)

                # 3. ACT & OBSERVE: Process each tool call requested by the model
                for tool_call in response.tool_calls:
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

                    print(f"-> ACT: Executing tool '{func_name}' with args {args}")
                    tool_output = self.tool_registry.execute(func_name, args)
                    print(f"<- OBSERVE: Tool output: {tool_output}")

                    # Feed tool output back into context as a tool role message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    })
            else:
                # Final text answer reached
                messages.append({"role": "assistant", "content": response.content})
                return response.content

        return f"Agent stopped: Reached maximum step limit ({self.max_steps}) without completion."