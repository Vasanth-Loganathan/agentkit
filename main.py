import os
from core.llm_client import LLMClient
from core.tool_registry import ToolRegistry
from core.agent_loop import AgentLoop
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from tools import register_agent_tools

def main():
    print("🚀 Booting up AgentKit...")

    # 1. Initialize Memory Engines
    llm_client = LLMClient()
    
    # Short-term memory keeps active context managed
    short_memory = ShortTermMemory(llm_client=llm_client, max_messages=6, tail_keep=2)
    
    # Long-term memory persists domain knowledge across restarts
    long_memory = LongTermMemory(table_name="knowledge_base", persist_dir="./lancedb_data")    
    
    # 2. Initialize Tool Registry and Register Tools
    registry = ToolRegistry()

    register_agent_tools(registry, long_memory)
    
    # 3. Wire into the Agent Loop
    agent = AgentLoop(
        llm_client=llm_client,
        tool_registry=registry,
        short_term_memory=short_memory,
        system_prompt=(
            "You are an expert AI assistant. Use tools whenever necessary to answer accurately. "
        ),
        max_steps=5
    )

    print("\n✅ AgentKit is ready! Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    # 4. Interactive Chat Loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down AgentKit. Goodbye!")
                break
            
            if not user_input.strip():
                continue

            # Run the Think-Act-Observe loop
            status, response = agent.run(user_input)
            
            print(f"\nAgent [{status}]:\n{response}")

        except KeyboardInterrupt:
            print("\nShutting down AgentKit. Goodbye!")
            break

if __name__ == "__main__":
    main()