import os
from core.llm_client import LLMClient
from core.tool_registry import ToolRegistry
from core.agent_loop import AgentLoop
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory

def main():
    print("🚀 Booting up AgentKit...")

    # 1. Initialize Memory Engines
    llm_client = LLMClient()
    
    # Short-term memory keeps active context from blowing up
    short_memory = ShortTermMemory(llm_client=llm_client, max_messages=6, tail_keep=2)
    
    # Long-term memory persists domain knowledge across restarts
    long_memory = LongTermMemory(table_name="knowledge_base", persist_dir="./lancedb_data")    
    
    # (Optional) Seed the knowledge base on first run
    long_memory.add_documents([
        "Project AgentKit was initiated to build modular, production-grade AI agent primitives.",
        "Our tech stack uses FastAPI, Uvicorn, and Svelte for frontend rendering.",
        "The standard timeout for database queries is 30 seconds."
    ])

    # 2. Initialize Tool Registry and Register Tools
    registry = ToolRegistry()

    @registry.register
    def search_knowledge_base(query: str) -> str:
        """Searches long-term memory for project details, tech stack info, and company facts."""
        return long_memory.search(query=query, top_k=2)

    @registry.register
    def calculate_math(expression: str) -> str:
        """Evaluates simple mathematical expressions (e.g., '15 * 4')."""
        try:
            # Safe evaluation for basic math
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error calculating: {e}"

    # 3. Wire into the Agent Loop
    agent = AgentLoop(
        llm_client=llm_client,
        tool_registry=registry,
        short_term_memory=short_memory,
        system_prompt=(
            "You are an expert AI assistant. Use tools whenever necessary to answer accurately. "
            "CRITICAL: When using a tool, you must format the XML tags perfectly. "
            "Do NOT add any spaces before the closing </function> tag."
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