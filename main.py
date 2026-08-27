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
    
    #Short-term memory 
    short_memory = ShortTermMemory(llm_client=llm_client, db_path="chat_history.db", max_tokens=16000)
    
    # Long-term memory persists domain knowledge across restarts
    long_memory = LongTermMemory(table_name="knowledge_base", persist_dir="./lancedb_data")    
    
    # 2. Initialize Tool Registry and Register Tools
    registry = ToolRegistry()
    register_agent_tools(registry, long_memory)
    
    # Boot Menu for Session Selection
    sessions = short_memory.get_all_sessions()
    
    if not sessions:
        print("No previous sessions found. Starting a new chat.")
        short_memory.load_session(None)
    else:
        print("\n=== Previous Chat Sessions ===")
        # Display the 5 most recent sessions with their Titles
        for i, (sess_id, msg_count, last_active, title) in enumerate(sessions[:5]):
            print(f"  [{i+1}] \"{title}\"")
            print(f"      (ID: {sess_id} | Messages: {msg_count} | Last Active: {last_active})")
        print("  [N] Start a New Chat")
        
        choice = input("\nSelect a session number or press 'N': ").strip().upper()
        
        if choice == 'N' or choice == '':
            short_memory.load_session(None)
            print(f"\n=> Started new session: {short_memory.session_id}")
        else:
            try:
                idx = int(choice) - 1
                selected_session = sessions[idx][0]
                short_memory.load_session(selected_session)
                print(f"\n=> Loaded session: {selected_session}")
            except (ValueError, IndexError):
                print("\n=> Invalid choice. Starting a new chat instead.")
                short_memory.load_session(None)

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