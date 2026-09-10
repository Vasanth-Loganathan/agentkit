import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm_client import LLMClient
from memory.short_term import ShortTermMemory


def test_short_term_memory_compaction():
    print("================ TEST: Short-Term Memory Compaction ================\n")

    # 1. Initialize LLM Client and ShortTermMemory (compacts if messages > 5)
    client = LLMClient()
    memory = ShortTermMemory(llm_client=client, max_messages=5, tail_keep=2)

    # 2. Simulate an 8-message execution history
    simulated_messages = [
        {"role": "system", "content": "You are a helpful research assistant."},  # HEAD [0]
        {"role": "user", "content": "Analyze company sales data for Q1, Q2, Q3, and Q4."},  # HEAD [1]
        {"role": "assistant", "content": "Checking Q1 sales data via API..."},  # MIDDLE
        {"role": "tool", "content": "Q1 Sales: $150,000"},  # MIDDLE
        {"role": "assistant", "content": "Checking Q2 sales data via API..."},  # MIDDLE
        {"role": "tool", "content": "Q2 Sales: $200,000"},  # MIDDLE
        {"role": "assistant", "content": "Checking Q3 sales data via API..."},  # TAIL [-2]
        {"role": "tool", "content": "Q3 Sales: $180,000"},  # TAIL [-1]
    ]

    print(f"Original message count: {len(simulated_messages)}")

    # 3. Trigger memory compaction
    compacted_messages = memory.compact_if_needed(simulated_messages)

    print(f"Compacted message count: {len(compacted_messages)}\n")

    # 4. Display the resulting structure
    print("--- Compacted Message Structure ---")
    for idx, msg in enumerate(compacted_messages):
        role = msg.get("role")
        content = msg.get("content", "")
        # Truncate content preview for clean console output
        preview = content[:120] + "..." if len(str(content)) > 120 else content
        print(f"[{idx}] Role: {role} | Content: {preview}")

    # Assertions
    assert len(compacted_messages) < len(simulated_messages), "Compaction failed to reduce message count."
    assert compacted_messages[0]["role"] == "system", "Head (system prompt) was modified."
    assert compacted_messages[1]["role"] == "user", "Head (initial goal) was modified."
    assert "[PREVIOUS CONTEXT SUMMARY]:" in compacted_messages[2]["content"], "Middle summary block missing."
    assert compacted_messages[-1]["content"] == "Q3 Sales: $180,000", "Tail (recent state) was modified."

    print("\n✅ ShortTermMemory compaction test PASSED!")


if __name__ == "__main__":
    test_short_term_memory_compaction()