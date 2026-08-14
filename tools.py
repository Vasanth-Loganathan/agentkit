# tools.py
from core.tool_registry import ToolRegistry
from memory.long_term import LongTermMemory

def register_agent_tools(registry: ToolRegistry, long_memory: LongTermMemory):
    """Registers all available tools to the provided tool registry."""

    @registry.register
    def search_knowledge_base(query: str) -> str:
        """Searches long-term memory to retrieve details about the user or project."""
        return long_memory.search(query=query, top_k=4)

    @registry.register
    def add_to_knowledge_base(information: str, category: str) -> str:
        """Saves new, important information about the user or project into long-term memory.
        
        Args:
            information: The exact fact or detail to remember.
            category: The type of data. You MUST choose exactly one of these: 
                      'user_profile', 'user_preference', 'project_detail', 'tech_stack', 'general_fact'.
        """
        try:
            long_memory.add_documents(
                documents=[information],
                metadatas=[{"source": "agent", "type": category.lower()}]
            )
            return f"Action successful: Saved to long-term memory under category '{category}'."
        except Exception as e:
            return f"Error saving to database: {e}"

    @registry.register
    def update_knowledge_base(record_id: str, new_information: str, category: str) -> str:
        """Updates an existing record in long-term memory.
        
        Args:
            record_id: The exact ID of the document to update (must be retrieved via search first).
            new_information: The complete updated text that will replace the old record.
            category: The type of data. You MUST choose exactly one of these: 
                      'user_profile', 'user_preference', 'project_detail', 'tech_stack', 'general_fact'.
        """
        try:
            long_memory.update(
                doc_id=record_id,
                text=new_information,
                metadata={"source": "agent", "type": category.lower()}
            )
            return f"Action successful: Record {record_id} updated."
        except Exception as e:
            return f"Error updating database: {e}"

    @registry.register
    def delete_from_knowledge_base(record_id: str) -> str:
        """Deletes a specific record from long-term memory by its record ID.
        
        Args:
            record_id: The exact ID of the document to delete (must be retrieved via search first).
        """
        try:
            long_memory.delete(doc_id=record_id)
            return f"Action successful: Record {record_id} deleted."
        except Exception as e:
            return f"Error deleting from database: {e}"

    @registry.register
    def calculate_math(expression: str) -> str:
        """Evaluates simple mathematical expressions (e.g., '15 * 4')."""
        try:
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error calculating: {e}"