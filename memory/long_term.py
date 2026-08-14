import os
import hashlib
import json
from typing import List, Dict, Any, Optional
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

# 1. Pull the standard lightweight model from the registry
embed_func = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")

# 2. Define the exact schema so LanceDB knows which field to auto-embed
class KnowledgeSchema(LanceModel):
    id: str
    text: str = embed_func.SourceField()
    vector: Vector(embed_func.ndims()) = embed_func.VectorField() # type: ignore
    metadata: str

class LongTermMemory:
    """Vector-backed long-term memory store with disk persistence using LanceDB."""

    def __init__(
        self,
        table_name: str = "agent_knowledge",
        persist_dir: str = "./lancedb_data",
    ):
        self.table_name = table_name
        self.persist_dir = persist_dir
        self.db = lancedb.connect(self.persist_dir)

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        if not documents:
            return

        # Generate IDs and format metadata safely
        if not ids:
            ids = [f"doc_{hashlib.md5(doc.encode('utf-8')).hexdigest()}" for doc in documents]
        
        if not metadatas:
            metadatas = [{} for _ in documents]

        # Notice: No manual vector math here! Just standard text dictionaries.
        data = []
        for i in range(len(documents)):
            data.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": json.dumps(metadatas[i])
            })

        try:
            table = self.db.open_table(self.table_name)
            table.add(data)
        except Exception:
            # Create the table explicitly using our automated schema
            self.db.create_table(self.table_name, schema=KnowledgeSchema, data=data)
            
        print(f"🧠 [LONG-TERM MEMORY]: Saved {len(documents)} document chunk(s) to '{self.persist_dir}'.")

    def delete(self, doc_id: str):
        """Deletes a document from the vector database by its ID."""
        try:
            table = self.db.open_table(self.table_name)
            table.delete(f"id = '{doc_id}'")
            print(f"🗑️ [LONG-TERM MEMORY]: Deleted document '{doc_id}' from LanceDB.")
        except Exception as e:
            print(f"❌ Failed to delete document '{doc_id}': {e}")
            raise e

    def update(self, doc_id: str, text: str, metadata: dict = None):
        """Updates a document by deleting the old one and re-embedding the new text."""
        self.delete(doc_id)
        
        self.add_documents(
            documents=[text],
            metadatas=[metadata] if metadata else None,
            ids=[doc_id]
        )
        print(f"🔄 [LONG-TERM MEMORY]: Successfully updated document '{doc_id}'.")

    def search(self, query: str, top_k: int = 4) -> str:
        try:
            table = self.db.open_table(self.table_name)
        except Exception:
            return "No relevant information found in long-term memory."
        
        results = table.search(query).limit(top_k).to_list()

        if not results:
            return "No relevant information found in long-term memory."

        docs = [f"ID: {res['id']} | Content: {res['text']}" for res in results]
        formatted_results = "\n---\n".join(docs)
        
        return f"Relevant Long-Term Memory Results:\n{formatted_results}"