import os
from typing import List, Dict, Any, Optional
import chromadb


class LongTermMemory:
    """Vector-backed long-term memory store with disk persistence using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "agent_knowledge",
        persist_dir: str = "./chroma_db",
    ):
        """Initializes ChromaDB with disk persistence.

        All embeddings and metadata are saved to disk in `persist_dir` and restored across runs.
        """
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """Embeds and indexes text chunks into the vector store."""
        if not ids:
            ids = [f"doc_{i}_{abs(hash(doc))}" for i, doc in enumerate(documents)]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"🧠 [LONG-TERM MEMORY]: Saved {len(documents)} document chunk(s) to '{self.persist_dir}'.")

    def search(self, query: str, top_k: int = 2) -> str:
        """Performs vector similarity search and returns matching context chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant information found in long-term memory."

        formatted_results = "\n---\n".join(docs)
        return f"Relevant Long-Term Memory Results:\n{formatted_results}"