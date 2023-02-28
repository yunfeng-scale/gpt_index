"""Postgres vector store index."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine

from gpt_index.indices.query.embedding_utils import get_top_k_embeddings
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)


class PostgresVectorStore(VectorStore):
    """Postgres Vector Store.

    In this vector store, embeddings are stored within Postgres.

    Args:
        Postgres_vector_store_data_dict (Optional[dict]): data dict
            containing the embeddings and doc_ids. See PostgresVectorStoreData
            for more details.
    """

    stores_text: bool = True
    indexed_results: List[NodeEmbeddingResult] = []

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        self.engine = create_engine("postgresql://postgres:@localhost:5432/llm_index")
        self.conn = self.engine.connect()

    @property
    def client(self) -> None:
        """Get client."""
        return None

    @property
    def config_dict(self) -> dict:
        """Get config dict."""
        return {}

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding_results to index."""
        for result in embedding_results:
            embedding = "{" + ",".join(str(x) for x in result.embedding) + "}"
            self.conn.execute(f"INSERT INTO nodes (id, text, embedding, doc_id) values (%s, %s, %s, %s)", (result.id, result.node.get_text(), embedding, result.doc_id))
        self.indexed_results.extend(embedding_results)
        return [result.id for result in embedding_results]

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        # text_ids_to_delete = set()
        # for text_id, doc_id_ in self._data.text_id_to_doc_id.items():
        #     if doc_id == doc_id_:
        #         text_ids_to_delete.add(text_id)

        # for text_id in text_ids_to_delete:
        #     del self._data.embedding_dict[text_id]
        #     del self._data.text_id_to_doc_id[text_id]
        raise NotImplemented

    def query(
        self, query_embedding: List[float], similarity_top_k: int
    ) -> VectorStoreQueryResult:
        raise NotImplemented
    
        """Get nodes for response."""
        # TODO: consolidate with get_query_text_embedding_similarities
        items = self._data.embedding_dict.items()
        node_ids = [t[0] for t in items]
        embeddings = [t[1] for t in items]

        top_similarities, top_ids = get_top_k_embeddings(
            query_embedding,
            embeddings,
            similarity_top_k=similarity_top_k,
            embedding_ids=node_ids,
        )

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)
