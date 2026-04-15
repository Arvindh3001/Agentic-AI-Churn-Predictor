"""
ChromaDB Vector Store — Agent Long-Term Memory
================================================
Provides semantic search over customer interaction history,
past explanations, and similar-customer retrieval.

Collections:
  - customer_interactions : past HITL decisions + outcomes per customer
  - churn_explanations    : narrative explanations (searchable by content)
  - similar_cases         : embeddings of high-risk customer profiles

Usage:
    from memory.vector_store import VectorStore

    vs = VectorStore()
    vs.upsert_customer_interaction(customer_id, interaction_dict)
    similar = vs.find_similar_customers(feature_vector, top_k=5)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import chromadb
import structlog
from chromadb.config import Settings

from config.settings import settings

logger = structlog.get_logger(__name__)

COLLECTION_INTERACTIONS = "customer_interactions"
COLLECTION_EXPLANATIONS = "churn_explanations"
COLLECTION_PROFILES = "customer_profiles"


class VectorStore:
    """
    ChromaDB-backed semantic memory for the agent system.

    Connects to the running ChromaDB container (from docker-compose).
    Falls back to an in-memory client when the server is unreachable —
    useful for local dev without Docker.
    """

    def __init__(self) -> None:
        self._client = self._connect()
        self._interactions = self._get_or_create(COLLECTION_INTERACTIONS)
        self._explanations = self._get_or_create(COLLECTION_EXPLANATIONS)
        self._profiles = self._get_or_create(COLLECTION_PROFILES)

    def _connect(self) -> chromadb.ClientAPI:
        try:
            client = chromadb.HttpClient(
                host=settings.chromadb.chromadb_host,
                port=settings.chromadb.chromadb_port,
                settings=Settings(anonymized_telemetry=False),
            )
            client.heartbeat()
            logger.info(
                "ChromaDB connected",
                host=settings.chromadb.chromadb_host,
                port=settings.chromadb.chromadb_port,
            )
            return client
        except Exception as exc:
            logger.warning(
                "ChromaDB server unreachable — using in-memory client",
                error=str(exc),
            )
            return chromadb.Client()

    def _get_or_create(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # Customer Interactions
    # ------------------------------------------------------------------ #

    def upsert_customer_interaction(
        self,
        customer_id: str,
        interaction: dict[str, Any],
    ) -> None:
        """
        Store or update a customer interaction record.
        The document text is a summary used for semantic search.
        """
        doc_text = _interaction_to_text(interaction)
        metadata = {
            "customer_id": customer_id,
            "timestamp": interaction.get("timestamp", datetime.utcnow().isoformat()),
            "churn_prob": float(interaction.get("churn_prob", 0.0)),
            "risk_tier": str(interaction.get("risk_tier", "UNKNOWN")),
            "action_taken": str(interaction.get("action_taken", "none")),
            "outcome": str(interaction.get("outcome", "pending")),
        }
        doc_id = f"{customer_id}_{interaction.get('run_id', 'latest')}"

        self._interactions.upsert(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[metadata],
        )
        logger.debug("Customer interaction stored", customer_id=customer_id, doc_id=doc_id)

    def get_customer_history(
        self,
        customer_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve past interactions for a specific customer."""
        results = self._interactions.query(
            query_texts=[f"customer {customer_id} churn history"],
            n_results=limit,
            where={"customer_id": customer_id},
        )
        return _unpack_results(results)

    # ------------------------------------------------------------------ #
    # Similar Customer Retrieval
    # ------------------------------------------------------------------ #

    def upsert_customer_profile(
        self,
        customer_id: str,
        feature_summary: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store a customer's risk profile for similarity search."""
        self._profiles.upsert(
            ids=[customer_id],
            documents=[feature_summary],
            metadatas=[{k: str(v) for k, v in metadata.items()}],
        )

    def find_similar_customers(
        self,
        query_text: str,
        top_k: int = 5,
        risk_tier: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find customers with similar profiles using semantic search.

        Args:
            query_text: Natural language description of the customer.
            top_k: Number of similar customers to return.
            risk_tier: Optional filter — only return customers of this tier.

        Returns:
            List of similar customer dicts with similarity scores.
        """
        where = {"risk_tier": risk_tier} if risk_tier else None
        results = self._profiles.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )
        return _unpack_results(results)

    # ------------------------------------------------------------------ #
    # Explanation Memory
    # ------------------------------------------------------------------ #

    def store_explanation(
        self,
        customer_id: str,
        run_id: str,
        narrative: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store a generated narrative explanation for future retrieval."""
        doc_id = f"exp_{customer_id}_{run_id}"
        self._explanations.upsert(
            ids=[doc_id],
            documents=[narrative],
            metadatas=[{
                "customer_id": customer_id,
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                **{k: str(v) for k, v in metadata.items()},
            }],
        )
        logger.debug("Explanation stored", doc_id=doc_id)

    def search_explanations(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Semantic search over past explanations — used by agents for context."""
        results = self._explanations.query(
            query_texts=[query],
            n_results=top_k,
        )
        return _unpack_results(results)

    def get_collection_stats(self) -> dict[str, int]:
        return {
            "interactions": self._interactions.count(),
            "explanations": self._explanations.count(),
            "profiles": self._profiles.count(),
        }


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _interaction_to_text(interaction: dict[str, Any]) -> str:
    """Convert interaction dict to a searchable text document."""
    parts = [
        f"Customer {interaction.get('customer_id', 'unknown')}",
        f"churn probability {interaction.get('churn_prob', 0):.0%}",
        f"risk tier {interaction.get('risk_tier', 'unknown')}",
        f"action: {interaction.get('action_taken', 'none')}",
        f"outcome: {interaction.get('outcome', 'pending')}",
    ]
    top_factors = interaction.get("top_risk_factors", [])
    if top_factors:
        factor_text = ", ".join(
            f["feature"] for f in top_factors[:3] if "feature" in f
        )
        parts.append(f"top drivers: {factor_text}")
    return ". ".join(parts)


def _unpack_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ChromaDB query results into a list of dicts."""
    output: list[dict[str, Any]] = []
    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for i, doc_id in enumerate(ids):
        output.append({
            "id": doc_id,
            "document": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
            "similarity": round(1 - distances[i], 4) if i < len(distances) else 0.0,
        })
    return output
