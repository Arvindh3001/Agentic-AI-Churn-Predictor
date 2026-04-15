"""
SQL Query Tool
===============
LangChain-compatible tool that executes read-only SQL queries against
PostgreSQL to fetch customer feature data and interaction history.

Safety: only SELECT statements are permitted — writes go through dedicated
repository methods, never through this tool.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, field_validator
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings

logger = structlog.get_logger(__name__)

_engine = create_engine(
    settings.db.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"connect_timeout": 10},
)


class SQLQueryInput(BaseModel):
    query: str
    params: dict[str, Any] = {}

    @field_validator("query")
    @classmethod
    def must_be_select(cls, v: str) -> str:
        normalised = v.strip().upper()
        if not normalised.startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted through this tool.")
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER"]
        if any(kw in normalised for kw in forbidden):
            raise ValueError(f"Forbidden SQL keyword detected in query.")
        return v


@tool("sql_query_tool", args_schema=SQLQueryInput)
def sql_query_tool(query: str, params: dict[str, Any] = {}) -> list[dict[str, Any]]:
    """
    Execute a read-only SQL SELECT query against the churn database.
    Returns a list of row dicts. Use :param_name placeholders for parameters.

    Example:
        sql_query_tool(
            query="SELECT * FROM customers WHERE customer_id = :cid",
            params={"cid": "C-4821"}
        )
    """
    try:
        with _engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = [dict(row._mapping) for row in result]
        logger.debug("SQL query executed", rows_returned=len(rows))
        return rows
    except SQLAlchemyError as exc:
        logger.error("SQL query failed", error=str(exc), query=query[:100])
        return [{"error": str(exc)}]


def fetch_customer_features(customer_id: str) -> dict[str, Any] | None:
    """
    Convenience function: fetch full feature row for a single customer.
    Returns None if customer not found.
    """
    rows = sql_query_tool.invoke({
        "query": "SELECT * FROM customers WHERE customer_id = :cid LIMIT 1",
        "params": {"cid": customer_id},
    })
    if not rows or "error" in rows[0]:
        return None
    return rows[0]


def fetch_intervention_history(customer_id: str) -> list[dict[str, Any]]:
    """Fetch all past interventions for a customer from the interventions table."""
    return sql_query_tool.invoke({
        "query": """
            SELECT action, outcome, cost_usd, approved_by, created_at
            FROM interventions
            WHERE customer_id = :cid
            ORDER BY created_at DESC
            LIMIT 20
        """,
        "params": {"cid": customer_id},
    })
