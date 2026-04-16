"""
Customer WebSocket Broadcaster
================================
Singleton connection manager that pushes real-time customer-update events
to every connected dashboard tab.

WebSocket endpoint:  /ws/customers
Message shape sent to clients:
    {"type": "customer_updated", "customer": { ...safe customer fields... }}

Usage (from PATCH endpoint):
    from app.websockets.customer_broadcast import broadcaster
    await broadcaster.broadcast({"type": "customer_updated", "customer": record})
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["websocket"])


class CustomerBroadcaster:
    """Thread-safe (asyncio) manager for all active /ws/customers connections."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("Customer WS connected", total=len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("Customer WS disconnected", total=len(self._connections))

    async def broadcast(self, message: dict) -> None:
        """Send *message* to every live connection; prune dead ones."""
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# Module-level singleton — shared across the whole process
broadcaster = CustomerBroadcaster()


@router.websocket("/ws/customers")
async def customer_updates_stream(websocket: WebSocket) -> None:
    """
    Persistent WebSocket for real-time customer data updates.

    Connect once from the dashboard; the backend pushes a message any time
    a customer is updated via PATCH /api/v1/customers/{id}.

    No authentication token is checked here (same pattern as /ws/agent).
    """
    await broadcaster.connect(websocket)
    try:
        # Keep the connection alive; the client only listens (no client→server messages needed).
        while True:
            await websocket.receive_text()  # detects disconnects / pings
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Customer WS error", error=str(exc))
    finally:
        broadcaster.disconnect(websocket)
