"""
gRPC server entrypoint for EmbodiedMemory.
"""

from __future__ import annotations

import logging
from concurrent import futures
from typing import Any, Optional

import grpc

from powermem.core.memory import Memory

from ..embodied_memory import EmbodiedMemory
from ..schema import initialize_embodied_schema
from .servicer import EmbodiedMemoryServicer

logger = logging.getLogger(__name__)


def serve(
    memory: Memory,
    db_conn: Any,
    host: str = "0.0.0.0",
    port: int = 50051,
    enable_plugin: bool = False,
    max_workers: int = 10,
) -> grpc.Server:
    """Start an insecure gRPC server exposing EmbodiedMemory.

    Args:
        memory: PowerMem Memory instance
        db_conn: Database connection (SQLite or SeekDB)
        host: Bind host
        port: Bind port
        enable_plugin: Enable EmbodiedIntelligencePlugin
        max_workers: Thread pool size

    Returns:
        grpc.Server instance (call server.wait_for_termination() to block)
    """
    initialize_embodied_schema(db_conn)
    em = EmbodiedMemory(
        memory=memory,
        db_conn=db_conn,
        enable_plugin=enable_plugin,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    from powermem.embodied.proto import embodied_memory_pb2_grpc

    embodied_memory_pb2_grpc.add_EmbodiedMemoryServiceServicer_to_server(
        EmbodiedMemoryServicer(em), server
    )

    address = f"{host}:{port}"
    server.add_insecure_port(address)
    server.start()
    logger.info("EmbodiedMemory gRPC server started on %s", address)
    return server
