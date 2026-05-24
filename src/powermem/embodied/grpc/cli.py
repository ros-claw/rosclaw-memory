"""
CLI command for starting the EmbodiedMemory gRPC server.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional

import click

logger = logging.getLogger(__name__)


def _open_db_conn(db_path: str, seekdb_path: Optional[str] = None) -> Any:
    """Open a database connection for embodied schema.

    Tries SeekDB first if available, falls back to SQLite.
    """
    try:
        import pyseekdb
        if seekdb_path:
            conn = pyseekdb.connect(seekdb_path)
            logger.info("Using SeekDB at %s", seekdb_path)
            return conn
    except ImportError:
        pass

    conn = sqlite3.connect(db_path, check_same_thread=False)
    logger.info("Using SQLite at %s", db_path)
    return conn


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=50051, type=int, help="Port to bind to")
@click.option("--db-path", default="./embodied.db", help="SQLite database path")
@click.option("--seekdb-path", default=None, help="SeekDB path (overrides --db-path if available)")
@click.option("--pmem-config", default=None, help="Path to PowerMem config JSON")
@click.option("--enable-plugin", is_flag=True, default=False, help="Enable EmbodiedIntelligencePlugin")
@click.option("--max-workers", default=10, type=int, help="gRPC thread pool size")
@click.option("--log-level", default="INFO", help="Log level")
def embodied_server(
    host: str,
    port: int,
    db_path: str,
    seekdb_path: Optional[str],
    pmem_config: Optional[str],
    enable_plugin: bool,
    max_workers: int,
    log_level: str,
):
    """Start the EmbodiedMemory gRPC server.

    Example:
        powermem-embodied-server --host 0.0.0.0 --port 50051 --db-path ./embodied.db
    """
    import sys

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Initialize PowerMem Memory
    from powermem import Memory, auto_config

    if pmem_config:
        import json
        with open(pmem_config) as f:
            cfg = json.load(f)
        memory = Memory(config=cfg)
    else:
        memory = Memory(config=auto_config())

    # Open embodied DB connection
    db_conn = _open_db_conn(db_path, seekdb_path)

    # Start gRPC server
    from .server import serve

    server = serve(
        memory=memory,
        db_conn=db_conn,
        host=host,
        port=port,
        enable_plugin=enable_plugin,
        max_workers=max_workers,
    )

    address = f"{host}:{port}"
    logger.info("EmbodiedMemory gRPC server running on %s", address)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5.0)


if __name__ == "__main__":
    embodied_server()
