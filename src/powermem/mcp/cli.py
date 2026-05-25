"""
RosClaw-Memory MCP Server CLI

Usage:
    powermem-mcp-server --db-path ./embodied.db
    python -m powermem.mcp.cli --db-path ./embodied.db
"""

from __future__ import annotations

import logging

import click

from .server import create_mcp_server

logger = logging.getLogger(__name__)


@click.command()
@click.option("--db-path", default="./embodied.db", help="Path to embodied SQLite database")
@click.option("--log-level", default="INFO", help="Logging level")
def mcp_server(db_path: str, log_level: str):
    """Start the RosClaw-Memory MCP Server"""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    mcp = create_mcp_server(db_path)
    logger.info("RosClaw-Memory MCP server starting (db=%s)", db_path)
    mcp.run()


if __name__ == "__main__":
    mcp_server()
