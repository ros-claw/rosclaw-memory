"""
Tests for schema versioning and automatic migration.
"""

from __future__ import annotations

import sqlite3

import pytest

from powermem.embodied.schema import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    initialize_embodied_schema,
    _get_current_schema_version,
    _table_exists,
    _column_exists,
)


class TestSchemaVersionDetection:
    def test_fresh_database_starts_at_version_0(self):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        assert _get_current_schema_version(cursor, "sqlite") == 0
        conn.close()

    def test_schema_version_table_created(self):
        conn = sqlite3.connect(":memory:")
        initialize_embodied_schema(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM embodied_schema_version")
        versions = [r[0] for r in cursor.fetchall()]
        assert CURRENT_SCHEMA_VERSION in versions
        conn.close()

    def test_current_version_detected_after_init(self):
        conn = sqlite3.connect(":memory:")
        initialize_embodied_schema(conn)
        cursor = conn.cursor()
        version = _get_current_schema_version(cursor, "sqlite")
        assert version == CURRENT_SCHEMA_VERSION
        conn.close()


class TestSchemaMigration:
    def test_migration_from_v1_to_v2(self):
        """Simulate an old v1 database and verify migration to v2."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Create v1 schema manually (without occlusion columns)
        cursor.execute("""
            CREATE TABLE embodied_world_objects (
                obj_id TEXT PRIMARY KEY,
                obj_type TEXT,
                name TEXT,
                pos_x REAL,
                pos_y REAL,
                pos_z REAL,
                orient_w REAL,
                orient_x REAL,
                orient_y REAL,
                orient_z REAL,
                size_json TEXT,
                color_json TEXT,
                mesh_path TEXT,
                physics_props_json TEXT,
                semantic_tags_json TEXT,
                scene_id TEXT,
                parent_obj_id TEXT,
                state TEXT DEFAULT 'present',
                memory_id INTEGER
            )
        """)
        # Insert a v1 object
        cursor.execute(
            "INSERT INTO embodied_world_objects (obj_id, obj_type, scene_id, state) VALUES (?, ?, ?, ?)",
            ("v1_cup", "cylinder", "kitchen", "present"),
        )
        conn.commit()

        # Verify v1 state
        assert _column_exists(cursor, "embodied_world_objects", "occlusion_status", "sqlite") is False
        assert _get_current_schema_version(cursor, "sqlite") == 1

        # Initialize schema (should auto-migrate v1 -> v2)
        initialize_embodied_schema(conn)

        # Verify v2 state
        cursor = conn.cursor()
        assert _column_exists(cursor, "embodied_world_objects", "occlusion_status", "sqlite") is True
        assert _column_exists(cursor, "embodied_world_objects", "confidence", "sqlite") is True
        assert _column_exists(cursor, "embodied_world_objects", "last_seen_sec", "sqlite") is True

        # Verify old data preserved
        cursor.execute("SELECT obj_id, occlusion_status, confidence FROM embodied_world_objects WHERE obj_id = ?", ("v1_cup",))
        row = cursor.fetchone()
        assert row[0] == "v1_cup"
        assert row[1] == "visible"  # default value
        assert row[2] == 1.0        # default value

        # Verify version recorded
        version = _get_current_schema_version(cursor, "sqlite")
        assert version == CURRENT_SCHEMA_VERSION
        conn.close()

    def test_idempotent_migration(self):
        """Running initialize twice should not fail or duplicate data."""
        conn = sqlite3.connect(":memory:")
        initialize_embodied_schema(conn)
        initialize_embodied_schema(conn)  # second call

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embodied_schema_version")
        count = cursor.fetchone()[0]
        # Should have exactly one record per version
        assert count == len(MIGRATIONS)

        version = _get_current_schema_version(cursor, "sqlite")
        assert version == CURRENT_SCHEMA_VERSION
        conn.close()


class TestSchemaHelpers:
    def test_table_exists(self):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER)")
        assert _table_exists(cursor, "test_table", "sqlite") is True
        assert _table_exists(cursor, "nonexistent", "sqlite") is False
        conn.close()

    def test_column_exists(self):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        assert _column_exists(cursor, "test_table", "id", "sqlite") is True
        assert _column_exists(cursor, "test_table", "name", "sqlite") is True
        assert _column_exists(cursor, "test_table", "missing", "sqlite") is False
        conn.close()
