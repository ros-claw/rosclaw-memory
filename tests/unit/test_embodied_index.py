"""
Unit tests for ROSClaw-Memory spatial and temporal indexing.

SpatialIndex / TemporalIndex tests use an in-memory SQLite DB
for portability (no SeekDB required for unit testing).
"""

import sqlite3

import pytest

from powermem.embodied.spatial_index import SpatialIndex, VoxelHash
from powermem.embodied.temporal_index import TemporalIndex
from powermem.embodied.types import IntervalRelation, TemporalInterval, Vec3
from powermem.embodied.schema import get_dialect_ddl


# ---------------------------------------------------------------------------
# VoxelHash
# ---------------------------------------------------------------------------

class TestVoxelHash:
    def test_insert_and_query(self):
        vh = VoxelHash(voxel_size=1.0)
        vh.insert(1, Vec3(0.5, 0.5, 0.5))
        vh.insert(2, Vec3(1.5, 0.5, 0.5))
        vh.insert(3, Vec3(5.0, 5.0, 5.0))

        ids = vh.query_near(Vec3(0.0, 0.0, 0.0), radius=1.5, frame_id="world")
        assert 1 in ids
        assert 2 in ids
        assert 3 not in ids

    def test_remove(self):
        vh = VoxelHash(voxel_size=1.0)
        vh.insert(1, Vec3(0.5, 0.5, 0.5))
        assert vh.remove(1) is True
        ids = vh.query_near(Vec3(0.0, 0.0, 0.0), radius=2.0)
        assert 1 not in ids
        assert vh.remove(1) is False

    def test_exact_query_with_positions(self):
        vh = VoxelHash(voxel_size=1.0)
        vh.insert(1, Vec3(0.0, 0.0, 0.0))
        vh.insert(2, Vec3(3.0, 4.0, 0.0))

        positions = {1: Vec3(0.0, 0.0, 0.0), 2: Vec3(3.0, 4.0, 0.0)}
        results = vh.query_exact(Vec3(0.0, 0.0, 0.0), radius=5.0, id_to_position=positions)
        ids = [mid for mid, _ in results]
        assert ids == [1, 2]

    def test_multi_frame_isolation(self):
        vh = VoxelHash(voxel_size=1.0)
        vh.insert(1, Vec3(0.5, 0.5, 0.5), frame_id="world")
        vh.insert(2, Vec3(0.5, 0.5, 0.5), frame_id="camera")

        world_ids = vh.query_near(Vec3(0.0, 0.0, 0.0), radius=2.0, frame_id="world")
        camera_ids = vh.query_near(Vec3(0.0, 0.0, 0.0), radius=2.0, frame_id="camera")

        assert 1 in world_ids
        assert 2 not in world_ids
        assert 2 in camera_ids
        assert 1 not in camera_ids

    def test_stats(self):
        vh = VoxelHash(voxel_size=0.5)
        for i in range(100):
            vh.insert(i, Vec3(i * 0.1, 0.0, 0.0))
        stats = vh.stats()
        assert stats["total_ids"] == 100
        assert stats["avg_load"] > 0


# ---------------------------------------------------------------------------
# SpatialIndex (with SQLite)
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_conn():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def seeded_spatial_index(sqlite_conn):
    """创建已填充数据的 SpatialIndex"""
    ddl_list, idx_list = get_dialect_ddl("sqlite")
    for ddl in ddl_list:
        sqlite_conn.execute(ddl)
    for idx in idx_list:
        try:
            sqlite_conn.execute(idx)
        except Exception:
            pass

    # Insert dummy embodied records
    cursor = sqlite_conn.cursor()
    for i in range(1, 6):
        cursor.execute(
            """
            INSERT INTO embodied_memories (
                memory_id, spatial_x, spatial_y, spatial_z, spatial_voxel_key, spatial_frame_id,
                temporal_start, temporal_end, temporal_frame_id, modality, action_type, prediction_error, embodied_meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (i, float(i), 0.0, 0.0, f"{i}:0:0:world", "world", None, None, "wall_clock", "rgb", "observe", 0.0, "{}"),
        )
    sqlite_conn.commit()

    si = SpatialIndex(sqlite_conn, voxel_size=1.0)
    si.rebuild_from_db()
    return si


class TestSpatialIndex:
    def test_rebuild_from_db(self, seeded_spatial_index):
        si = seeded_spatial_index
        assert si._loaded is True
        assert len(si.get_all_ids()) == 5

    def test_query_radius(self, seeded_spatial_index):
        si = seeded_spatial_index
        results = si.query_radius(Vec3(2.5, 0.0, 0.0), radius=2.0)
        ids = [mid for mid, _ in results]
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids
        assert 5 not in ids

    def test_query_nearest(self, seeded_spatial_index):
        si = seeded_spatial_index
        results = si.query_nearest(Vec3(0.0, 0.0, 0.0), k=3)
        assert len(results) <= 3
        assert results[0][0] == 1  # closest

    def test_add_and_remove(self, sqlite_conn):
        ddl_list, _ = get_dialect_ddl("sqlite")
        for ddl in ddl_list:
            sqlite_conn.execute(ddl)
        si = SpatialIndex(sqlite_conn, voxel_size=1.0)
        si.add(100, Vec3(10.0, 10.0, 10.0))
        assert 100 in si.get_all_ids()
        si.remove(100)
        assert 100 not in si.get_all_ids()


# ---------------------------------------------------------------------------
# TemporalIndex (with SQLite)
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_temporal_index(sqlite_conn):
    ddl_list, _ = get_dialect_ddl("sqlite")
    for ddl in ddl_list:
        sqlite_conn.execute(ddl)

    cursor = sqlite_conn.cursor()
    intervals = [
        (1, 0.0, 5.0),
        (2, 3.0, 8.0),
        (3, 10.0, 15.0),
        (4, 5.0, 5.0),   # point interval
    ]
    for mid, start, end in intervals:
        cursor.execute(
            """
            INSERT INTO embodied_memories (
                memory_id, spatial_x, spatial_y, spatial_z, spatial_voxel_key,
                temporal_start, temporal_end, temporal_frame_id, modality, action_type, prediction_error, embodied_meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (mid, None, None, None, None, start, end, "wall_clock", "rgb", "observe", 0.0, "{}"),
        )
    sqlite_conn.commit()
    return TemporalIndex(sqlite_conn)


class TestTemporalIndex:
    def test_query_overlapping(self, seeded_temporal_index):
        ti = seeded_temporal_index
        results = ti.query_overlapping(TemporalInterval(2.0, 6.0))
        ids = [mid for mid, _ in results]
        assert 1 in ids  # [0,5] overlaps [2,6]
        assert 2 in ids  # [3,8] overlaps [2,6]
        assert 3 not in ids

    def test_query_before(self, seeded_temporal_index):
        ti = seeded_temporal_index
        results = ti.query_before(6.0)
        ids = [mid for mid, _ in results]
        assert 1 in ids  # ends at 5
        assert 2 not in ids  # ends at 8

    def test_query_after(self, seeded_temporal_index):
        ti = seeded_temporal_index
        results = ti.query_after(6.0)
        ids = [mid for mid, _ in results]
        assert 3 in ids  # starts at 10
        assert 1 not in ids

    def test_query_contains_point(self, seeded_temporal_index):
        ti = seeded_temporal_index
        results = ti.query_contains_point(4.0)
        ids = [mid for mid, _ in results]
        assert 1 in ids
        assert 2 in ids
        assert 3 not in ids

    def test_query_allen_relations(self, seeded_temporal_index):
        ti = seeded_temporal_index
        # [0,5] meets [5,5] (id=4) at point 5
        results = ti.query(TemporalInterval(5.0, 5.0), IntervalRelation.MEETS)
        ids = [mid for mid, _ in results]
        assert 1 in ids  # [0,5] ends at 5

        # [0,5] equals [0,5]
        results = ti.query(TemporalInterval(0.0, 5.0), IntervalRelation.EQUALS)
        ids = [mid for mid, _ in results]
        assert 1 in ids
