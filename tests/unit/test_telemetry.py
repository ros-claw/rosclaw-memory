"""
Tests for MemoryTelemetry — cache and query metrics
"""

import sqlite3
import time
from typing import Any, Dict, List, Optional

import pytest

from powermem.embodied import (
    EmbodiedMemory,
    MemoryAtom,
    MemoryTelemetry,
    Pose,
    Vec3,
    WorldObject,
)
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.telemetry import _Timer


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------

class MockStorageAdapter:
    def __init__(self):
        self._store: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1000

    def add_memory(self, payload: Dict[str, Any]) -> int:
        mid = self._next_id
        self._next_id += 1
        self._store[mid] = {
            "id": mid, "data": payload.get("content", ""),
            "content": payload.get("content", ""), "metadata": payload.get("metadata", {}),
            "user_id": "", "agent_id": "", "run_id": "", "created_at": "2024-01-01",
        }
        return mid

    def get_memory(self, mid: int) -> Optional[Dict[str, Any]]:
        return self._store.get(mid)

    def get_many_memories(self, mids):
        return [self._store.get(m) for m in mids]

    def delete_memory(self, mid: int, **kw) -> bool:
        return self._store.pop(mid, None) is not None

    def search_memories(self, **kw) -> List[Dict[str, Any]]:
        return [{"id": mid, "memory": item["data"], "score": 0.9, "metadata": {}}
                for mid, item in list(self._store.items())[:kw.get("limit", 30)]]


class MockMemory:
    def __init__(self):
        self.storage = MockStorageAdapter()
        self.agent_id = "test_agent"

    def search(self, *a, **kw):
        return {"results": self.storage.search_memories(**kw), "relations": []}

    def delete(self, mid): return self.storage.delete_memory(mid)
    def update(self, mid, content, **kw): return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def telemetry():
    return MemoryTelemetry(enabled=True)


@pytest.fixture
def embodied_memory_with_telemetry(telemetry):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    initialize_embodied_schema(conn)
    em = EmbodiedMemory(
        memory=MockMemory(), db_conn=conn, voxel_size=1.0,
        enable_plugin=False, telemetry=telemetry,
    )
    yield em
    conn.close()


# ========================================================================
# Unit tests: MemoryTelemetry
# ========================================================================

class TestTelemetryBasic:
    def test_init(self, telemetry):
        assert telemetry.enabled
        s = telemetry.snapshot()
        assert s["atom_cache_hits"] == 0
        assert s["atom_cache_hit_rate"] == 0.0

    def test_disabled(self):
        t = MemoryTelemetry(enabled=False)
        t.record_cache_hit("atom")
        t.record_cache_miss("atom")
        t.record_latency("spatial_query", 1.5)
        s = t.snapshot()
        assert s["atom_cache_hits"] == 0
        assert s["spatial_query"]["count"] == 0

    def test_reset(self, telemetry):
        telemetry.record_cache_hit("atom")
        telemetry.record_cache_hit("atom")
        telemetry.record_cache_miss("atom")
        telemetry.record_latency("spatial_query", 1.0)
        telemetry.reset()
        s = telemetry.snapshot()
        assert s["atom_cache_hits"] == 0
        assert s["atom_cache_misses"] == 0
        assert s["spatial_query"]["count"] == 0


class TestCacheCounters:
    def test_hit_rate(self, telemetry):
        telemetry.record_cache_hit("atom")
        telemetry.record_cache_hit("atom")
        telemetry.record_cache_miss("atom")
        s = telemetry.snapshot()
        assert s["atom_cache_hits"] == 2
        assert s["atom_cache_misses"] == 1
        assert abs(s["atom_cache_hit_rate"] - 2/3) < 0.01
        assert s["atom_cache_total"] == 3

    def test_all_caches(self, telemetry):
        for name in ["atom", "world_object", "scene_graph", "model", "collision"]:
            telemetry.record_cache_hit(name)
            telemetry.record_cache_miss(name)
        s = telemetry.snapshot()
        for name in ["atom", "world_object", "scene_graph", "model", "collision"]:
            assert s[f"{name}_cache_hits"] == 1
            assert s[f"{name}_cache_misses"] == 1
            assert s[f"{name}_cache_total"] == 2

    def test_unknown_cache_ignored(self, telemetry):
        # Should not raise
        telemetry.record_cache_hit("unknown_cache")
        telemetry.record_cache_miss("unknown_cache")


class TestLatencyCounters:
    def test_record_latency(self, telemetry):
        telemetry.record_latency("spatial_query", 5.0)
        telemetry.record_latency("spatial_query", 10.0)
        telemetry.record_latency("spatial_query", 15.0)
        s = telemetry.snapshot()
        assert s["spatial_query"]["count"] == 3
        assert abs(s["spatial_query"]["avg_ms"] - 10.0) < 0.01
        assert abs(s["spatial_query"]["min_ms"] - 5.0) < 0.01
        assert abs(s["spatial_query"]["max_ms"] - 15.0) < 0.01

    def test_all_operations(self, telemetry):
        for op in ["spatial_query", "temporal_query", "trajectory_query",
                   "atom_get", "atom_add", "scene_graph_build"]:
            telemetry.record_latency(op, 1.0)
        s = telemetry.snapshot()
        for op in ["spatial_query", "temporal_query", "trajectory_query",
                   "atom_get", "atom_add", "scene_graph_build"]:
            assert s[op]["count"] == 1

    def test_unknown_operation_ignored(self, telemetry):
        telemetry.record_latency("unknown_op", 1.0)
        # Should not raise


class TestTimer:
    def test_timer_records_latency(self, telemetry):
        with _Timer(telemetry, "spatial_query"):
            time.sleep(0.001)  # ~1ms
        s = telemetry.snapshot()
        assert s["spatial_query"]["count"] == 1
        assert s["spatial_query"]["total_ms"] > 0

    def test_timer_with_none_telemetry(self):
        # Should not raise when telemetry is None
        with _Timer(None, "spatial_query"):
            time.sleep(0.001)

    def test_timer_disabled_telemetry(self):
        t = MemoryTelemetry(enabled=False)
        with _Timer(t, "spatial_query"):
            time.sleep(0.001)
        assert t.snapshot()["spatial_query"]["count"] == 0

    def test_timer_exception_still_records(self, telemetry):
        try:
            with _Timer(telemetry, "spatial_query"):
                raise ValueError("test")
        except ValueError:
            pass
        # Timer.__exit__ returns False so exception propagates,
        # but since we catch it above, the record_latency was called
        # (because __exit__ runs before exception propagates)
        # Actually with our impl: __exit__ records THEN returns False
        s = telemetry.snapshot()
        assert s["spatial_query"]["count"] == 1


class TestPrometheus:
    def test_prometheus_format(self, telemetry):
        telemetry.record_cache_hit("atom")
        telemetry.record_cache_miss("atom")
        telemetry.record_latency("spatial_query", 5.0)
        text = telemetry.prometheus_metrics()
        assert "powermem_cache_hits_total" in text
        assert 'cache="atom"' in text
        assert "powermem_cache_misses_total" in text
        assert "powermem_query_duration_ms_count" in text
        assert 'op="spatial"' in text

    def test_prometheus_empty_when_disabled(self):
        t = MemoryTelemetry(enabled=False)
        text = t.prometheus_metrics()
        # Still returns valid (empty counters) output
        assert "powermem_cache_hits_total" in text


# ========================================================================
# Integration tests: EmbodiedMemory + MemoryTelemetry
# ========================================================================

class TestEmbodiedMemoryTelemetry:
    def test_telemetry_none_by_default(self):
        conn = sqlite3.connect(":memory:")
        initialize_embodied_schema(conn)
        em = EmbodiedMemory(memory=MockMemory(), db_conn=conn, enable_plugin=False)
        assert em._telemetry is None
        assert em.get_telemetry() is None
        conn.close()

    def test_get_atom_cache_hit_miss(self, embodied_memory_with_telemetry, telemetry):
        em = embodied_memory_with_telemetry
        atom = MemoryAtom(content="telemetry_test")
        mid = em.add_atom(atom)

        # add_atom puts atom in cache; invalidate to force a miss on first read
        em._invalidate_atom_cache(mid)

        # First read → cache miss
        result = em.get_atom(mid)
        assert result is not None
        s = telemetry.snapshot()
        assert s["atom_cache_misses"] == 1

        # Second read → cache hit
        result2 = em.get_atom(mid)
        assert result2 is not None
        s = telemetry.snapshot()
        assert s["atom_cache_hits"] == 1
        assert s["atom_cache_misses"] == 1

    def test_get_world_object_cache_hit_miss(self, embodied_memory_with_telemetry, telemetry):
        em = embodied_memory_with_telemetry
        obj = WorldObject(
            obj_id="tel_obj", pose=Pose(position=Vec3(0, 0, 0)),
            scene_id="test",
        )
        em.add_world_object(obj)

        # add_world_object puts object in cache; invalidate to force a miss
        em.world_object_store._invalidate_object_cache("tel_obj")

        # First read → miss
        em.get_world_object("tel_obj")
        s = telemetry.snapshot()
        assert s["world_object_cache_misses"] >= 1

        # Second read → hit
        em.get_world_object("tel_obj")
        s = telemetry.snapshot()
        assert s["world_object_cache_hits"] >= 1

    def test_search_near_records_spatial_latency(self, embodied_memory_with_telemetry, telemetry):
        em = embodied_memory_with_telemetry
        for i in range(5):
            atom = MemoryAtom(content=f"near_{i}", spatial=Vec3(i, 0, 0))
            em.add_atom(atom)

        em.search_near(center=Vec3(0, 0, 0), radius=5.0)
        s = telemetry.snapshot()
        assert s["spatial_query"]["count"] == 1
        assert s["spatial_query"]["total_ms"] > 0

    def test_search_temporal_records_latency(self, embodied_memory_with_telemetry, telemetry):
        from powermem.embodied import TemporalInterval
        em = embodied_memory_with_telemetry
        em.search_temporal(interval=TemporalInterval(start_sec=0.0, end_sec=1.0))
        s = telemetry.snapshot()
        assert s["temporal_query"]["count"] == 1

    def test_get_scene_graph_cache(self, embodied_memory_with_telemetry, telemetry):
        em = embodied_memory_with_telemetry
        obj = WorldObject(
            obj_id="sg_obj", pose=Pose(position=Vec3(0, 0, 0)), scene_id="sg_scene",
        )
        em.add_world_object(obj)

        # First call → miss + build
        sg1 = em.get_scene_graph("sg_scene")
        s = telemetry.snapshot()
        assert s["scene_graph_cache_misses"] == 1
        assert s["scene_graph_build"]["count"] == 1

        # Second call → hit
        sg2 = em.get_scene_graph("sg_scene")
        s = telemetry.snapshot()
        assert s["scene_graph_cache_hits"] == 1

    def test_get_telemetry_snapshot(self, embodied_memory_with_telemetry, telemetry):
        em = embodied_memory_with_telemetry
        atom = MemoryAtom(content="snap")
        em.add_atom(atom)
        s = em.get_telemetry()
        assert s is not None
        assert "atom_cache_hits" in s
        assert "spatial_query" in s

    def test_prometheus_metrics_export(self, embodied_memory_with_telemetry, telemetry):
        em = embodied_memory_with_telemetry
        atom = MemoryAtom(content="prom")
        em.add_atom(atom)
        text = em.prometheus_metrics()
        assert "powermem_cache" in text

    def test_telemetry_no_overhead_when_none(self):
        """Verify no telemetry overhead when telemetry=None."""
        conn = sqlite3.connect(":memory:")
        initialize_embodied_schema(conn)
        em = EmbodiedMemory(memory=MockMemory(), db_conn=conn, enable_plugin=False)
        # Should work without telemetry
        atom = MemoryAtom(content="no_tel", spatial=Vec3(0, 0, 0))
        em.add_atom(atom)
        em.search_near(center=Vec3(0, 0, 0), radius=1.0)
        conn.close()
