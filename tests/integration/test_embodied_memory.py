"""
Integration tests for EmbodiedMemory

Uses SQLite in-memory DB + mock PowerMem Memory.
No external services (LLM/vector DB) required.
"""

import sqlite3
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from powermem.embodied import (
    EmbodiedMemory,
    MemoryAtom,
    Pose,
    SensorFrame,
    Vec3,
    TemporalInterval,
    IntervalRelation,
    Modality,
    UncertaintyEstimate,
    AffectiveTag,
    PhysicalInvariant,
    MemoryAction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockStorageAdapter:
    """Mock PowerMem StorageAdapter"""

    def __init__(self):
        self._store: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1000

    def add_memory(self, payload: Dict[str, Any]) -> int:
        mid = self._next_id
        self._next_id += 1
        self._store[mid] = {
            "id": mid,
            "data": payload.get("content", ""),
            "content": payload.get("content", ""),
            "metadata": payload.get("metadata", {}),
            "user_id": payload.get("user_id", ""),
            "agent_id": payload.get("agent_id", ""),
            "run_id": payload.get("run_id", ""),
            "created_at": "2024-01-01T00:00:00",
        }
        return mid

    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        return self._store.get(memory_id)

    def delete_memory(self, memory_id: int, user_id=None, agent_id=None) -> bool:
        return self._store.pop(memory_id, None) is not None

    def search_memories(
        self,
        query_embedding=None,
        user_id=None,
        agent_id=None,
        run_id=None,
        filters=None,
        limit=30,
        query=None,
        threshold=None,
    ) -> List[Dict[str, Any]]:
        """Mock semantic search: return all memories with dummy scores"""
        results = []
        for mid, item in list(self._store.items())[:limit]:
            results.append({
                "id": mid,
                "memory": item["data"],
                "score": 0.9,
                "metadata": item.get("metadata", {}),
            })
        return results


class MockMemory:
    """Mock PowerMem Memory"""

    def __init__(self):
        self.storage = MockStorageAdapter()
        self.agent_id = "test_agent"
        self._deleted_ids: List[int] = []

    def search(self, query, user_id=None, agent_id=None, run_id=None, filters=None, limit=30, threshold=None):
        results = self.storage.search_memories(
            query_embedding=None, user_id=user_id, agent_id=agent_id, run_id=run_id,
            filters=filters, limit=limit, query=query, threshold=threshold,
        )
        return {"results": results, "relations": []}

    def delete(self, memory_id: int) -> bool:
        ok = self.storage.delete_memory(memory_id)
        if ok:
            self._deleted_ids.append(memory_id)
        return ok


@pytest.fixture
def sqlite_conn():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def embodied_memory(sqlite_conn):
    mock_mem = MockMemory()
    em = EmbodiedMemory(
        memory=mock_mem,
        db_conn=sqlite_conn,
        voxel_size=1.0,
        enable_plugin=False,  # disable plugin to avoid side effects in tests
    )
    return em


# ---------------------------------------------------------------------------
# Add / Get
# ---------------------------------------------------------------------------

class TestAddGet:
    def test_add_atom_simple(self, embodied_memory):
        atom = MemoryAtom(content="hello world")
        mid = embodied_memory.add_atom(atom)
        assert mid >= 1000

    def test_add_atom_with_spatial(self, embodied_memory):
        atom = MemoryAtom(
            content="object at corner",
            spatial=Vec3(1.5, 2.5, 0.0),
            spatial_frame_id="world",
        )
        mid = embodied_memory.add_atom(atom)
        assert mid >= 1000

        # Verify spatial index
        hits = embodied_memory.spatial_index.query_radius(Vec3(1.5, 2.5, 0.0), radius=0.5)
        assert any(h[0] == mid for h in hits)

    def test_add_atom_with_temporal(self, embodied_memory):
        atom = MemoryAtom(
            content="event during interval",
            temporal=TemporalInterval(10.0, 20.0),
        )
        mid = embodied_memory.add_atom(atom)

        # Verify temporal index
        hits = embodied_memory.temporal_index.query_overlapping(TemporalInterval(15.0, 25.0))
        assert any(h[0] == mid for h in hits)

    def test_add_atom_full(self, embodied_memory):
        atom = MemoryAtom(
            content="full embodied memory",
            spatial=Vec3(1.0, 2.0, 3.0),
            temporal=TemporalInterval(0.0, 5.0),
            perceptual=None,
            physical=PhysicalInvariant(entity_id="link_1", mass_kg=1.5),
            uncertainty=UncertaintyEstimate(std=0.1, confidence=0.95),
            affective=AffectiveTag(salience=0.8, valence=0.2),
            action=MemoryAction.OBSERVE,
        )
        mid = embodied_memory.add_atom(atom)
        assert mid >= 1000

        # Verify get_atom roundtrip
        retrieved = embodied_memory.get_atom(mid)
        assert retrieved is not None
        assert retrieved.content == "full embodied memory"
        assert retrieved.spatial == Vec3(1.0, 2.0, 3.0)
        assert retrieved.temporal == TemporalInterval(0.0, 5.0)
        assert retrieved.physical.entity_id == "link_1"
        assert retrieved.uncertainty.confidence == 0.95
        assert retrieved.affective.salience == 0.8

    def test_get_atom_nonexistent(self, embodied_memory):
        assert embodied_memory.get_atom(999999) is None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_semantic_only(self, embodied_memory):
        # Add a few atoms
        for i in range(5):
            embodied_memory.add_atom(MemoryAtom(content=f"memory number {i}"))

        results = embodied_memory.search("memory", limit=10)
        assert len(results) == 5

    def test_search_spatial_filter(self, embodied_memory):
        # Add atoms at different positions
        embodied_memory.add_atom(MemoryAtom(content="near origin", spatial=Vec3(0.5, 0.5, 0.0)))
        embodied_memory.add_atom(MemoryAtom(content="far away", spatial=Vec3(100.0, 100.0, 0.0)))

        # Mock search returns all, spatial filter should narrow down
        results = embodied_memory.search(
            "memory",
            spatial_center=Vec3(0.0, 0.0, 0.0),
            spatial_radius=2.0,
        )
        contents = [r.content for r in results]
        assert "near origin" in contents
        assert "far away" not in contents

    def test_search_temporal_filter(self, embodied_memory):
        embodied_memory.add_atom(MemoryAtom(
            content="early event",
            temporal=TemporalInterval(0.0, 5.0),
        ))
        embodied_memory.add_atom(MemoryAtom(
            content="late event",
            temporal=TemporalInterval(100.0, 105.0),
        ))

        results = embodied_memory.search(
            "event",
            temporal_interval=TemporalInterval(2.0, 10.0),
            temporal_relation=IntervalRelation.OVERLAPS,
        )
        contents = [r.content for r in results]
        assert "early event" in contents
        assert "late event" not in contents

    def test_search_near(self, embodied_memory):
        embodied_memory.add_atom(MemoryAtom(content="A", spatial=Vec3(0.0, 0.0, 0.0)))
        embodied_memory.add_atom(MemoryAtom(content="B", spatial=Vec3(3.0, 4.0, 0.0)))
        embodied_memory.add_atom(MemoryAtom(content="C", spatial=Vec3(100.0, 0.0, 0.0)))

        results = embodied_memory.search_near(Vec3(0.0, 0.0, 0.0), radius=6.0)
        contents = [r.content for r in results]
        assert "A" in contents
        assert "B" in contents
        assert "C" not in contents

    def test_search_temporal(self, embodied_memory):
        embodied_memory.add_atom(MemoryAtom(
            content="morning",
            temporal=TemporalInterval(8.0, 10.0),
        ))
        embodied_memory.add_atom(MemoryAtom(
            content="evening",
            temporal=TemporalInterval(18.0, 20.0),
        ))

        results = embodied_memory.search_temporal(
            TemporalInterval(9.0, 11.0),
            IntervalRelation.OVERLAPS,
        )
        contents = [r.content for r in results]
        assert "morning" in contents
        assert "evening" not in contents


# ---------------------------------------------------------------------------
# Causal Graph
# ---------------------------------------------------------------------------

class TestCausalGraph:
    def test_causal_edges(self, embodied_memory):
        cause1 = embodied_memory.add_atom(MemoryAtom(content="cause 1"))
        cause2 = embodied_memory.add_atom(MemoryAtom(content="cause 2"))
        effect = embodied_memory.add_atom(MemoryAtom(
            content="effect",
            causal_parents=[cause1, cause2],
        ))

        causes = embodied_memory.get_causes(effect)
        cause_contents = [c.content for c in causes]
        assert "cause 1" in cause_contents
        assert "cause 2" in cause_contents

        effects = embodied_memory.get_effects(cause1)
        effect_contents = [e.content for e in effects]
        assert "effect" in effect_contents


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_atom(self, embodied_memory):
        atom = MemoryAtom(content="to be deleted", spatial=Vec3(1.0, 2.0, 3.0))
        mid = embodied_memory.add_atom(atom)

        # Verify exists in spatial index
        assert mid in embodied_memory.spatial_index.get_all_ids()

        ok = embodied_memory.delete_atom(mid)
        assert ok is True

        # Verify removed from spatial index
        assert mid not in embodied_memory.spatial_index.get_all_ids()
        assert embodied_memory.get_atom(mid) is None


# ---------------------------------------------------------------------------
# Ingest Pipeline
# ---------------------------------------------------------------------------

class TestIngestPipeline:
    def test_ingest_single_frame(self, embodied_memory):
        frame = SensorFrame(
            modality=Modality.RGB,
            timestamp_sec=0.0,
            data=[0.1, 0.2, 0.3, 0.4, 0.5],
            sensor_pose=Pose(position=Vec3(1.0, 0.0, 0.0)),
        )
        # First few frames pass surprisal gate (initialization phase)
        for i in range(15):
            embodied_memory.ingest(frame, content=f"frame {i}")

        # Flush to persist buffered frames
        mid = embodied_memory.flush_pipeline()
        assert mid is not None

    def test_flush_pipeline(self, embodied_memory):
        frame = SensorFrame(
            modality=Modality.DEPTH,
            timestamp_sec=1.0,
            data=[0.5] * 10,
            sensor_pose=Pose(position=Vec3(2.0, 0.0, 0.0)),
        )
        for i in range(5):
            embodied_memory.ingest(frame)

        # Force flush
        mid = embodied_memory.flush_pipeline()
        assert mid is not None

    def test_pipeline_stats(self, embodied_memory):
        stats = embodied_memory.get_pipeline().get_stats()
        assert "buffer_size" in stats
        assert "surprisal_state" in stats


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class TestPlugin:
    def test_plugin_disabled_by_default(self, embodied_memory):
        assert embodied_memory._plugin is None

    def test_plugin_enabled(self, sqlite_conn):
        mock_mem = MockMemory()
        em = EmbodiedMemory(
            memory=mock_mem,
            db_conn=sqlite_conn,
            enable_plugin=True,
            plugin_config={"enabled": True, "uncertainty_check": True},
        )
        assert em._plugin is not None
        assert em._plugin.enabled is True


# ---------------------------------------------------------------------------
# Backward Compatibility Proxy
# ---------------------------------------------------------------------------

class TestProxy:
    def test_proxy_agent_id(self, embodied_memory):
        assert embodied_memory.agent_id == "test_agent"

    def test_proxy_storage(self, embodied_memory):
        assert embodied_memory.storage is not None
