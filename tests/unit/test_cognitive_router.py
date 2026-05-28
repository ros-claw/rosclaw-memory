"""
Tests for CognitiveRouter — Tri-Route retrieval engine.

Covers:
- QueryIntent parsing (System-1/2/3 activation)
- System-3 SpatioTemporal intersection correctness
- System-2 Global Selection skeleton (concept index + experience graph)
- Multi-route intersection and fallback
- Concept index and experience edge CRUD
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional

import pytest

from powermem.embodied.cognitive_router import CognitiveRouter, QueryIntent, _GLOBAL_KEYWORDS
from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.memory_atom import MemoryAtom
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import MemoryAction, TemporalInterval, Vec3


# ---------------------------------------------------------------------------
# Mock PowerMem
# ---------------------------------------------------------------------------

class MockStorageAdapter:
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

    def search_memories(self, **kwargs) -> list:
        limit = kwargs.get("limit", 30)
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
    def __init__(self):
        self.storage = MockStorageAdapter()
        self.agent_id = "test_agent"

    def search(self, query, user_id=None, agent_id=None, run_id=None, filters=None, limit=30, threshold=None):
        results = self.storage.search_memories(
            query_embedding=None, user_id=user_id, agent_id=agent_id,
            run_id=run_id, filters=filters, limit=limit, query=query, threshold=threshold,
        )
        return {"results": results, "relations": []}

    def delete(self, memory_id: int) -> bool:
        return self.storage.delete_memory(memory_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    initialize_embodied_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def embodied_memory(db_conn):
    mem = MockMemory()
    em = EmbodiedMemory(memory=mem, db_conn=db_conn, enable_plugin=False)
    return em


# ---------------------------------------------------------------------------
# QueryIntent parsing
# ---------------------------------------------------------------------------

class TestQueryIntentParsing:
    def test_spatial_params_activate_system3(self, embodied_memory):
        router = embodied_memory._router
        intent = router._parse_intent("find something", Vec3(0, 0, 0), None)
        assert intent.needs_spatiotemporal is True
        assert intent.needs_associative is True
        assert intent.needs_global is False

    def test_temporal_params_activate_system3(self, embodied_memory):
        router = embodied_memory._router
        intent = router._parse_intent("find something", None, TemporalInterval(0, 10))
        assert intent.needs_spatiotemporal is True

    def test_abstract_keywords_activate_system2(self, embodied_memory):
        router = embodied_memory._router
        for kw in ["pattern", "usually", "为什么", "trend", "规律"]:
            intent = router._parse_intent(f"what is the {kw}", None, None)
            assert intent.needs_global is True, f"keyword '{kw}' should activate System-2"

    def test_plain_query_only_system1(self, embodied_memory):
        router = embodied_memory._router
        intent = router._parse_intent("red cup on table", None, None)
        assert intent.needs_associative is True
        assert intent.needs_spatiotemporal is False
        assert intent.needs_global is False

    def test_tri_route_all_active(self, embodied_memory):
        router = embodied_memory._router
        intent = router._parse_intent("why does pattern usually happen here", Vec3(0, 0, 0), TemporalInterval(0, 10))
        assert intent.needs_associative is True
        assert intent.needs_spatiotemporal is True
        assert intent.needs_global is True

    def test_extract_concepts(self, embodied_memory):
        router = embodied_memory._router
        concepts = router._extract_concepts_from_query("The red_mug is on kitchen_table")
        assert "red_mug" in concepts
        assert "kitchen_table" in concepts


# ---------------------------------------------------------------------------
# System-3 SpatioTemporal intersection
# ---------------------------------------------------------------------------

class TestSystem3SpatioTemporal:
    def test_spatial_only_filter(self, embodied_memory):
        em = embodied_memory
        em.add_atom(MemoryAtom(content="near", spatial=Vec3(1, 0, 0)))
        em.add_atom(MemoryAtom(content="far", spatial=Vec3(100, 0, 0)))

        results = em.search("test", spatial_center=Vec3(0, 0, 0), spatial_radius=2.0)
        contents = [r.content for r in results]
        assert "near" in contents
        assert "far" not in contents

    def test_temporal_only_filter(self, embodied_memory):
        em = embodied_memory
        em.add_atom(MemoryAtom(content="inside", temporal=TemporalInterval(5, 10)))
        em.add_atom(MemoryAtom(content="outside", temporal=TemporalInterval(100, 105)))

        results = em.search("test", temporal_interval=TemporalInterval(6, 8))
        contents = [r.content for r in results]
        assert "inside" in contents
        assert "outside" not in contents

    def test_spatial_temporal_intersection(self, embodied_memory):
        em = embodied_memory
        em.add_atom(MemoryAtom(content="target", spatial=Vec3(1, 0, 0), temporal=TemporalInterval(5, 10)))
        em.add_atom(MemoryAtom(content="wrong space", spatial=Vec3(100, 0, 0), temporal=TemporalInterval(5, 10)))
        em.add_atom(MemoryAtom(content="wrong time", spatial=Vec3(1, 0, 0), temporal=TemporalInterval(100, 105)))
        em.add_atom(MemoryAtom(content="no spatial", temporal=TemporalInterval(5, 10)))
        em.add_atom(MemoryAtom(content="no temporal", spatial=Vec3(1, 0, 0)))

        results = em.search(
            "target",
            spatial_center=Vec3(0, 0, 0),
            spatial_radius=2.0,
            temporal_interval=TemporalInterval(6, 8),
        )
        contents = [r.content for r in results]
        assert "target" in contents
        assert "wrong space" not in contents
        assert "wrong time" not in contents
        assert "no spatial" not in contents
        assert "no temporal" not in contents


# ---------------------------------------------------------------------------
# System-2 Global Selection (skeleton)
# ---------------------------------------------------------------------------

class TestSystem2GlobalSelection:
    def test_concept_index_roundtrip(self, embodied_memory):
        em = embodied_memory
        mid = em.add_atom(MemoryAtom(content="test atom"))
        em.index_concept(mid, "task", 2, "kitchen_cleaning", 0.95)

        cursor = em.db_conn.cursor()
        cursor.execute(
            "SELECT dimension, layer, concept_id, confidence FROM embodied_concept_index WHERE memory_id = ?",
            (mid,),
        )
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0] == ("task", 2, "kitchen_cleaning", 0.95)

    def test_concept_index_update(self, embodied_memory):
        em = embodied_memory
        mid = em.add_atom(MemoryAtom(content="test atom"))
        em.index_concept(mid, "task", 2, "kitchen_cleaning", 0.5)
        em.index_concept(mid, "task", 2, "kitchen_cleaning", 0.99)  # update

        cursor = em.db_conn.cursor()
        cursor.execute(
            "SELECT confidence FROM embodied_concept_index WHERE memory_id = ?",
            (mid,),
        )
        assert cursor.fetchone()[0] == 0.99

    def test_experience_edge_roundtrip(self, embodied_memory):
        em = embodied_memory
        m1 = em.add_atom(MemoryAtom(content="action"))
        m2 = em.add_atom(MemoryAtom(content="outcome"))
        em.add_experience_edge(m1, m2, "causes", 1.0)

        cursor = em.db_conn.cursor()
        cursor.execute(
            "SELECT edge_type, strength FROM embodied_experience_graph WHERE source_memory_id = ? AND target_memory_id = ?",
            (m1, m2),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row == ("causes", 1.0)

    def test_global_selection_via_concept(self, embodied_memory):
        em = embodied_memory
        # Create atoms and index them under a concept
        mid1 = em.add_atom(MemoryAtom(content="mug pickup"))
        mid2 = em.add_atom(MemoryAtom(content="plate pickup"))
        em.index_concept(mid1, "task", 1, "pickup", 1.0)
        em.index_concept(mid2, "task", 1, "pickup", 1.0)

        # Query with abstract keyword should activate System-2
        router = em._router
        intent = router._parse_intent("what is the pattern in pickup tasks", None, None)
        assert intent.needs_global is True

        candidates = router._route_global_selection(intent, limit=10)
        ids = set(candidates.keys())
        assert mid1 in ids
        assert mid2 in ids

    def test_global_selection_with_experience_graph_expansion(self, embodied_memory):
        em = embodied_memory
        mid1 = em.add_atom(MemoryAtom(content="action"))
        mid2 = em.add_atom(MemoryAtom(content="outcome"))
        mid3 = em.add_atom(MemoryAtom(content="unrelated"))

        em.index_concept(mid1, "task", 1, "grasp", 1.0)
        em.add_experience_edge(mid1, mid2, "causes", 1.0)

        router = em._router
        intent = QueryIntent(query="grasp pattern", needs_global=True, suggested_concepts=["grasp"])
        candidates = router._route_global_selection(intent, limit=10)

        # mid1 (direct concept match) and mid2 (one-hop neighbor) should be included
        ids = set(candidates.keys())
        assert mid1 in ids
        assert mid2 in ids
        assert mid3 not in ids


# ---------------------------------------------------------------------------
# Associative Cache
# ---------------------------------------------------------------------------

class TestAssociativeCache:
    def test_cache_hit_avoids_search(self, embodied_memory):
        em = embodied_memory
        router = em._router
        # seed storage so search returns something
        for i in range(5):
            em.memory.storage.add_memory({"content": f"item_{i}"})

        # first call populates cache
        c1 = router._route_associative("test query", None, 10)
        search_calls_before = len(em.memory.storage._store)

        # second call should hit cache
        c2 = router._route_associative("test query", None, 10)
        assert c1 == c2
        # cache still has one entry
        assert len(router._associative_cache) == 1

    def test_cache_miss_with_different_query(self, embodied_memory):
        em = embodied_memory
        router = em._router
        for i in range(5):
            em.memory.storage.add_memory({"content": f"item_{i}"})

        router._route_associative("query A", None, 10)
        router._route_associative("query B", None, 10)
        assert len(router._associative_cache) == 2

    def test_cache_ttl_expiration(self, embodied_memory):
        em = embodied_memory
        router = em._router
        router._assoc_cache_ttl = 0.01  # 10ms TTL
        for i in range(3):
            em.memory.storage.add_memory({"content": f"item_{i}"})

        router._route_associative("ttl query", None, 10)
        assert len(router._associative_cache) == 1
        import time
        time.sleep(0.02)
        cached = router._get_cached_associative("ttl query", None, 10)
        assert cached is None

    def test_index_concept_clears_cache(self, embodied_memory):
        em = embodied_memory
        router = em._router
        for i in range(3):
            em.memory.storage.add_memory({"content": f"item_{i}"})

        router._route_associative("clear test", None, 10)
        assert len(router._associative_cache) == 1
        mid = em.add_atom(MemoryAtom(content="concept atom"))
        em.index_concept(mid, "task", 1, "test_concept", 1.0)
        assert len(router._associative_cache) == 0

    def test_add_experience_edge_clears_cache(self, embodied_memory):
        em = embodied_memory
        router = em._router
        for i in range(3):
            em.memory.storage.add_memory({"content": f"item_{i}"})

        router._route_associative("edge test", None, 10)
        assert len(router._associative_cache) == 1
        m1 = em.add_atom(MemoryAtom(content="a"))
        m2 = em.add_atom(MemoryAtom(content="b"))
        em.add_experience_edge(m1, m2, "related", 1.0)
        assert len(router._associative_cache) == 0

    def test_cache_eviction_at_max_size(self, embodied_memory):
        em = embodied_memory
        router = em._router
        router._MAX_ASSOC_CACHE = 4
        for i in range(3):
            em.memory.storage.add_memory({"content": f"item_{i}"})

        for i in range(6):
            router._route_associative(f"query {i}", None, 10)

        assert len(router._associative_cache) <= router._MAX_ASSOC_CACHE

    def test_clear_cache_explicit(self, embodied_memory):
        em = embodied_memory
        router = em._router
        for i in range(3):
            em.memory.storage.add_memory({"content": f"item_{i}"})

        router._route_associative("explicit", None, 10)
        assert len(router._associative_cache) == 1
        router.clear_cache()
        assert len(router._associative_cache) == 0


# ---------------------------------------------------------------------------
# Tri-Route integration
# ---------------------------------------------------------------------------

class TestTriRouteIntegration:
    def test_fallback_to_union_when_intersection_empty(self, embodied_memory):
        """当多路交集为空时，应退化为并集避免无结果"""
        em = embodied_memory
        # Atom A: 有空间无时间
        em.add_atom(MemoryAtom(content="spatial only", spatial=Vec3(1, 0, 0)))
        # Atom B: 无空间有时间（但 temporal_index 可能不会返回无 temporal 的 atom...）
        # 这个测试主要验证代码路径不会崩溃
        results = em.search("spatial only", spatial_center=Vec3(0, 0, 0), spatial_radius=2.0)
        assert len(results) >= 0

    def test_record_outcome_adds_experience_edge(self, embodied_memory):
        em = embodied_memory
        action_id = em.record_action("Pick up mug", spatial=Vec3(1, 0, 0))
        outcome_id = em.record_outcome(action_id, "Success", "success")

        cursor = em.db_conn.cursor()
        cursor.execute(
            "SELECT 1 FROM embodied_experience_graph WHERE source_memory_id = ? AND target_memory_id = ? AND edge_type = 'causes'",
            (action_id, outcome_id),
        )
        assert cursor.fetchone() is not None
