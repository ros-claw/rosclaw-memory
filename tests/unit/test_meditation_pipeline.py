"""
Tests for MeditationPipeline — offline abstraction pipeline.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional

import pytest

from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.memory_atom import MemoryAtom
from powermem.embodied.meditation_pipeline import (
    EntityConsolidator,
    MeditationPipeline,
    PatternExtractor,
    RelationCrystallizer,
)
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import MemoryAction, TemporalInterval, Vec3, WorldObject, Pose


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
# EntityConsolidator
# ---------------------------------------------------------------------------

class TestEntityConsolidator:
    def test_consolidate_same_entity_multiple_observations(self, embodied_memory):
        em = embodied_memory
        # Add 3 observations of same entity within 30s window
        for i in range(3):
            atom = MemoryAtom(
                content=f"obs {i}",
                spatial=Vec3(i * 0.1, 0, 0),
                temporal=TemporalInterval(i, i + 1),
                embodied_meta={"entity_id": "cup_01"},
            )
            em.add_atom(atom)

        # Manually set entity_id in DB (add_atom doesn't set it currently)
        cursor = em.db_conn.cursor()
        cursor.execute("UPDATE embodied_memories SET entity_id = 'cup_01' WHERE entity_id IS NULL")
        em.db_conn.commit()

        consolidator = EntityConsolidator(em.db_conn)
        events = consolidator.consolidate(window_sec=30.0)

        assert len(events) == 1
        assert events[0].entity_id == "cup_01"
        assert events[0].observation_count == 3

    def test_consolidate_splits_by_time_window(self, embodied_memory):
        em = embodied_memory
        # Two groups separated by 60s
        for i in range(2):
            atom = MemoryAtom(
                content=f"obs {i}",
                temporal=TemporalInterval(i * 60, i * 60 + 1),
                embodied_meta={"entity_id": "cup_02"},
            )
            em.add_atom(atom)

        cursor = em.db_conn.cursor()
        cursor.execute("UPDATE embodied_memories SET entity_id = 'cup_02' WHERE entity_id IS NULL")
        em.db_conn.commit()

        consolidator = EntityConsolidator(em.db_conn)
        events = consolidator.consolidate(window_sec=30.0)

        assert len(events) == 2

    def test_concept_index_written(self, embodied_memory):
        em = embodied_memory
        atom = MemoryAtom(
            content="single obs",
            temporal=TemporalInterval(0, 1),
            embodied_meta={"entity_id": "cup_03"},
        )
        em.add_atom(atom)

        cursor = em.db_conn.cursor()
        cursor.execute("UPDATE embodied_memories SET entity_id = 'cup_03' WHERE memory_id = ?", (atom.memory_id,))
        em.db_conn.commit()

        consolidator = EntityConsolidator(em.db_conn)
        consolidator.consolidate(window_sec=30.0)

        cursor.execute(
            "SELECT concept_id FROM embodied_concept_index WHERE dimension = 'entity'"
        )
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert "cup_03_event" in rows[0][0]


# ---------------------------------------------------------------------------
# RelationCrystallizer
# ---------------------------------------------------------------------------

class TestRelationCrystallizer:
    def test_crystallize_with_changed_objects(self, embodied_memory):
        em = embodied_memory
        # Add two world objects close to each other
        obj_a = WorldObject(
            obj_id="table_1", obj_type="box", name="table",
            pose=Pose(position=Vec3(0, 0, 0.4)),
            size=(1.0, 1.0, 0.8),
            scene_id="room",
        )
        obj_b = WorldObject(
            obj_id="mug_1", obj_type="cylinder", name="mug",
            pose=Pose(position=Vec3(0, 0, 0.86)),  # bottom at z=0.80, on table top
            size=(0.08, 0.08, 0.12),
            scene_id="room",
        )
        em.add_world_object(obj_a)
        em.add_world_object(obj_b)

        crystallizer = RelationCrystallizer(em)
        relations = crystallizer.crystallize(changed_obj_ids=["mug_1"])

        # mug is on table
        on_relations = [r for r in relations if r.relation == "on"]
        assert len(on_relations) >= 1

    def test_detect_changed_objects(self, embodied_memory):
        em = embodied_memory
        # Add a world object change event
        obj = WorldObject(
            obj_id="mug_2", obj_type="cylinder", name="mug",
            pose=Pose(position=Vec3(1, 0, 0.9)),
            size=(0.08, 0.08, 0.12),
            scene_id="room",
        )
        em.add_world_object(obj)
        em.update_world_object_pose("mug_2", Pose(position=Vec3(2, 0, 0.9)))

        crystallizer = RelationCrystallizer(em)
        changed = crystallizer._detect_changed_objects()
        assert "mug_2" in changed


# ---------------------------------------------------------------------------
# PatternExtractor
# ---------------------------------------------------------------------------

class TestPatternExtractor:
    def test_extract_action_outcome_pattern(self, embodied_memory):
        em = embodied_memory
        # Create 3 identical action->outcome chains
        for i in range(3):
            action_id = em.record_action("Pick up mug", spatial=Vec3(0, 0, 0.5))
            outcome_id = em.record_outcome(action_id, "Slipped", "failure")
            # Ensure outcome has failure status in meta
            cursor = em.db_conn.cursor()
            cursor.execute(
                "UPDATE embodied_memories SET embodied_meta = ? WHERE memory_id = ?",
                ('{"outcome_status": "failure"}', outcome_id),
            )
            em.db_conn.commit()

        extractor = PatternExtractor(em.db_conn)
        patterns = extractor.extract_patterns(lookback_hours=24, min_support=3)

        assert len(patterns) >= 1
        assert any("failure" in p.description for p in patterns)

    def test_min_support_filter(self, embodied_memory):
        em = embodied_memory
        # Only 1 action->outcome chain
        action_id = em.record_action("Pick up mug", spatial=Vec3(0, 0, 0.5))
        em.record_outcome(action_id, "Slipped", "failure")

        extractor = PatternExtractor(em.db_conn)
        patterns = extractor.extract_patterns(lookback_hours=24, min_support=3)

        assert len(patterns) == 0


# ---------------------------------------------------------------------------
# MeditationPipeline integration
# ---------------------------------------------------------------------------

class TestMeditationPipeline:
    def test_run_all_phases(self, embodied_memory):
        em = embodied_memory
        # Add some data
        for i in range(3):
            atom = MemoryAtom(
                content=f"obs {i}",
                temporal=TemporalInterval(i, i + 1),
                embodied_meta={"entity_id": "cup_10"},
            )
            em.add_atom(atom)
        cursor = em.db_conn.cursor()
        cursor.execute("UPDATE embodied_memories SET entity_id = 'cup_10' WHERE entity_id IS NULL")
        em.db_conn.commit()

        report = em.run_meditation(phases=["consolidate", "crystallize", "extract"])

        assert report.success is True
        assert report.consolidated_count >= 1
        assert report.elapsed_sec >= 0

    def test_run_single_phase(self, embodied_memory):
        em = embodied_memory
        report = em.run_meditation(phases=["consolidate"])
        assert report.success is True
        assert report.crystallized_count == 0  # not run
        assert report.extracted_patterns == 0  # not run

    def test_report_to_dict(self, embodied_memory):
        em = embodied_memory
        report = em.run_meditation()
        d = report.to_dict()
        assert "success" in d
        assert "elapsed_sec" in d
        assert "consolidated_count" in d
