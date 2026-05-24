"""
Integration tests for EmbodiedMemory world object methods.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

import pytest

from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.memory_atom import MemoryAtom
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import Pose, Vec3, WorldObject


class _MockStorageAdapter:
    def __init__(self):
        self._store: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1000

    def add_memory(self, payload: Dict[str, Any]) -> int:
        mid = self._next_id
        self._next_id += 1
        self._store[mid] = {"id": mid, "content": payload.get("content", ""), "metadata": payload.get("metadata", {})}
        return mid

    def get_memory(self, memory_id: int) -> Any:
        return self._store.get(memory_id)

    def delete_memory(self, memory_id: int, **kwargs) -> bool:
        return self._store.pop(memory_id, None) is not None

    def search_memories(self, **kwargs) -> list:
        limit = kwargs.get("limit", 30)
        results = []
        for mid, item in list(self._store.items())[:limit]:
            results.append({"id": mid, "memory": item["content"], "score": 0.9, "metadata": item.get("metadata", {})})
        return results

    def update_memory(self, memory_id: int, content: str, **kwargs) -> Dict[str, Any]:
        item = self._store.get(memory_id)
        if item is None:
            raise KeyError(memory_id)
        item["content"] = content
        return item


class _MockMemory:
    def __init__(self):
        self.storage = _MockStorageAdapter()

    def add(self, content, **kwargs):
        return self.storage.add_memory({"content": content, "metadata": kwargs.get("metadata", {})})

    def search(self, query, **kwargs):
        results = self.storage.search_memories(limit=kwargs.get("limit", 30))
        return {"results": results, "relations": []}

    def get(self, memory_id, **kwargs):
        return self.storage.get_memory(memory_id)

    def delete(self, memory_id):
        return self.storage.delete_memory(memory_id)

    def update(self, memory_id, content, **kwargs):
        return self.storage.update_memory(memory_id, content)


@pytest.fixture
def em():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    initialize_embodied_schema(conn)
    mock_mem = _MockMemory()
    instance = EmbodiedMemory(memory=mock_mem, db_conn=conn, enable_plugin=False)
    yield instance
    conn.close()


class TestAddWorldObject:
    def test_add_world_object_creates_memory_atom(self, em):
        obj = WorldObject(
            obj_id="cube_01",
            obj_type="box",
            name="red cube",
            pose=Pose(position=Vec3(1, 2, 3)),
            scene_id="scene_a",
        )
        oid = em.add_world_object(obj)
        assert oid == "cube_01"

        # Should be retrievable from store
        loaded = em.get_world_object("cube_01")
        assert loaded is not None
        assert loaded.memory_id is not None

    def test_add_world_objects_from_parse_result(self, em):
        class FakeParseResult:
            world_objects = [
                {"id": "obj1", "name": "table", "type": "box", "pose": {"position": {"x": 0, "y": 0, "z": 0}}},
                {"id": "obj2", "name": "mug", "type": "cylinder", "pose": {"position": {"x": 1, "y": 0, "z": 0}}},
            ]

        ids = em.add_world_objects(FakeParseResult())
        assert len(ids) == 2
        assert "obj1" in ids
        assert em.get_world_object("obj1") is not None


class TestSearchWorldObjects:
    def test_search_by_scene(self, em):
        em.add_world_object(WorldObject(obj_id="a", pose=Pose(position=Vec3(0, 0, 0)), scene_id="s1"))
        em.add_world_object(WorldObject(obj_id="b", pose=Pose(position=Vec3(10, 0, 0)), scene_id="s1"))
        em.add_world_object(WorldObject(obj_id="c", pose=Pose(position=Vec3(0, 0, 0)), scene_id="s2"))

        results = em.search_world_objects(Vec3(0, 0, 0), radius=1.0, scene_id="s1")
        assert len(results) == 1
        assert results[0].obj_id == "a"

    def test_search_without_scene_uses_spatial_index(self, em):
        em.add_world_object(WorldObject(obj_id="near", pose=Pose(position=Vec3(0, 0, 0)), scene_id="s1"))
        em.add_world_object(WorldObject(obj_id="far", pose=Pose(position=Vec3(100, 0, 0)), scene_id="s1"))

        results = em.search_world_objects(Vec3(0, 0, 0), radius=5.0)
        ids = [r.obj_id for r in results]
        assert "near" in ids
        assert "far" not in ids


class TestUpdatePose:
    def test_update_pose_records_history(self, em):
        em.add_world_object(WorldObject(obj_id="movable", pose=Pose(position=Vec3(0, 0, 0)), scene_id="s1"))
        ok = em.update_world_object_pose("movable", Pose(position=Vec3(1, 1, 1)), state="moved")
        assert ok is True

        loaded = em.get_world_object("movable")
        assert loaded.pose.position == Vec3(1, 1, 1)
        assert loaded.state == "moved"


class TestSceneGraphIntegration:
    def test_get_scene_graph(self, em):
        em.add_world_object(WorldObject(obj_id="room", scene_id="home"))
        em.add_world_object(WorldObject(obj_id="table", scene_id="home", parent_obj_id="room"))
        em.add_world_object(WorldObject(obj_id="cup", scene_id="home", parent_obj_id="table"))

        sg = em.get_scene_graph("home")
        assert len(sg.get_objects()) == 3
        children = sg.get_children("room")
        assert len(children) == 1
        assert children[0].obj_id == "table"

    def test_auto_compute_relations(self, em):
        # Table at z=0, size 1x1x1
        em.add_world_object(WorldObject(
            obj_id="table",
            obj_type="box",
            pose=Pose(position=Vec3(0, 0, 0)),
            size=(1.0, 1.0, 1.0),
            scene_id="kitchen",
        ))
        # Cup on table
        em.add_world_object(WorldObject(
            obj_id="cup",
            obj_type="box",
            pose=Pose(position=Vec3(0, 0, 0.6)),
            size=(0.1, 0.1, 0.1),
            scene_id="kitchen",
        ))
        relations = em.auto_compute_relations("kitchen", spatial_tolerance=0.05)
        on_rels = [r for r in relations if r.relation == "on"]
        assert len(on_rels) == 1
        assert on_rels[0].subject_id == "cup"
        assert on_rels[0].object_id == "table"
