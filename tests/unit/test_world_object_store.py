"""
Tests for WorldObjectStore and SceneGraph.
"""

from __future__ import annotations

import sqlite3

import pytest

from powermem.embodied.scene_graph import AABB, SceneGraph
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import Pose, Quaternion, SpatialRelation, Vec3, WorldObject
from powermem.embodied.world_object_store import WorldObjectStore


@pytest.fixture
def store():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    initialize_embodied_schema(conn)
    yield WorldObjectStore(conn)
    conn.close()


class TestWorldObjectStore:
    def test_save_and_load(self, store):
        obj = WorldObject(
            obj_id="cube_01",
            obj_type="box",
            name="red cube",
            pose=Pose(position=Vec3(1, 2, 3)),
            size=(0.1, 0.1, 0.1),
            scene_id="scene_a",
        )
        store.save(obj)
        loaded = store.load("cube_01")
        assert loaded is not None
        assert loaded.obj_id == "cube_01"
        assert loaded.name == "red cube"
        assert loaded.pose.position == Vec3(1, 2, 3)

    def test_load_missing(self, store):
        assert store.load("nonexistent") is None

    def test_list_by_scene(self, store):
        store.save(WorldObject(obj_id="a1", name="a", scene_id="s1"))
        store.save(WorldObject(obj_id="a2", name="b", scene_id="s1"))
        store.save(WorldObject(obj_id="b1", name="c", scene_id="s2"))
        results = store.list_by_scene("s1")
        assert len(results) == 2
        assert {r.obj_id for r in results} == {"a1", "a2"}

    def test_list_by_type(self, store):
        store.save(WorldObject(obj_id="s1", obj_type="sphere", name="s"))
        store.save(WorldObject(obj_id="b1", obj_type="box", name="b"))
        results = store.list_by_type("sphere")
        assert len(results) == 1
        assert results[0].obj_id == "s1"

    def test_update_pose(self, store):
        store.save(WorldObject(obj_id="obj1", pose=Pose(position=Vec3(0, 0, 0))))
        ok = store.update_pose("obj1", Pose(position=Vec3(1, 1, 1)), state="moved")
        assert ok is True
        loaded = store.load("obj1")
        assert loaded.pose.position == Vec3(1, 1, 1)
        assert loaded.state == "moved"

    def test_update_state(self, store):
        store.save(WorldObject(obj_id="obj1", state="present"))
        ok = store.update_state("obj1", "removed")
        assert ok is True
        assert store.load("obj1").state == "removed"

    def test_delete(self, store):
        store.save(WorldObject(obj_id="del1"))
        assert store.delete("del1") is True
        assert store.load("del1") is None

    def test_load_many(self, store):
        store.save(WorldObject(obj_id="m1"))
        store.save(WorldObject(obj_id="m2"))
        store.save(WorldObject(obj_id="m3"))
        results = store.load_many(["m1", "m3"])
        assert len(results) == 2
        assert {r.obj_id for r in results} == {"m1", "m3"}


class TestRelations:
    def test_add_and_get_relations(self, store):
        store.save(WorldObject(obj_id="table"))
        store.save(WorldObject(obj_id="cup"))
        rid = store.add_relation(SpatialRelation("cup", "table", "on", 0.95))
        assert rid is not None
        rels = store.get_relations("cup", direction="outgoing")
        assert len(rels) == 1
        assert rels[0].relation == "on"

    def test_get_relations_both_directions(self, store):
        store.save(WorldObject(obj_id="a"))
        store.save(WorldObject(obj_id="b"))
        store.add_relation(SpatialRelation("a", "b", "next_to"))
        outgoing = store.get_relations("a", direction="outgoing")
        incoming = store.get_relations("b", direction="incoming")
        assert len(outgoing) == 1
        assert len(incoming) == 1

    def test_scene_graph(self, store):
        store.save(WorldObject(obj_id="room", scene_id="home"))
        store.save(WorldObject(obj_id="table", scene_id="home", parent_obj_id="room"))
        store.save(WorldObject(obj_id="chair", scene_id="home", parent_obj_id="room"))
        graph = store.get_scene_graph("home")
        assert "room" in graph
        assert "table" in graph
        assert "chair" in graph


class TestSceneGraphCompute:
    def test_compute_on_relation(self, store):
        # Table at z=0, size 1x1x1 -> top at z=0.5
        store.save(WorldObject(
            obj_id="table",
            obj_type="box",
            pose=Pose(position=Vec3(0, 0, 0)),
            size=(1.0, 1.0, 1.0),
            scene_id="s1",
        ))
        # Cup on top of table at z=0.501, size 0.1x0.1x0.1 -> bottom at z=0.451
        store.save(WorldObject(
            obj_id="cup",
            obj_type="box",
            pose=Pose(position=Vec3(0, 0, 0.501)),
            size=(0.1, 0.1, 0.1),
            scene_id="s1",
        ))
        sg = SceneGraph("s1", store)
        relations = sg.compute_relations(spatial_tolerance=0.05)
        on_rels = [r for r in relations if r.relation == "on"]
        assert len(on_rels) == 1
        assert on_rels[0].subject_id == "cup"
        assert on_rels[0].object_id == "table"

    def test_compute_in_relation(self, store):
        # Big box
        store.save(WorldObject(
            obj_id="box_big",
            obj_type="box",
            pose=Pose(position=Vec3(0, 0, 0)),
            size=(2.0, 2.0, 2.0),
            scene_id="s1",
        ))
        # Small box inside
        store.save(WorldObject(
            obj_id="box_small",
            obj_type="box",
            pose=Pose(position=Vec3(0, 0, 0)),
            size=(0.5, 0.5, 0.5),
            scene_id="s1",
        ))
        sg = SceneGraph("s1", store)
        relations = sg.compute_relations(spatial_tolerance=0.01)
        in_rels = [r for r in relations if r.relation == "in"]
        assert len(in_rels) == 1
        assert in_rels[0].subject_id == "box_small"
        assert in_rels[0].object_id == "box_big"

    def test_compute_next_to(self, store):
        store.save(WorldObject(
            obj_id="obj_a",
            obj_type="box",
            pose=Pose(position=Vec3(0, 0, 0)),
            size=(0.5, 0.5, 0.5),
            scene_id="s1",
        ))
        store.save(WorldObject(
            obj_id="obj_b",
            obj_type="box",
            pose=Pose(position=Vec3(0.4, 0, 0)),
            size=(0.5, 0.5, 0.5),
            scene_id="s1",
        ))
        sg = SceneGraph("s1", store)
        relations = sg.compute_relations(spatial_tolerance=0.05)
        next_to_rels = [r for r in relations if r.relation == "next_to"]
        assert len(next_to_rels) >= 1
