"""
Tests for ObjectPermanence — 对象恒存引擎

被遮挡 ≠ 消失
"""

from __future__ import annotations

import sqlite3

import pytest

from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.object_permanence import ObjectPermanenceTracker, PermanenceReport
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import Pose, Vec3, WorldObject


class MockStorageAdapter:
    def __init__(self):
        self._store = {}
        self._next_id = 1000

    def add_memory(self, payload):
        mid = self._next_id
        self._next_id += 1
        self._store[mid] = {"id": mid, "content": payload.get("content", "")}
        return mid

    def get_memory(self, memory_id):
        return self._store.get(memory_id)

    def search_memories(self, **kwargs):
        return []

    def delete_memory(self, memory_id, **kwargs):
        return self._store.pop(memory_id, None) is not None


class MockMemory:
    def __init__(self):
        self.storage = MockStorageAdapter()
        self.agent_id = "test_agent"

    def search(self, query, **kwargs):
        return {"results": [], "relations": []}

    def delete(self, memory_id):
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


@pytest.fixture
def tracker(embodied_memory):
    return ObjectPermanenceTracker(
        embodied_memory.world_object_store,
        decay_rate=0.05,
        missing_threshold=0.2,
    )


class TestObjectPermanenceVisible:
    def test_visible_object_confirms_pose(self, tracker, embodied_memory):
        # Pre-existing object
        embodied_memory.add_world_object(WorldObject(
            obj_id="cup_1", obj_type="cylinder", name="cup",
            pose=Pose(position=Vec3(1, 0, 0)), scene_id="kitchen",
        ))

        # Detect same object at new position
        detections = [WorldObject(
            obj_id="cup_1", obj_type="cylinder", name="cup",
            pose=Pose(position=Vec3(2, 0, 0)), scene_id="kitchen",
        )]
        report = tracker.sync_detections("kitchen", detections, timestamp_sec=10.0)

        assert "cup_1" in report.visible
        obj = embodied_memory.get_world_object("cup_1")
        assert obj.pose.position == Vec3(2, 0, 0)
        assert obj.occlusion_status == "visible"
        assert obj.confidence == pytest.approx(1.0)
        assert obj.last_seen_sec == 10.0

    def test_visible_no_change(self, tracker, embodied_memory):
        embodied_memory.add_world_object(WorldObject(
            obj_id="plate_1", obj_type="box", scene_id="kitchen",
        ))
        detections = [WorldObject(
            obj_id="plate_1", obj_type="box", scene_id="kitchen",
        )]
        report = tracker.sync_detections("kitchen", detections, timestamp_sec=1.0)
        assert "plate_1" in report.visible
        assert len(report.transitions) == 0


class TestObjectPermanenceOcclusion:
    def test_unseen_object_decays_confidence(self, tracker, embodied_memory):
        # Object exists but not detected this frame
        embodied_memory.add_world_object(WorldObject(
            obj_id="mug_1", obj_type="cylinder", scene_id="kitchen",
            pose=Pose(position=Vec3(0, 0, 0)),
            last_seen_sec=0.0, confidence=1.0,
        ))

        # Empty detection list
        report = tracker.sync_detections("kitchen", [], timestamp_sec=10.0)

        assert "mug_1" in report.occluded
        assert len(report.transitions) == 1
        assert "visible -> occluded" in report.transitions[0]

        obj = embodied_memory.get_world_object("mug_1")
        assert obj.occlusion_status == "occluded"
        # confidence = 1.0 * exp(-0.05 * 10) ≈ 0.606
        assert obj.confidence == pytest.approx(0.606, abs=0.01)

    def test_confidence_below_threshold_becomes_missing(self, tracker, embodied_memory):
        # Object with very low confidence, long time unseen
        embodied_memory.add_world_object(WorldObject(
            obj_id="fork_1", obj_type="mesh", scene_id="kitchen",
            pose=Pose(position=Vec3(0, 0, 0)),
            occlusion_status="occluded", confidence=0.25, last_seen_sec=0.0,
        ))

        # 30 seconds later, still not detected
        report = tracker.sync_detections("kitchen", [], timestamp_sec=30.0)

        assert "fork_1" in report.missing
        assert "-> missing" in report.transitions[0]

        obj = embodied_memory.get_world_object("fork_1")
        assert obj.occlusion_status == "missing"
        assert obj.confidence < 0.2


class TestObjectPermanenceRedetection:
    def test_occluded_object_redetected_restores(self, tracker, embodied_memory):
        # Object was occluded
        embodied_memory.add_world_object(WorldObject(
            obj_id="bowl_1", obj_type="sphere", scene_id="kitchen",
            pose=Pose(position=Vec3(1, 1, 0)),
            occlusion_status="occluded", confidence=0.3, last_seen_sec=5.0,
        ))

        # Now detected again
        detections = [WorldObject(
            obj_id="bowl_1", obj_type="sphere", scene_id="kitchen",
            pose=Pose(position=Vec3(1.1, 1.0, 0)),
        )]
        report = tracker.sync_detections("kitchen", detections, timestamp_sec=10.0)

        assert "bowl_1" in report.visible
        assert "occluded -> visible" in report.transitions[0]

        obj = embodied_memory.get_world_object("bowl_1")
        assert obj.occlusion_status == "visible"
        assert obj.confidence == pytest.approx(1.0)

    def test_redetect_by_spatial_match(self, tracker, embodied_memory):
        # Object without obj_id in detection, matched by proximity
        embodied_memory.add_world_object(WorldObject(
            obj_id="spoon_1", obj_type="mesh", scene_id="kitchen",
            pose=Pose(position=Vec3(2, 2, 0)),
            occlusion_status="occluded", confidence=0.5, last_seen_sec=0.0,
        ))

        # Detection has no obj_id but same type and close position
        detections = [WorldObject(
            obj_id="", obj_type="mesh", scene_id="kitchen",
            pose=Pose(position=Vec3(2.1, 2.0, 0)),
        )]
        report = tracker.sync_detections(
            "kitchen", detections, timestamp_sec=5.0, occlusion_radius=0.5
        )

        # Should match spoon_1 by spatial proximity
        obj = embodied_memory.get_world_object("spoon_1")
        assert obj.occlusion_status == "visible"
        assert obj.confidence == pytest.approx(1.0)


class TestObjectPermanenceNewObjects:
    def test_new_detection_adds_object(self, tracker, embodied_memory):
        detections = [WorldObject(
            obj_id="", obj_type="box", name="new_box",
            pose=Pose(position=Vec3(5, 5, 0)), scene_id="kitchen",
        )]
        report = tracker.sync_detections("kitchen", detections, timestamp_sec=1.0)

        assert len(report.added) == 1
        added_id = report.added[0]
        obj = embodied_memory.get_world_object(added_id)
        assert obj is not None
        assert obj.obj_type == "box"
        assert obj.occlusion_status == "visible"
        assert obj.confidence == pytest.approx(1.0)


class TestObjectPermanenceEmbodiedMemoryIntegration:
    def test_sync_scene_objects_through_embodied_memory(self, embodied_memory):
        # Pre-existing
        embodied_memory.add_world_object(WorldObject(
            obj_id="pot_1", obj_type="cylinder", scene_id="dining",
            pose=Pose(position=Vec3(0, 0, 0)),
        ))

        report = embodied_memory.sync_scene_objects(
            scene_id="dining",
            detections=[WorldObject(
                obj_id="pot_1", obj_type="cylinder",
                pose=Pose(position=Vec3(0.5, 0, 0)), scene_id="dining",
            )],
            timestamp_sec=2.0,
        )

        assert "pot_1" in report.visible
        obj = embodied_memory.get_world_object("pot_1")
        assert obj.pose.position == Vec3(0.5, 0, 0)
