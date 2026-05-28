"""
Tests for Protocol definitions — v1.0 type-safe integration

These tests verify that powermem's concrete types satisfy the
Protocol contracts that v1.0 MemoryInterface should use as type
annotations (ISSUE-001 in audit-memory.md).
"""

import pytest

from powermem.embodied.protocols import (
    EmbodiedMemoryLike,
    MemoryAtomLike,
    PermanenceReportLike,
    PoseLike,
    QuaternionLike,
    SceneGraphLike,
    SpatialRelationLike,
    TelemetryLike,
    TemporalIntervalLike,
    Vec3Like,
    WorldObjectLike,
)
from powermem.embodied.types import (
    Pose,
    Quaternion,
    SpatialRelation,
    TemporalInterval,
    Vec3,
    WorldObject,
)
from powermem.embodied.memory_atom import MemoryAtom
from powermem.embodied.telemetry import MemoryTelemetry


class TestProtocolConformance:
    """Concrete powermem types must satisfy their Protocol counterparts."""

    def test_vec3_satisfies_vec3like(self):
        v = Vec3(1.0, 2.0, 3.0)
        assert isinstance(v, Vec3Like)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0

    def test_quaternion_satisfies_quaternionlike(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert isinstance(q, QuaternionLike)

    def test_pose_satisfies_poselike(self):
        p = Pose(position=Vec3(0, 0, 0), orientation=Quaternion(1, 0, 0, 0))
        assert isinstance(p, PoseLike)
        assert isinstance(p.position, Vec3Like)
        assert isinstance(p.orientation, QuaternionLike)

    def test_temporal_interval_satisfies_protocol(self):
        ti = TemporalInterval(start_sec=0.0, end_sec=10.0)
        assert isinstance(ti, TemporalIntervalLike)
        assert ti.start_sec == 0.0
        assert ti.end_sec == 10.0
        assert ti.frame_id == "wall_clock"

    def test_world_object_satisfies_protocol(self):
        wo = WorldObject(
            obj_id="cup_1",
            obj_type="box",
            name="red cup",
            pose=Pose(position=Vec3(1, 2, 0.5), orientation=Quaternion(1, 0, 0, 0)),
            scene_id="kitchen",
        )
        assert isinstance(wo, WorldObjectLike)
        assert wo.obj_id == "cup_1"
        assert wo.confidence == 1.0
        assert wo.occlusion_status == "visible"
        assert wo.to_dict()["obj_id"] == "cup_1"

    def test_spatial_relation_satisfies_protocol(self):
        rel = SpatialRelation(
            subject_id="cup_1",
            object_id="table_1",
            relation="on",
            confidence=0.95,
        )
        assert isinstance(rel, SpatialRelationLike)
        assert rel.relation == "on"

    def test_memory_atom_satisfies_protocol(self):
        atom = MemoryAtom(
            content="red cup on table",
            spatial=Vec3(1.0, 2.0, 0.5),
        )
        assert isinstance(atom, MemoryAtomLike)
        assert atom.content == "red cup on table"
        assert isinstance(atom.spatial, Vec3Like)
        assert isinstance(atom.embodied_meta, dict)

    def test_telemetry_satisfies_protocol(self):
        t = MemoryTelemetry(enabled=True)
        assert isinstance(t, TelemetryLike)
        assert t.enabled
        s = t.snapshot()
        assert isinstance(s, dict)
        p = t.prometheus_metrics()
        assert isinstance(p, str)
        t.reset()


class TestProtocolDuckTyping:
    """Protocols should be satisfied by arbitrary duck-typed objects
    (no inheritance from powermem types required)."""

    def test_duck_vec3(self):
        class MyVec:
            x = 1.0
            y = 2.0
            z = 3.0

        assert isinstance(MyVec(), Vec3Like)

    def test_duck_pose(self):
        class MyVec:
            x = 0.0
            y = 0.0
            z = 0.0

        class MyQuat:
            w = 1.0
            x = 0.0
            y = 0.0
            z = 0.0

        class MyPose:
            position = MyVec()
            orientation = MyQuat()

        assert isinstance(MyPose(), PoseLike)

    def test_duck_world_object(self):
        class MyVec:
            x = 0.0
            y = 0.0
            z = 0.0

        class MyQuat:
            w = 1.0
            x = 0.0
            y = 0.0
            z = 0.0

        class MyPose:
            position = MyVec()
            orientation = MyQuat()

        class MyObj:
            obj_id = "obj1"
            obj_type = "box"
            name = ""
            pose = MyPose()
            scene_id = None
            state = "present"
            occlusion_status = "visible"
            confidence = 1.0
            last_seen_sec = 0.0
            semantic_tags = []

            def to_dict(self):
                return {"obj_id": self.obj_id}

        assert isinstance(MyObj(), WorldObjectLike)


class TestV10IntegrationPattern:
    """Demonstrate the recommended v1.0 integration pattern.

    v1.0 MemoryInterface should use these protocols as type annotations
    instead of `Any`, enabling IDE autocompletion and mypy checking
    when powermem is installed.
    """

    def test_conditional_import_pattern(self):
        """v1.0 should use this import pattern to handle optional powermem."""
        try:
            from powermem.embodied.protocols import (
                WorldObjectLike,
                PoseLike,
                Vec3Like,
                PermanenceReportLike,
            )
            has_powermem = True
        except ImportError:
            has_powermem = False
            WorldObjectLike = object  # type: ignore
            PoseLike = object  # type: ignore
            Vec3Like = object  # type: ignore
            PermanenceReportLike = object  # type: ignore

        assert has_powermem

        # Type-safe proxy method signature:
        # def add_world_object(self, obj: WorldObjectLike) -> Optional[str]:
        #     if self._embodied is None:
        #         return None
        #     return self._embodied.add_world_object(obj)

        wo = WorldObject(
            obj_id="test", pose=Pose(), scene_id="s",
        )
        assert isinstance(wo, WorldObjectLike)
