"""
Tests for physical constraint memory, action experience, and predictive state persistence.

Uses SQLite in-memory for DB-level tests, direct object tests for MemoryAtom factories.
"""

import sqlite3

import pytest

from typing import Any, Dict, List, Optional, Tuple

from powermem.embodied.memory_atom import MemoryAtom
from powermem.embodied.physical_model import PhysicalConstraint
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import IntervalRelation, MemoryAction, TemporalInterval, Vec3


# ---------------------------------------------------------------------------
# MemoryAtom factories
# ---------------------------------------------------------------------------

class TestMemoryAtomConstraint:
    def test_from_constraint_with_region(self):
        c = PhysicalConstraint(
            constraint_type="collision_free",
            description="safe zone near base",
            region_center=(1.0, 2.0, 0.5),
            region_radius=0.3,
        )
        atom = MemoryAtom.from_constraint("safe zone", c)
        assert atom.spatial == Vec3(1.0, 2.0, 0.5)
        assert atom.embodied_meta["constraint"]["constraint_type"] == "collision_free"
        assert atom.action == MemoryAction.PREDICT

    def test_from_constraint_without_region(self):
        c = PhysicalConstraint(constraint_type="graspable", description="can grasp")
        atom = MemoryAtom.from_constraint("graspable", c)
        assert atom.spatial is None
        assert atom.embodied_meta["constraint"]["constraint_type"] == "graspable"

    def test_roundtrip_metadata(self):
        c = PhysicalConstraint(
            constraint_type="reachability",
            params={"joint_idx": 2, "max_reach": 1.2},
            region_center=(0, 0, 0),
            region_radius=0.5,
        )
        atom = MemoryAtom.from_constraint("reach", c)
        meta = atom.to_metadata()
        restored = MemoryAtom.from_metadata(atom.content, meta)
        assert restored.embodied_meta["constraint"]["constraint_type"] == "reachability"
        assert restored.embodied_meta["constraint"]["params"]["max_reach"] == 1.2


class TestMemoryAtomAction:
    def test_from_action_with_outcome(self):
        atom = MemoryAtom.from_action(
            content="move arm to target",
            action_type=MemoryAction.ACT,
            spatial=Vec3(1, 0, 0),
            outcome_status="collision",
        )
        assert atom.action == MemoryAction.ACT
        assert atom.embodied_meta["outcome_status"] == "collision"
        assert atom.spatial == Vec3(1, 0, 0)

    def test_from_action_without_outcome(self):
        atom = MemoryAtom.from_action(
            content="observe scene",
            action_type=MemoryAction.OBSERVE,
        )
        assert "outcome_status" not in atom.embodied_meta


# ---------------------------------------------------------------------------
# Predictive state persistence (direct DB tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    initialize_embodied_schema(conn)
    yield conn
    conn.close()


class TestPredictiveStatePersistence:
    def test_save_and_load_state(self, db):
        from powermem.embodied.embodied_memory import EmbodiedMemory
        # We can't easily instantiate EmbodiedMemory (needs PowerMem Memory),
        # so we test the SQL methods directly via a minimal wrapper.

        class _TestWrapper:
            def __init__(self, conn):
                self.db_conn = conn

            save_surprisal_state = EmbodiedMemory.save_surprisal_state
            get_surprisal_state = EmbodiedMemory.get_surprisal_state
            reset_surprisal_state = EmbodiedMemory.reset_surprisal_state

        w = _TestWrapper(db)
        w.save_surprisal_state("rgb_cam", {"count": 100, "mean": 0.5, "m2": 25.0})

        loaded = w.get_surprisal_state("rgb_cam")
        assert loaded is not None
        assert loaded["count"] == 100
        assert loaded["mean"] == pytest.approx(0.5)
        # m2 = std^2 * count = sqrt(25/100)^2 * 100 = 25
        assert loaded["m2"] == pytest.approx(25.0)

    def test_load_missing(self, db):
        from powermem.embodied.embodied_memory import EmbodiedMemory

        class _TestWrapper:
            def __init__(self, conn):
                self.db_conn = conn
            get_surprisal_state = EmbodiedMemory.get_surprisal_state

        w = _TestWrapper(db)
        assert w.get_surprisal_state("missing") is None

    def test_reset_specific(self, db):
        from powermem.embodied.embodied_memory import EmbodiedMemory

        class _TestWrapper:
            def __init__(self, conn):
                self.db_conn = conn
            save_surprisal_state = EmbodiedMemory.save_surprisal_state
            reset_surprisal_state = EmbodiedMemory.reset_surprisal_state
            get_surprisal_state = EmbodiedMemory.get_surprisal_state

        w = _TestWrapper(db)
        w.save_surprisal_state("a", {"count": 10, "mean": 0.1, "m2": 1.0})
        w.save_surprisal_state("b", {"count": 20, "mean": 0.2, "m2": 2.0})
        w.reset_surprisal_state("a")
        assert w.get_surprisal_state("a") is None
        assert w.get_surprisal_state("b") is not None

    def test_reset_all(self, db):
        from powermem.embodied.embodied_memory import EmbodiedMemory

        class _TestWrapper:
            def __init__(self, conn):
                self.db_conn = conn
            save_surprisal_state = EmbodiedMemory.save_surprisal_state
            reset_surprisal_state = EmbodiedMemory.reset_surprisal_state
            get_surprisal_state = EmbodiedMemory.get_surprisal_state

        w = _TestWrapper(db)
        w.save_surprisal_state("a", {"count": 10, "mean": 0.1, "m2": 1.0})
        w.reset_surprisal_state()
        assert w.get_surprisal_state("a") is None

    def test_update_existing(self, db):
        from powermem.embodied.embodied_memory import EmbodiedMemory

        class _TestWrapper:
            def __init__(self, conn):
                self.db_conn = conn
            save_surprisal_state = EmbodiedMemory.save_surprisal_state
            get_surprisal_state = EmbodiedMemory.get_surprisal_state

        w = _TestWrapper(db)
        w.save_surprisal_state("same", {"count": 10, "mean": 0.1, "m2": 1.0})
        w.save_surprisal_state("same", {"count": 20, "mean": 0.2, "m2": 4.0})
        loaded = w.get_surprisal_state("same")
        assert loaded["count"] == 20
        assert loaded["mean"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Causal edges (action -> outcome)
# ---------------------------------------------------------------------------

class TestCausalActionOutcome:
    def test_outcome_atom_has_action_parent(self):
        action = MemoryAtom.from_action(
            content="move to A",
            action_type=MemoryAction.ACT,
            spatial=Vec3(0, 0, 0),
        )
        outcome = MemoryAtom.from_action(
            content="collision detected",
            action_type=MemoryAction.CORRECT,
            outcome_status="collision",
            spatial=Vec3(0, 0, 0),
        )
        outcome.causal_parents = [42]  # simulate action_id
        assert outcome.causal_parents == [42]
        meta = outcome.to_metadata()
        assert meta["causal_parents"] == [42]


# ---------------------------------------------------------------------------
# World object memory
# ---------------------------------------------------------------------------

class TestMemoryAtomWorldObject:
    def test_from_world_object_with_pose_dict(self):
        obj = {
            "name": "table",
            "type": "box",
            "pose": {"position": {"x": 1.0, "y": 2.0, "z": 0.0}},
            "size": [1.0, 0.5, 0.8],
        }
        atom = MemoryAtom.from_world_object("table in room", obj)
        assert atom.spatial == Vec3(1.0, 2.0, 0.0)
        assert atom.embodied_meta["physical_type"] == "world_object"
        assert atom.embodied_meta["world_object"]["name"] == "table"
        assert atom.action == MemoryAction.OBSERVE

    def test_from_world_object_with_pose_list(self):
        obj = {
            "name": "ball",
            "type": "sphere",
            "pose": [3.0, 4.0, 0.5],
            "radius": 0.1,
        }
        atom = MemoryAtom.from_world_object("red ball", obj)
        assert atom.spatial == Vec3(3.0, 4.0, 0.5)

    def test_from_world_object_without_pose(self):
        obj = {"name": "light", "type": "point_light"}
        atom = MemoryAtom.from_world_object("ceiling light", obj)
        assert atom.spatial is None
        assert atom.embodied_meta["world_object"]["type"] == "point_light"

    def test_world_object_roundtrip(self):
        obj = {"name": "shelf", "type": "mesh", "pose": {"position": {"x": 0.5, "y": 0, "z": 1.0}}}
        atom = MemoryAtom.from_world_object("storage shelf", obj)
        meta = atom.to_metadata()
        restored = MemoryAtom.from_metadata(atom.content, meta)
        assert restored.embodied_meta["world_object"]["name"] == "shelf"
        assert restored.embodied_meta["physical_type"] == "world_object"
        assert restored.spatial == Vec3(0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Trajectory memory
# ---------------------------------------------------------------------------

class TestMemoryAtomTrajectory:
    def test_from_trajectory(self):
        waypoints = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(1, 0, 0), 1.0),
            (Vec3(2, 0, 0), 2.0),
        ]
        atom = MemoryAtom.from_trajectory("move along x", waypoints)
        assert atom.action == MemoryAction.ACT
        assert atom.spatial == Vec3(1, 0, 0)  # midpoint
        assert atom.temporal.start_sec == 0.0
        assert atom.temporal.end_sec == 2.0
        assert atom.embodied_meta["trajectory"]["waypoint_count"] == 3
        assert atom.embodied_meta["trajectory"]["duration"] == 2.0
        # 签名预计算
        assert "signature" in atom.embodied_meta["trajectory"]
        assert len(atom.embodied_meta["trajectory"]["signature"]) == 8
        assert atom.embodied_meta["physical_type"] == "trajectory"

    def test_empty_trajectory(self):
        atom = MemoryAtom.from_trajectory("no movement", [])
        assert atom.spatial is None
        assert atom.temporal is None
        assert atom.embodied_meta["trajectory"]["waypoints"] == []
        assert atom.embodied_meta["trajectory"]["signature"] == [0.0] * 8

    def test_trajectory_roundtrip(self):
        waypoints = [(Vec3(0, 1, 2), 0.5), (Vec3(3, 4, 5), 1.5)]
        atom = MemoryAtom.from_trajectory("test path", waypoints)
        meta = atom.to_metadata()
        restored = MemoryAtom.from_metadata(atom.content, meta)
        traj = restored.embodied_meta["trajectory"]
        assert traj["waypoint_count"] == 2
        assert traj["waypoints"][0]["position"] == {"x": 0, "y": 1, "z": 2}
        assert restored.temporal.start_sec == 0.5
        # 签名在 roundtrip 后保持
        assert "signature" in traj
        assert len(traj["signature"]) == 8


# ---------------------------------------------------------------------------
# EmbodiedMemory world object / trajectory integration (mocked PowerMem)
# ---------------------------------------------------------------------------

class _MockStorage:
    """最小化 PowerMem storage 模拟，支持 add_memory / get_memory"""

    def __init__(self):
        self._data: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def add_memory(self, payload: Dict[str, Any]) -> int:
        mid = self._next_id
        self._next_id += 1
        self._data[mid] = {
            "id": mid,
            "data": payload["content"],
            "content": payload["content"],
            "metadata": payload.get("metadata", {}),
            "user_id": payload.get("user_id", ""),
            "agent_id": payload.get("agent_id", ""),
            "run_id": payload.get("run_id", ""),
            "created_at": "2024-01-01T00:00:00",
        }
        return mid

    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        return self._data.get(memory_id)

    def delete(self, memory_id: int) -> bool:
        return self._data.pop(memory_id, None) is not None


@pytest.fixture
def mock_embodied_memory():
    """提供带有 mock storage 的 EmbodiedMemory"""
    import sqlite3
    from unittest.mock import MagicMock
    from powermem.embodied.embodied_memory import EmbodiedMemory

    conn = sqlite3.connect(":memory:")
    initialize_embodied_schema(conn)

    mock_memory = MagicMock()
    mock_storage = _MockStorage()
    mock_memory.storage = mock_storage
    mock_memory.agent_id = "test_agent"

    em = EmbodiedMemory(memory=mock_memory, db_conn=conn, enable_plugin=False)
    yield em
    conn.close()


class TestWorldObjectMemoryIntegration:
    def test_add_and_search_world_objects(self, mock_embodied_memory):
        em = mock_embodied_memory
        objs = [
            {"name": "table", "type": "box", "pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}}},
            {"name": "chair", "type": "box", "pose": {"position": {"x": 1.2, "y": 0.0, "z": 0.0}}},
            {"name": "lamp", "type": "mesh", "pose": {"position": {"x": 5.0, "y": 5.0, "z": 0.0}}},
        ]
        from powermem.embodied.parsers.base import ParseResult
        result = ParseResult()
        result.world_objects = objs

        ids = em.add_world_objects(result)
        assert len(ids) == 3

        # 搜索 table 附近
        found = em.search_world_objects(Vec3(1.0, 0.0, 0.0), radius=0.5)
        names = [a.name for a in found]
        assert "table" in names
        assert "chair" in names
        assert "lamp" not in names

    def test_search_world_objects_by_type(self, mock_embodied_memory):
        em = mock_embodied_memory
        objs = [
            {"name": "table", "type": "box", "pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}}},
            {"name": "ball", "type": "sphere", "pose": {"position": {"x": 1.1, "y": 0.0, "z": 0.0}}},
        ]
        from powermem.embodied.parsers.base import ParseResult
        result = ParseResult()
        result.world_objects = objs
        em.add_world_objects(result)

        spheres = em.search_world_objects(Vec3(1.0, 0.0, 0.0), radius=1.0, obj_type="sphere")
        assert len(spheres) == 1
        assert spheres[0].name == "ball"


class TestTrajectoryMemoryIntegration:
    def test_record_and_search_trajectory(self, mock_embodied_memory):
        em = mock_embodied_memory
        waypoints = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(1, 0, 0), 1.0),
            (Vec3(2, 0, 0), 2.0),
        ]
        mid = em.record_trajectory("grasp trajectory", waypoints)
        assert mid is not None

        # 搜索轨迹中点附近
        found = em.search_trajectory_near(Vec3(1.0, 0.0, 0.0), radius=0.5)
        assert len(found) == 1
        assert found[0].content == "grasp trajectory"

    def test_search_trajectory_miss(self, mock_embodied_memory):
        em = mock_embodied_memory
        waypoints = [
            (Vec3(10, 10, 0), 0.0),
            (Vec3(11, 10, 0), 1.0),
        ]
        em.record_trajectory("far away", waypoints)

        # 查询远离轨迹的点
        found = em.search_trajectory_near(Vec3(0, 0, 0), radius=1.0)
        assert len(found) == 0

    def test_search_trajectory_temporal(self, mock_embodied_memory):
        em = mock_embodied_memory
        waypoints = [
            (Vec3(0, 0, 0), 10.0),
            (Vec3(1, 0, 0), 11.0),
            (Vec3(2, 0, 0), 12.0),
        ]
        em.record_trajectory("timed trajectory", waypoints)

        # 时间命中（默认任意重叠）
        interval = TemporalInterval(start_sec=9.0, end_sec=11.0)
        found = em.search_trajectory_temporal(interval)
        assert len(found) == 1

        # 时间未命中
        interval2 = TemporalInterval(start_sec=0.0, end_sec=5.0)
        found2 = em.search_trajectory_temporal(interval2)
        assert len(found2) == 0

    def test_search_trajectory_spatiotemporal(self, mock_embodied_memory):
        em = mock_embodied_memory
        waypoints = [
            (Vec3(0, 0, 0), 10.0),
            (Vec3(1, 0, 0), 11.0),
        ]
        em.record_trajectory("spatiotemporal test", waypoints)

        # 空间命中 + 时间命中
        interval = TemporalInterval(start_sec=9.0, end_sec=11.0)
        found = em.search_trajectory_near(Vec3(0.5, 0.0, 0.0), radius=1.0, temporal_interval=interval)
        assert len(found) == 1

        # 空间命中 + 时间未命中
        interval2 = TemporalInterval(start_sec=0.0, end_sec=5.0)
        found2 = em.search_trajectory_near(Vec3(0.5, 0.0, 0.0), radius=1.0, temporal_interval=interval2)
        assert len(found2) == 0
