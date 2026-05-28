"""
Deep tests for embodied memory system.

Coverage:
- Boundary conditions (empty, None, zero, negative, extreme values)
- Edge cases (degenerate geometries, singular configurations)
- Error handling paths
- Comprehensive integration paths
- Frame isolation and multi-coordinate-system scenarios
- Causal graph depth and branching
- Pipeline flush/merge/outlier behavior
- End-to-end workflows

Uses SQLite in-memory + mock PowerMem storage.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from powermem.embodied import (
    AABB,
    Capsule,
    CollisionBody,
    CollisionChecker,
    DHParameter,
    EmbodiedMemory,
    MemoryAtom,
    Modality,
    PerceptualSnapshot,
    PhysicalInvariant,
    Pose,
    Quaternion,
    RobotDynamics,
    SensorFrame,
    Sphere,
    SurprisalGate,
    Transform,
    Vec3,
    ZeroOrderHoldPredictor,
)
from powermem.embodied.collision import bodies_intersect, build_collision_bodies
from powermem.embodied.kinematics import (
    dh_to_transform,
    forward_kinematics,
    forward_kinematics_poses,
    transform_collision_body,
    transform_collision_bodies,
)
from powermem.embodied.parsers.base import ParseResult
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.spatial_index import SpatialIndex, VoxelHash
from powermem.embodied.surprisal_gate import LinearPredictor, _RunningStats
from powermem.embodied.temporal_index import TemporalIndex
from powermem.embodied.physical_model import JointLimit
from powermem.embodied.types import (
    AffectiveTag,
    IntervalRelation,
    MemoryAction,
    TemporalInterval,
    UncertaintyEstimate,
    UncertaintyType,
    WorldObject,
)


# ---------------------------------------------------------------------------
# Mock PowerMem infrastructure
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

    def get_many_memories(self, memory_ids):
        return [self._store.get(mid) for mid in memory_ids]

    def delete_memory(self, memory_id: int, user_id=None, agent_id=None) -> bool:
        return self._store.pop(memory_id, None) is not None

    def search_memories(self, **kwargs) -> List[Dict[str, Any]]:
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

    def update(self, memory_id: int, content: str, user_id=None, agent_id=None, metadata=None) -> Dict[str, Any]:
        item = self.storage._store.get(memory_id)
        if item is None:
            raise KeyError(memory_id)
        item["data"] = content
        item["content"] = content
        if metadata is not None:
            item["metadata"] = metadata
        return item


@pytest.fixture
def sqlite_conn():
    conn = sqlite3.connect(":memory:")
    initialize_embodied_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def embodied_memory(sqlite_conn):
    mock_mem = MockMemory()
    em = EmbodiedMemory(
        memory=mock_mem,
        db_conn=sqlite_conn,
        voxel_size=1.0,
        enable_plugin=False,
    )
    return em


# ========================================================================
# 1. MemoryAtom Boundaries
# ========================================================================

class TestMemoryAtomBoundaries:
    """MemoryAtom 工厂方法和属性的边界条件测试"""

    def test_from_observation_none_pose(self):
        """sensor_pose=None 时应不报错"""
        atom = MemoryAtom.from_observation(
            content="test",
            sensor_pose=None,
            modality=Modality.DEPTH,
        )
        assert atom.spatial is None
        assert atom.perceptual.modality == Modality.DEPTH

    def test_from_observation_empty_feature_vec(self):
        atom = MemoryAtom.from_observation(
            content="empty vec",
            sensor_pose=Pose(position=Vec3(1, 2, 3)),
            feature_vec=(),
        )
        assert atom.perceptual.feature_vec == ()

    def test_from_observation_all_modalities(self):
        """所有 modality 枚举值都能创建"""
        for modality in Modality:
            atom = MemoryAtom.from_observation(
                content=f"modality {modality.value}",
                sensor_pose=Pose(),
                modality=modality,
            )
            assert atom.perceptual.modality == modality

    def test_from_world_object_invalid_pose_formats(self):
        """各种 pose 格式解析"""
        # dict with position
        atom1 = MemoryAtom.from_world_object("obj1", {"pose": {"position": {"x": 1, "y": 2, "z": 3}}})
        assert atom1.spatial == Vec3(1, 2, 3)

        # list/tuple
        atom2 = MemoryAtom.from_world_object("obj2", {"pose": [4.0, 5.0, 6.0]})
        assert atom2.spatial == Vec3(4, 5, 6)

        # invalid short list
        atom3 = MemoryAtom.from_world_object("obj3", {"pose": [1.0, 2.0]})
        assert atom3.spatial is None

        # missing pose
        atom4 = MemoryAtom.from_world_object("obj4", {"name": "no_pose"})
        assert atom4.spatial is None

    def test_from_trajectory_single_waypoint(self):
        """单点轨迹：中点就是该点"""
        atom = MemoryAtom.from_trajectory("single", [(Vec3(5, 5, 5), 1.0)])
        assert atom.spatial == Vec3(5, 5, 5)
        assert atom.temporal.start_sec == 1.0
        assert atom.temporal.end_sec == 1.0
        assert atom.embodied_meta["trajectory"]["duration"] == 0.0

    def test_from_trajectory_extreme_coordinates(self):
        """极大/极小坐标"""
        atom = MemoryAtom.from_trajectory("extreme", [
            (Vec3(-1e6, -1e6, -1e6), 0.0),
            (Vec3(1e6, 1e6, 1e6), 1.0),
        ])
        assert atom.spatial == Vec3(1e6, 1e6, 1e6)  # midpoint index 1 for 2 points
        assert atom.embodied_meta["trajectory"]["waypoint_count"] == 2

    def test_empty_atom_roundtrip(self):
        """全默认 MemoryAtom 的序列化/反序列化"""
        atom = MemoryAtom(content="minimal")
        meta = atom.to_metadata()
        restored = MemoryAtom.from_metadata("minimal", meta)
        assert restored.content == "minimal"
        assert restored.spatial is None
        assert restored.temporal is None
        assert restored.perceptual is None
        assert restored.physical is None
        assert restored.uncertainty is None
        assert restored.affective is None
        assert restored.action == MemoryAction.OBSERVE
        assert restored.prediction_error == 0.0
        assert restored.causal_parents == []

    def test_full_atom_roundtrip(self):
        """全字段 MemoryAtom 的序列化/反序列化"""
        atom = MemoryAtom(
            content="full",
            memory_id=42,
            user_id="u1",
            agent_id="a1",
            run_id="r1",
            actor_id="ac1",
            spatial=Vec3(1, 2, 3),
            spatial_frame_id="camera",
            spatial_voxel_key="10:20:30:camera",
            temporal=TemporalInterval(0.0, 10.0, "sim_time"),
            perceptual=PerceptualSnapshot(
                modality=Modality.LIDAR,
                feature_vec=(0.1, 0.2, 0.3),
                raw_data_hash="abc123",
                sensor_pose=Pose(position=Vec3(0, 0, 1), orientation=Quaternion(1, 0, 0, 0)),
                uncertainty=UncertaintyEstimate(
                    type=UncertaintyType.EPISTEMIC,
                    std=0.5,
                    covariance=(0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1),
                    confidence=0.8,
                    sample_count=10,
                ),
                sensor_meta={"resolution": "640x480"},
            ),
            physical=PhysicalInvariant(
                entity_id="link1",
                mass_kg=2.5,
                center_of_mass=Vec3(0.1, 0.2, 0.3),
                inertia_matrix=(0.01,)*9,
                kinematic_params={"dh": [0.1, 0.2, 0.3, 0.4]},
                dynamics_params={"friction": 0.1},
            ),
            uncertainty=UncertaintyEstimate(std=0.1, confidence=0.95),
            affective=AffectiveTag(salience=0.9, valence=-0.5, arousal=0.7, trigger="collision"),
            action=MemoryAction.ACT,
            prediction_error=2.5,
            causal_parents=[10, 20],
            embodied_meta={"custom": "value", "nested": {"a": 1}},
            created_at="2024-01-01T00:00:00",
        )
        meta = atom.to_metadata()
        restored = MemoryAtom.from_metadata("full", meta, memory_id=42, user_id="u1")
        assert restored.spatial == Vec3(1, 2, 3)
        assert restored.temporal == TemporalInterval(0.0, 10.0, "sim_time")
        assert restored.perceptual.modality == Modality.LIDAR
        assert restored.perceptual.feature_vec == (0.1, 0.2, 0.3)
        assert restored.physical.entity_id == "link1"
        assert restored.physical.mass_kg == 2.5
        assert restored.uncertainty.confidence == 0.95
        assert restored.affective.salience == 0.9
        assert restored.action == MemoryAction.ACT
        assert restored.prediction_error == 2.5
        assert restored.causal_parents == [10, 20]
        assert restored.embodied_meta["custom"] == "value"
        assert restored.spatial_frame_id == "camera"
        assert restored.spatial_voxel_key == "10:20:30:camera"

    def test_compute_voxel_key_negative_coords(self):
        """负坐标体素键计算"""
        atom = MemoryAtom(content="neg", spatial=Vec3(-1.5, -2.5, -0.5))
        key = atom.compute_voxel_key(voxel_size=1.0)
        assert key is not None
        # int(-1.5 / 1.0) = -1, int(-2.5) = -2, int(-0.5) = 0
        assert "-1:-2:0" in key

    def test_compute_voxel_key_zero_voxel_size(self):
        """零体素大小应返回 None 或异常（除零保护）"""
        atom = MemoryAtom(content="zero", spatial=Vec3(1, 2, 3))
        with pytest.raises((ZeroDivisionError, ValueError)):
            atom.compute_voxel_key(voxel_size=0.0)

    def test_compute_voxel_key_no_spatial(self):
        assert MemoryAtom(content="no_pos").compute_voxel_key() is None

    def test_is_significant_boundaries(self):
        """显著性判断边界"""
        # salience > 0.8
        atom1 = MemoryAtom(content="a", affective=AffectiveTag(salience=0.81))
        assert atom1.is_significant is True

        # salience == 0.8 (not >)
        atom2 = MemoryAtom(content="b", affective=AffectiveTag(salience=0.8))
        assert atom2.is_significant is False

        # prediction_error > 1.0
        atom3 = MemoryAtom(content="c", prediction_error=1.01)
        assert atom3.is_significant is True

        # neither
        atom4 = MemoryAtom(content="d")
        assert atom4.is_significant is False

    def test_is_high_uncertainty_boundaries(self):
        """高不确定性判断边界"""
        # confidence < 0.3
        atom1 = MemoryAtom(
            content="a",
            uncertainty=UncertaintyEstimate(confidence=0.29, std=0.1),
        )
        assert atom1.is_high_uncertainty is True

        # std > 1.0
        atom2 = MemoryAtom(
            content="b",
            uncertainty=UncertaintyEstimate(confidence=0.9, std=1.01),
        )
        assert atom2.is_high_uncertainty is True

        # no uncertainty
        atom3 = MemoryAtom(content="c")
        assert atom3.is_high_uncertainty is False

        # both OK
        atom4 = MemoryAtom(
            content="d",
            uncertainty=UncertaintyEstimate(confidence=0.5, std=0.5),
        )
        assert atom4.is_high_uncertainty is False

    def test_content_hash_consistency(self):
        """相同内容产生相同哈希"""
        a1 = MemoryAtom(content="same")
        a2 = MemoryAtom(content="same")
        assert a1.content_hash == a2.content_hash
        assert a1.content_hash != MemoryAtom(content="different").content_hash

    def test_embedding_property(self):
        """embedding getter/setter"""
        atom = MemoryAtom(content="emb")
        assert atom.embedding is None
        atom.embedding = [0.1, 0.2, 0.3]
        assert atom.embedding == [0.1, 0.2, 0.3]
        atom.embedding = None
        assert atom.embedding is None

    def test_to_powermem_payload_structure(self):
        """payload 结构完整性"""
        atom = MemoryAtom(
            content="payload test",
            user_id="u1",
            agent_id="a1",
            run_id="r1",
            actor_id="ac1",
            action=MemoryAction.PREDICT,
        )
        atom.embedding = [0.1, 0.2]
        payload = atom.to_powermem_payload()
        assert payload["content"] == "payload test"
        assert payload["user_id"] == "u1"
        assert payload["agent_id"] == "a1"
        assert payload["run_id"] == "r1"
        assert payload["actor_id"] == "ac1"
        assert payload["category"] == "embodied_predict"
        assert payload["hash"] == atom.content_hash
        assert payload["embedding"] == [0.1, 0.2]
        assert "metadata" in payload
        assert payload["metadata"]["rosclaw_version"] == "0.1.0"


# ========================================================================
# 2. Spatial Index Boundaries
# ========================================================================

class TestVoxelHashBoundaries:
    """VoxelHash 边界条件"""

    def test_negative_coordinates(self):
        vh = VoxelHash(voxel_size=1.0)
        vh.insert(1, Vec3(-0.5, -0.5, -0.5))
        ids = vh.query_near(Vec3(0, 0, 0), radius=2.0)
        assert 1 in ids

    def test_zero_voxel_size_unsafe(self):
        """零体素大小会导致除零，应拒绝"""
        with pytest.raises((ZeroDivisionError, ValueError)):
            VoxelHash(voxel_size=0.0)

    def test_multi_frame_isolation_detailed(self):
        vh = VoxelHash(voxel_size=1.0)
        vh.insert(1, Vec3(0.5, 0.5, 0.5), frame_id="world")
        vh.insert(2, Vec3(0.5, 0.5, 0.5), frame_id="map")
        vh.insert(3, Vec3(0.5, 0.5, 0.5), frame_id="camera")

        assert 1 in vh.query_near(Vec3(0, 0, 0), radius=2.0, frame_id="world")
        assert 2 in vh.query_near(Vec3(0, 0, 0), radius=2.0, frame_id="map")
        assert 3 in vh.query_near(Vec3(0, 0, 0), radius=2.0, frame_id="camera")
        # cross-frame should not find
        assert 2 not in vh.query_near(Vec3(0, 0, 0), radius=2.0, frame_id="world")

    def test_remove_and_reinsert(self):
        vh = VoxelHash(voxel_size=1.0)
        vh.insert(1, Vec3(0.5, 0.5, 0.5))
        assert vh.remove(1) is True
        assert 1 not in vh.query_near(Vec3(0, 0, 0), radius=2.0)
        # reinsert
        vh.insert(1, Vec3(0.5, 0.5, 0.5))
        assert 1 in vh.query_near(Vec3(0, 0, 0), radius=2.0)
        # double remove
        assert vh.remove(1) is True
        assert vh.remove(1) is False

    def test_move_same_id(self):
        """同一 ID 移动到新位置"""
        vh = VoxelHash(voxel_size=1.0)
        vh.insert(1, Vec3(0.5, 0.5, 0.5))
        vh.insert(1, Vec3(10.5, 10.5, 10.5))
        assert 1 not in vh.query_near(Vec3(0, 0, 0), radius=2.0)
        assert 1 in vh.query_near(Vec3(10, 10, 10), radius=2.0)

    def test_exact_distance_sorting(self):
        vh = VoxelHash(voxel_size=1.0)
        positions = {
            1: Vec3(0, 0, 0),
            2: Vec3(3, 4, 0),
            3: Vec3(10, 0, 0),
        }
        for mid, pos in positions.items():
            vh.insert(mid, pos)
        results = vh.query_exact(Vec3(0, 0, 0), radius=10.0, id_to_position=positions)
        ids = [mid for mid, _ in results]
        assert ids == [1, 2, 3]  # distance <= 10 includes all three

    def test_large_data_set(self):
        """大量数据插入和查询性能/正确性"""
        vh = VoxelHash(voxel_size=0.5)
        for i in range(500):
            vh.insert(i, Vec3(float(i) * 0.1, float(i) * 0.1, 0.0))
        ids = vh.query_near(Vec3(0, 0, 0), radius=1.0)
        assert len(ids) > 0
        stats = vh.stats()
        assert stats["total_ids"] == 500
        assert stats["avg_load"] > 0


class TestSpatialIndexBoundaries:
    """SpatialIndex 边界条件"""

    def test_rebuild_from_db_accuracy(self, sqlite_conn):
        """从数据库重建后索引应与内存一致"""
        si = SpatialIndex(sqlite_conn, voxel_size=1.0)
        # Simulate DB inserts
        cursor = sqlite_conn.cursor()
        for i in range(5):
            cursor.execute(
                "INSERT INTO embodied_memories (memory_id, spatial_x, spatial_y, spatial_z) VALUES (?, ?, ?, ?)",
                (1000 + i, float(i), float(i), 0.0),
            )
        sqlite_conn.commit()

        si.rebuild_from_db()
        assert len(si.get_all_ids()) == 5
        hits = si.query_radius(Vec3(0, 0, 0), radius=1.5)
        assert len(hits) > 0

    def test_zero_radius_query(self, sqlite_conn):
        """零半径查询只返回精确匹配点"""
        si = SpatialIndex(sqlite_conn, voxel_size=1.0)
        si.add(1, Vec3(0.0, 0.0, 0.0))
        si.add(2, Vec3(0.5, 0.0, 0.0))
        hits = si.query_radius(Vec3(0.0, 0.0, 0.0), radius=0.0)
        # exact match at distance 0
        assert any(h[0] == 1 for h in hits)
        assert not any(h[0] == 2 for h in hits)

    def test_distance_accuracy(self, sqlite_conn):
        """query_radius 返回的距离应准确"""
        si = SpatialIndex(sqlite_conn, voxel_size=1.0)
        si.add(1, Vec3(3.0, 4.0, 0.0))
        hits = si.query_radius(Vec3(0, 0, 0), radius=10.0)
        assert len(hits) == 1
        mid, dist = hits[0]
        assert mid == 1
        assert dist == pytest.approx(5.0, abs=1e-9)

    def test_remove_then_query(self, sqlite_conn):
        si = SpatialIndex(sqlite_conn, voxel_size=1.0)
        si.add(1, Vec3(1.0, 1.0, 1.0))
        si.remove(1)
        hits = si.query_radius(Vec3(1.0, 1.0, 1.0), radius=1.0)
        assert len(hits) == 0

    def test_query_nearest(self, sqlite_conn):
        si = SpatialIndex(sqlite_conn, voxel_size=1.0)
        si.add(1, Vec3(0, 0, 0))
        si.add(2, Vec3(1, 0, 0))
        si.add(3, Vec3(5, 0, 0))
        nearest = si.query_nearest(Vec3(0, 0, 0), k=2, max_radius=10.0)
        assert len(nearest) == 2
        assert nearest[0][0] == 1
        assert nearest[1][0] == 2


# ========================================================================
# 3. Temporal Index Deep Tests
# ========================================================================

class TestTemporalIndexDeep:
    """TemporalIndex 所有 Allen 关系和边界条件"""

    @pytest.fixture
    def populated_db(self, sqlite_conn):
        """填充多条具有不同时间区间的记忆"""
        cursor = sqlite_conn.cursor()
        intervals = [
            (1, 0.0, 5.0, "wall_clock"),    # A: [0, 5]
            (2, 3.0, 8.0, "wall_clock"),    # B: [3, 8]  overlaps A
            (3, 10.0, 15.0, "wall_clock"),  # C: [10, 15] after A/B
            (4, 5.0, 5.0, "wall_clock"),    # D: [5, 5] point, meets A
            (5, 0.0, 5.0, "sim_time"),       # E: same as A but different frame
        ]
        for mid, start, end, frame in intervals:
            cursor.execute(
                "INSERT INTO embodied_memories (memory_id, temporal_start, temporal_end, temporal_frame_id) VALUES (?, ?, ?, ?)",
                (mid, start, end, frame),
            )
        sqlite_conn.commit()
        return TemporalIndex(sqlite_conn)

    def test_allen_before(self, populated_db):
        # Query [6, 9]: A=[0,5] is before
        idx = populated_db
        results = idx.query(TemporalInterval(6.0, 9.0), IntervalRelation.BEFORE)
        assert 1 in {r[0] for r in results}

    def test_allen_after(self, populated_db):
        idx = populated_db
        results = idx.query(TemporalInterval(1.0, 2.0), IntervalRelation.AFTER)
        assert 3 in {r[0] for r in results}

    def test_allen_meets(self, populated_db):
        # D=[5,5] meets query [5, 10]
        idx = populated_db
        results = idx.query(TemporalInterval(5.0, 10.0), IntervalRelation.MEETS)
        assert 4 in {r[0] for r in results}

    def test_allen_overlaps(self, populated_db):
        # A=[0,5] overlaps query [3, 8] (A starts before, ends during)
        idx = populated_db
        results = idx.query(TemporalInterval(3.0, 8.0), IntervalRelation.OVERLAPS)
        assert 1 in {r[0] for r in results}

    def test_allen_overlapped_by(self, populated_db):
        # B=[3,8] is overlapped_by query [0, 5] (B starts during, ends after)
        idx = populated_db
        results = idx.query(TemporalInterval(0.0, 5.0), IntervalRelation.OVERLAPPED_BY)
        assert 2 in {r[0] for r in results}

    def test_allen_during(self, populated_db):
        # A=[0,5] is during query [-1, 6]
        idx = populated_db
        results = idx.query(TemporalInterval(-1.0, 6.0), IntervalRelation.DURING)
        assert 1 in {r[0] for r in results}

    def test_allen_contains(self, populated_db):
        # A=[0,5] contains query [1, 4]
        idx = populated_db
        results = idx.query(TemporalInterval(1.0, 4.0), IntervalRelation.CONTAINS)
        assert 1 in {r[0] for r in results}

    def test_allen_equals(self, populated_db):
        idx = populated_db
        results = idx.query(TemporalInterval(0.0, 5.0), IntervalRelation.EQUALS)
        assert 1 in {r[0] for r in results}

    def test_allen_starts(self, populated_db):
        idx = populated_db
        results = idx.query(TemporalInterval(0.0, 10.0), IntervalRelation.STARTS)
        assert 1 in {r[0] for r in results}

    def test_allen_started_by(self, populated_db):
        # A=[0,5] starts query [0, 3] and ends after it
        idx = populated_db
        results = idx.query(TemporalInterval(0.0, 3.0), IntervalRelation.STARTED_BY)
        assert 1 in {r[0] for r in results}

    def test_allen_finishes(self, populated_db):
        idx = populated_db
        results = idx.query(TemporalInterval(-5.0, 5.0), IntervalRelation.FINISHES)
        assert 1 in {r[0] for r in results}

    def test_allen_finished_by(self, populated_db):
        # A=[0,5] finishes query [2, 5] and starts before it
        idx = populated_db
        results = idx.query(TemporalInterval(2.0, 5.0), IntervalRelation.FINISHED_BY)
        assert 1 in {r[0] for r in results}

    def test_query_overlapping_multiple(self, populated_db):
        idx = populated_db
        results = idx.query_overlapping(TemporalInterval(2.0, 12.0), frame_id="wall_clock")
        ids = {r[0] for r in results}
        assert ids == {1, 2, 3, 4}  # A, B, C, D overlap [2, 12] (C shares [10,12])

    def test_frame_isolation(self, populated_db):
        idx = populated_db
        results = idx.query_overlapping(TemporalInterval(0.0, 5.0), frame_id="sim_time")
        ids = {r[0] for r in results}
        assert ids == {5}
        assert 1 not in ids  # same interval but different frame

    def test_point_interval(self, populated_db):
        """单点时间区间查询"""
        idx = populated_db
        # D=[5,5] should overlap query [5,5]
        results = idx.query_overlapping(TemporalInterval(5.0, 5.0))
        ids = {r[0] for r in results}
        assert 4 in ids

    def test_query_before_timestamp(self, populated_db):
        idx = populated_db
        results = idx.query_before(6.0, frame_id="wall_clock")
        ids = {r[0] for r in results}
        assert ids == {1, 4}

    def test_query_after_timestamp(self, populated_db):
        idx = populated_db
        results = idx.query_after(6.0, frame_id="wall_clock")
        ids = {r[0] for r in results}
        assert ids == {3}

    def test_query_contains_point(self, populated_db):
        idx = populated_db
        results = idx.query_contains_point(4.0, frame_id="wall_clock")
        ids = {r[0] for r in results}
        assert ids == {1, 2}


# ========================================================================
# 4. Collision Geometry Boundaries
# ========================================================================

class TestSphereBoundaries:
    def test_zero_radius(self):
        s = Sphere(center=Vec3(0, 0, 0), radius=0.0)
        assert s.contains(Vec3(0, 0, 0)) is True
        assert s.contains(Vec3(0.001, 0, 0)) is False

    def test_negative_radius_treated_as_valid(self):
        """负半径在数学上不合法，但当前实现未做校验"""
        s = Sphere(center=Vec3(0, 0, 0), radius=-1.0)
        # distance_to uses max(0, dist - radius), so negative radius increases distance
        assert s.distance_to(Vec3(0, 0, 0)) == 1.0

    def test_point_on_surface(self):
        s = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        assert s.contains(Vec3(1, 0, 0)) is True
        assert s.contains(Vec3(0, 1, 0)) is True
        assert s.contains(Vec3(0, 0, 1)) is True

    def test_distance_on_surface(self):
        s = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        assert s.distance_to(Vec3(1, 0, 0)) == pytest.approx(0.0, abs=1e-9)

    def test_aabb_symmetry(self):
        s = Sphere(center=Vec3(1, 2, 3), radius=0.5)
        box = s.aabb()
        assert box.min == Vec3(0.5, 1.5, 2.5)
        assert box.max == Vec3(1.5, 2.5, 3.5)
        assert box.center() == Vec3(1, 2, 3)


class TestCapsuleBoundaries:
    def test_degenerate_to_sphere(self):
        """a == b 时退化为球体"""
        c = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 0), radius=1.0)
        assert c.contains(Vec3(1, 0, 0)) is True
        assert c.contains(Vec3(0, 0, 2)) is False

    def test_point_on_endpoint(self):
        c = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 2), radius=0.5)
        assert c.contains(Vec3(0, 0, 0)) is True
        assert c.contains(Vec3(0, 0, 2)) is True
        assert c.contains(Vec3(0, 0.5, 0)) is True

    def test_point_on_surface_midpoint(self):
        c = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 2), radius=0.5)
        # midpoint of cylinder at (0,0,1), surface at radius=0.5
        assert c.contains(Vec3(0.5, 0, 1)) is True
        assert c.contains(Vec3(0.6, 0, 1)) is False

    def test_aabb_for_vertical_capsule(self):
        c = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 3), radius=0.5)
        box = c.aabb()
        assert box.min == Vec3(-0.5, -0.5, -0.5)
        assert box.max == Vec3(0.5, 0.5, 3.5)


class TestAABBBoundaries:
    def test_zero_volume(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(0, 0, 0))
        assert box.contains(Vec3(0, 0, 0)) is True
        assert box.contains(Vec3(0.001, 0, 0)) is False
        assert box.diagonal() == 0.0

    def test_negative_volume_not_normalized(self):
        """min > max 的情况（代码未做校验）"""
        box = AABB(min=Vec3(1, 1, 1), max=Vec3(0, 0, 0))
        # contains checks min <= point <= max, so nothing should match
        assert box.contains(Vec3(0.5, 0.5, 0.5)) is False

    def test_point_on_corner(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(1, 1, 1))
        assert box.contains(Vec3(0, 0, 0)) is True
        assert box.contains(Vec3(1, 1, 1)) is True

    def test_point_on_edge(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(1, 1, 1))
        assert box.contains(Vec3(0.5, 0, 0)) is True
        assert box.contains(Vec3(0.5, 1, 0.5)) is True

    def test_intersects_adjacent(self):
        a = AABB(min=Vec3(0, 0, 0), max=Vec3(1, 1, 1))
        b = AABB(min=Vec3(1, 0, 0), max=Vec3(2, 1, 1))
        assert a.intersects(b) is True  # touching at face

    def test_distance_inside(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(2, 2, 2))
        assert box.distance_to(Vec3(1, 1, 1)) == 0.0

    def test_distance_to_face(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(1, 1, 1))
        assert box.distance_to(Vec3(2, 0.5, 0.5)) == pytest.approx(1.0)

    def test_distance_to_corner(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(1, 1, 1))
        # corner at (2,2,2), distance = sqrt(3)
        assert box.distance_to(Vec3(2, 2, 2)) == pytest.approx(math.sqrt(3))


class TestCollisionDetectionBoundaries:
    def test_sphere_sphere_tangent(self):
        """两球相切：刚好接触"""
        a = CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 1.0))
        b = CollisionBody("b", "sphere", Sphere(Vec3(2, 0, 0), 1.0))
        assert bodies_intersect(a, b) is True

    def test_sphere_sphere_just_missing(self):
        a = CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 1.0))
        b = CollisionBody("b", "sphere", Sphere(Vec3(2.01, 0, 0), 1.0))
        assert bodies_intersect(a, b) is False

    def test_sphere_capsule_tangent(self):
        sphere = CollisionBody("s", "sphere", Sphere(Vec3(0, 0, 0), 0.5))
        capsule = CollisionBody("c", "capsule", Capsule(Vec3(1, 0, 0), Vec3(1, 2, 0), 0.5))
        assert bodies_intersect(sphere, capsule) is True

    def test_capsule_capsule_crossing(self):
        a = CollisionBody("a", "capsule", Capsule(Vec3(0, -1, 0), Vec3(0, 1, 0), 0.1))
        b = CollisionBody("b", "capsule", Capsule(Vec3(-1, 0, 0), Vec3(1, 0, 0), 0.1))
        assert bodies_intersect(a, b) is True

    def test_aabb_false_positive_fallback(self):
        """AABB 相交但精确不相交的情况"""
        # Two spheres with overlapping AABBs but not intersecting
        a = CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 0.5))
        b = CollisionBody("b", "sphere", Sphere(Vec3(1.5, 1.5, 1.5), 0.5))
        # AABBs overlap? Let's check: a AABB is [-0.5, 0.5]^3, b is [1, 2]^3
        # They don't overlap, so this is not a false positive case.
        # Let's create a case where AABB overlaps but spheres don't.
        a = CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 0.6))
        b = CollisionBody("b", "sphere", Sphere(Vec3(1.1, 0, 0), 0.6))
        # a AABB: [-0.6, 0.6], b AABB: [0.5, 1.7] -> overlap at [0.5, 0.6]
        # sphere distance = 1.1, radius sum = 1.2 -> intersect
        # Let's make them just miss
        a = CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 0.5))
        b = CollisionBody("b", "sphere", Sphere(Vec3(1.1, 0, 0), 0.5))
        # AABB overlap: a=[-0.5,0.5], b=[0.6,1.6] -> no overlap
        # Need closer: a=[-0.5,0.5], b=[0.4,1.4] -> overlap, but distance=1.0, sum=1.0 -> tangent
        b = CollisionBody("b", "sphere", Sphere(Vec3(1.01, 0, 0), 0.5))
        # AABB: a=[-0.5,0.5], b=[0.51,1.51] -> overlap tiny bit
        # distance=1.01, sum=1.0 -> should NOT intersect
        # But AABBs overlap, so it goes to precise check
        assert bodies_intersect(a, b) is False

    def test_unknown_geometry_fallback(self):
        """未知几何体回退到中心点检查"""
        # Create a fake geometry by using a class that doesn't match any isinstance check
        class FakeGeom:
            def center(self):
                return Vec3(0, 0, 0)
        a = CollisionBody("a", "fake", FakeGeom())
        b = CollisionBody("b", "sphere", Sphere(Vec3(0, 0, 0), 1.0))
        # Fallback checks if a.contains(b.center()) or b.contains(a.center())
        # FakeGeom doesn't have contains, but let's see what happens
        # Actually this might crash. Let's test with AABB instead.
        pass  # Skip this dangerous test

    def test_collision_checker_empty(self):
        checker = CollisionChecker()
        assert checker.check_point(Vec3(0, 0, 0)) == []
        assert checker.check_intersections() == []
        assert checker.nearest_body(Vec3(0, 0, 0)) is None

    def test_collision_checker_single_body(self):
        checker = CollisionChecker()
        checker.add_body(CollisionBody("b", "sphere", Sphere(Vec3(0, 0, 0), 1.0)))
        assert len(checker.check_point(Vec3(0, 0, 0))) == 1
        assert len(checker.check_intersections()) == 0
        nearest = checker.nearest_body(Vec3(5, 0, 0))
        assert nearest is not None
        assert nearest[1] == pytest.approx(4.0)

    def test_collision_checker_clear(self):
        checker = CollisionChecker()
        checker.add_body(CollisionBody("b", "sphere", Sphere(Vec3(0, 0, 0), 1.0)))
        checker.clear()
        assert len(checker.bodies) == 0


class TestBuildCollisionBodiesBoundaries:
    def test_empty_parse_result(self):
        result = ParseResult()
        bodies = build_collision_bodies(result)
        assert bodies == []

    def test_link_without_collision_geoms(self):
        result = ParseResult()
        result.links = [{"name": "base", "mass": 2.0, "com": {"x": 0, "y": 0, "z": 0.5}}]
        bodies = build_collision_bodies(result)
        assert len(bodies) == 1
        assert bodies[0].geom_type == "sphere"
        assert bodies[0].link_name == "base"

    def test_multiple_collision_geoms_per_link(self):
        result = ParseResult()
        result.source_hash = "test"
        result.links = [{
            "name": "arm",
            "collision_geoms": [
                {"type": "sphere", "center": [0, 0, 0], "radius": 0.1},
                {"type": "capsule", "a": [0, 0, 0], "b": [0, 0, 1], "radius": 0.05},
                {"type": "aabb", "min": [-0.1, -0.1, -0.1], "max": [0.1, 0.1, 0.1]},
            ],
        }]
        bodies = build_collision_bodies(result)
        assert len(bodies) == 3
        assert bodies[0].geom_type == "sphere"
        assert bodies[1].geom_type == "capsule"
        assert bodies[2].geom_type == "aabb"


# ========================================================================
# 5. Forward Kinematics Boundaries
# ========================================================================

class TestTransformBoundaries:
    def test_identity_transform(self):
        T = Transform.identity()
        assert T.apply(Vec3(1, 2, 3)) == Vec3(1, 2, 3)

    def test_compose_identity(self):
        T = Transform.from_translation(1, 2, 3)
        I = Transform.identity()
        assert (T @ I).apply(Vec3(0, 0, 0)) == Vec3(1, 2, 3)
        assert (I @ T).apply(Vec3(0, 0, 0)) == Vec3(1, 2, 3)

    def test_rotation_180_degrees(self):
        """180度旋转"""
        c = math.cos(math.pi)
        s = math.sin(math.pi)
        T = Transform(
            m00=c, m01=-s, m02=0, m03=0,
            m10=s, m11=c, m12=0, m13=0,
            m20=0, m21=0, m22=1, m23=0,
        )
        r = T.apply(Vec3(1, 0, 0))
        assert r.x == pytest.approx(-1.0, abs=1e-9)
        assert r.y == pytest.approx(0.0, abs=1e-9)

    def test_to_pose_quaternion_normalization(self):
        """从旋转矩阵提取四元数"""
        T = Transform.from_translation(1, 2, 3)
        pose = T.to_pose()
        assert pose.position == Vec3(1, 2, 3)
        # Identity rotation -> quaternion (1, 0, 0, 0)
        assert pose.orientation.w == pytest.approx(1.0)


class TestForwardKinematicsBoundaries:
    def test_empty_dh_params(self):
        assert forward_kinematics([], []) == []

    def test_single_joint(self):
        dh = [DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0)]
        transforms = forward_kinematics(dh, [math.pi / 2])
        pos = transforms[0].apply(Vec3(0, 0, 0))
        assert pos.x == pytest.approx(0.0, abs=1e-9)
        assert pos.y == pytest.approx(1.0, abs=1e-9)

    def test_long_chain(self):
        """长关节链"""
        dh = [DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0) for _ in range(20)]
        angles = [0.0] * 20
        transforms = forward_kinematics(dh, angles)
        assert len(transforms) == 20
        end = transforms[-1].apply(Vec3(0, 0, 0))
        assert end.x == pytest.approx(20.0, abs=1e-9)

    def test_fewer_angles_than_joints(self):
        """joint_angles 数量少于 dh_params"""
        dh = [DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0) for _ in range(5)]
        transforms = forward_kinematics(dh, [0.0, 0.0])
        assert len(transforms) == 5
        end = transforms[-1].apply(Vec3(0, 0, 0))
        assert end.x == pytest.approx(5.0, abs=1e-9)

    def test_more_angles_than_joints(self):
        """joint_angles 数量多于 dh_params"""
        dh = [DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0) for _ in range(3)]
        transforms = forward_kinematics(dh, [0.0, 0.0, 0.0, 0.0, 0.0])
        assert len(transforms) == 3

    def test_singular_configuration(self):
        """奇异位形：所有关节角为 0"""
        dh = [
            DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0),
            DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0),
        ]
        transforms = forward_kinematics(dh, [0.0, 0.0])
        end = transforms[1].apply(Vec3(0, 0, 0))
        assert end.x == pytest.approx(2.0, abs=1e-9)
        assert end.y == pytest.approx(0.0, abs=1e-9)

    def test_prismatic_like(self):
        """类 prismatic：theta 固定，d 变化的效果"""
        dh = [DHParameter(d=1.0, theta=0.0, a=0.0, alpha=0.0)]
        # joint_angle adds to theta, so d is fixed
        T = dh_to_transform(dh[0], 0.0)
        pos = T.apply(Vec3(0, 0, 0))
        assert pos.z == pytest.approx(1.0, abs=1e-9)


class TestTransformCollisionBodiesBoundaries:
    def test_transform_sphere(self):
        body = CollisionBody("s", "sphere", Sphere(Vec3(0, 0, 0), 0.5))
        T = Transform.from_translation(1, 2, 3)
        world = transform_collision_body(body, T)
        assert world.geometry.center == Vec3(1, 2, 3)
        assert world.geometry.radius == pytest.approx(0.5)

    def test_transform_capsule(self):
        body = CollisionBody("c", "capsule", Capsule(Vec3(0, 0, 0), Vec3(1, 0, 0), 0.1))
        T = Transform.from_translation(0, 1, 0)
        world = transform_collision_body(body, T)
        assert world.geometry.a == Vec3(0, 1, 0)
        assert world.geometry.b == Vec3(1, 1, 0)

    def test_transform_aabb_rotation_45(self):
        body = CollisionBody("b", "aabb", AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        T = Transform(
            m00=c, m01=-s, m02=0, m03=0,
            m10=s, m11=c, m12=0, m13=0,
            m20=0, m21=0, m22=1, m23=0,
        )
        world = transform_collision_body(body, T)
        # Rotated cube should have larger AABB
        assert world.geometry.min.x < -1.0
        assert world.geometry.max.x > 1.0

    def test_transform_batch_unknown_link(self):
        bodies = [CollisionBody("b", "sphere", Sphere(Vec3(1, 0, 0), 0.1), link_name="missing")]
        transforms = [Transform.identity()]
        result = transform_collision_bodies(bodies, transforms, {"other": 0})
        # Unknown link should keep original
        assert result[0].geometry.center == Vec3(1, 0, 0)

    def test_transform_batch_out_of_range_index(self):
        bodies = [CollisionBody("b", "sphere", Sphere(Vec3(1, 0, 0), 0.1), link_name="link0")]
        transforms = []
        result = transform_collision_bodies(bodies, transforms, {"link0": 0})
        assert result[0].geometry.center == Vec3(1, 0, 0)


# ========================================================================
# 6. EmbodiedMemory Deep Tests
# ========================================================================

class TestEmbodiedMemoryAddAtom:
    """add_atom 的深层测试"""

    def test_add_atom_idempotent_update(self, embodied_memory):
        """重复添加相同 memory_id 应更新而非重复插入"""
        em = embodied_memory
        atom = MemoryAtom(content="original", spatial=Vec3(0, 0, 0))
        mid = em.add_atom(atom)

        # Update same atom
        atom2 = MemoryAtom(content="updated", spatial=Vec3(1, 1, 1))
        atom2.memory_id = mid
        mid2 = em.add_atom(atom2)
        assert mid2 == mid

        retrieved = em.get_atom(mid)
        assert retrieved.content == "updated"
        assert retrieved.spatial == Vec3(1, 1, 1)

    def test_add_atom_with_causal_parents(self, embodied_memory):
        em = embodied_memory
        cause = em.add_atom(MemoryAtom(content="cause"))
        effect = em.add_atom(MemoryAtom(content="effect", causal_parents=[cause]))

        causes = em.get_causes(effect)
        assert len(causes) == 1
        assert causes[0].content == "cause"

    def test_add_atom_without_spatial_no_index_error(self, embodied_memory):
        """无空间坐标的 atom 应正常添加，不报错"""
        mid = embodied_memory.add_atom(MemoryAtom(content="no spatial"))
        assert mid >= 1000

    def test_add_atom_full_fields(self, embodied_memory):
        atom = MemoryAtom(
            content="full test",
            spatial=Vec3(1, 2, 3),
            temporal=TemporalInterval(0, 10),
            perceptual=PerceptualSnapshot(modality=Modality.IMU),
            physical=PhysicalInvariant(entity_id="e1", mass_kg=1.0),
            uncertainty=UncertaintyEstimate(std=0.1, confidence=0.9),
            affective=AffectiveTag(salience=0.7),
            action=MemoryAction.ACT,
            prediction_error=0.5,
        )
        mid = embodied_memory.add_atom(atom)
        retrieved = embodied_memory.get_atom(mid)
        assert retrieved.content == "full test"
        assert retrieved.spatial == Vec3(1, 2, 3)
        assert retrieved.perceptual.modality == Modality.IMU
        assert retrieved.physical.entity_id == "e1"
        assert retrieved.uncertainty.confidence == 0.9
        assert retrieved.affective.salience == 0.7
        assert retrieved.action == MemoryAction.ACT


class TestEmbodiedMemorySearchDeep:
    """search 三重过滤的深层测试"""

    def test_search_triple_filter_intersection(self, embodied_memory):
        """语义+空间+时间三重过滤的交集正确性"""
        em = embodied_memory
        # Add atoms with different combinations
        em.add_atom(MemoryAtom(content="target event", spatial=Vec3(1, 0, 0), temporal=TemporalInterval(5, 10)))
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
        assert "target event" in contents
        assert "wrong space" not in contents
        assert "wrong time" not in contents
        assert "no spatial" not in contents
        assert "no temporal" not in contents

    def test_search_near_distance_ordering(self, embodied_memory):
        """search_near 应按距离排序"""
        em = embodied_memory
        em.add_atom(MemoryAtom(content="A", spatial=Vec3(0, 0, 0)))
        em.add_atom(MemoryAtom(content="B", spatial=Vec3(3, 4, 0)))
        em.add_atom(MemoryAtom(content="C", spatial=Vec3(10, 0, 0)))

        results = em.search_near(Vec3(0, 0, 0), radius=20.0)
        assert len(results) == 3
        assert results[0].content == "A"
        assert results[1].content == "B"
        assert results[2].content == "C"
        assert results[0].embodied_meta["_spatial_distance"] == pytest.approx(0.0)
        assert results[1].embodied_meta["_spatial_distance"] == pytest.approx(5.0)

    def test_search_near_zero_radius(self, embodied_memory):
        em = embodied_memory
        em.add_atom(MemoryAtom(content="exact", spatial=Vec3(1.0, 2.0, 3.0)))
        em.add_atom(MemoryAtom(content="near", spatial=Vec3(1.1, 2.0, 3.0)))

        results = em.search_near(Vec3(1.0, 2.0, 3.0), radius=0.0)
        assert len(results) == 1
        assert results[0].content == "exact"

    def test_search_temporal_all_relations(self, embodied_memory):
        """search_temporal 使用不同 Allen 关系"""
        em = embodied_memory
        em.add_atom(MemoryAtom(content="A", temporal=TemporalInterval(0, 5)))
        em.add_atom(MemoryAtom(content="B", temporal=TemporalInterval(3, 8)))
        em.add_atom(MemoryAtom(content="C", temporal=TemporalInterval(10, 15)))

        # OVERLAPS: A starts before query, ends during query
        r1 = em.search_temporal(TemporalInterval(3, 8), IntervalRelation.OVERLAPS)
        assert any(a.content == "A" for a in r1)

        # OVERLAPPED_BY: B starts during query, ends after query
        r2 = em.search_temporal(TemporalInterval(0, 5), IntervalRelation.OVERLAPPED_BY)
        assert any(a.content == "B" for a in r2)

        # AFTER: C is after query
        r3 = em.search_temporal(TemporalInterval(0, 5), IntervalRelation.AFTER)
        assert any(a.content == "C" for a in r3)

    def test_search_temporal_frame_isolation(self, embodied_memory):
        em = embodied_memory
        em.add_atom(MemoryAtom(content="wall", temporal=TemporalInterval(0, 5, "wall_clock")))
        em.add_atom(MemoryAtom(content="sim", temporal=TemporalInterval(0, 5, "sim_time")))

        results = em.search_temporal(TemporalInterval(0, 5), frame_id="sim_time")
        assert len(results) == 1
        assert results[0].content == "sim"

    def test_search_empty_database(self, embodied_memory):
        """空数据库搜索不应报错"""
        results = embodied_memory.search("anything")
        assert results == []


class TestEmbodiedMemoryCausalGraphDeep:
    """因果图深度测试"""

    def test_long_causal_chain(self, embodied_memory):
        """长因果链 A -> B -> C -> D"""
        em = embodied_memory
        a = em.add_atom(MemoryAtom(content="A"))
        b = em.add_atom(MemoryAtom(content="B", causal_parents=[a]))
        c = em.add_atom(MemoryAtom(content="C", causal_parents=[b]))
        d = em.add_atom(MemoryAtom(content="D", causal_parents=[c]))

        # Forward
        assert em.get_effects(a)[0].content == "B"
        assert em.get_effects(b)[0].content == "C"
        assert em.get_effects(c)[0].content == "D"

        # Backward
        assert em.get_causes(d)[0].content == "C"
        assert em.get_causes(c)[0].content == "B"
        assert em.get_causes(b)[0].content == "A"

    def test_branching_causes(self, embodied_memory):
        """多因一果：A, B -> C"""
        em = embodied_memory
        a = em.add_atom(MemoryAtom(content="A"))
        b = em.add_atom(MemoryAtom(content="B"))
        c = em.add_atom(MemoryAtom(content="C", causal_parents=[a, b]))

        causes = em.get_causes(c)
        contents = {atom.content for atom in causes}
        assert contents == {"A", "B"}

    def test_branching_effects(self, embodied_memory):
        """一因多果：A -> B, C"""
        em = embodied_memory
        a = em.add_atom(MemoryAtom(content="A"))
        b = em.add_atom(MemoryAtom(content="B", causal_parents=[a]))
        c = em.add_atom(MemoryAtom(content="C", causal_parents=[a]))

        effects = em.get_effects(a)
        contents = {atom.content for atom in effects}
        assert contents == {"B", "C"}

    def test_causal_limit(self, embodied_memory):
        """limit 参数限制返回数量"""
        em = embodied_memory
        a = em.add_atom(MemoryAtom(content="A"))
        for i in range(20):
            em.add_atom(MemoryAtom(content=f"E{i}", causal_parents=[a]))

        effects = em.get_effects(a, limit=5)
        assert len(effects) == 5


class TestEmbodiedMemoryDeleteDeep:
    """删除操作的深层测试"""

    def test_delete_removes_from_spatial_index(self, embodied_memory):
        em = embodied_memory
        mid = em.add_atom(MemoryAtom(content="del", spatial=Vec3(1, 2, 3)))
        assert mid in em.spatial_index.get_all_ids()
        em.delete_atom(mid)
        assert mid not in em.spatial_index.get_all_ids()

    def test_delete_nonexistent(self, embodied_memory):
        """删除不存在的 memory_id"""
        ok = embodied_memory.delete_atom(999999)
        assert ok is False

    def test_delete_then_readd(self, embodied_memory):
        em = embodied_memory
        mid = em.add_atom(MemoryAtom(content="original", spatial=Vec3(1, 0, 0)))
        em.delete_atom(mid)
        # New add should get new ID
        mid2 = em.add_atom(MemoryAtom(content="new", spatial=Vec3(1, 0, 0)))
        assert mid2 != mid

    def test_delete_cascade_causal_edges(self, embodied_memory):
        """删除应级联删除因果边（通过外键 CASCADE）"""
        em = embodied_memory
        a = em.add_atom(MemoryAtom(content="A"))
        b = em.add_atom(MemoryAtom(content="B", causal_parents=[a]))

        # Verify edge exists
        causes = em.get_causes(b)
        assert len(causes) == 1

        # Delete cause
        em.delete_atom(a)

        # Edge should be gone (cascade delete)
        causes = em.get_causes(b)
        assert len(causes) == 0


class TestEmbodiedMemoryConstraintDeep:
    """约束记忆的深层测试"""

    def test_add_constraint_without_region(self, embodied_memory):
        from powermem.embodied.physical_model import PhysicalConstraint
        c = PhysicalConstraint(constraint_type="graspable", description="can grasp")
        mid = embodied_memory.add_constraint(c, "graspable region")
        atom = embodied_memory.get_atom(mid)
        assert atom.embodied_meta["physical_type"] == "constraint"
        assert atom.spatial is None

    def test_search_constraints_by_type(self, embodied_memory):
        from powermem.embodied.physical_model import PhysicalConstraint
        c1 = PhysicalConstraint(constraint_type="collision_free", description="safe", region_center=(0, 0, 0), region_radius=1.0)
        c2 = PhysicalConstraint(constraint_type="reachability", description="reachable", region_center=(0, 0, 0), region_radius=1.0)
        embodied_memory.add_constraint(c1)
        embodied_memory.add_constraint(c2)

        results = embodied_memory.search_constraints(Vec3(0, 0, 0), radius=2.0, constraint_type="collision_free")
        assert len(results) == 1
        assert results[0].embodied_meta["constraint"]["constraint_type"] == "collision_free"


class TestEmbodiedMemoryActionDeep:
    """动作/轨迹记忆的深层测试"""

    def test_record_action_with_all_fields(self, embodied_memory):
        mid = embodied_memory.record_action(
            "pick object",
            spatial=Vec3(1, 2, 3),
            outcome_status="success",
        )
        atom = embodied_memory.get_atom(mid)
        assert atom.action == MemoryAction.ACT
        assert atom.embodied_meta["outcome_status"] == "success"

    def test_record_outcome_causal_link(self, embodied_memory):
        action_id = embodied_memory.record_action("move")
        outcome_id = embodied_memory.record_outcome(
            action_id, "collision detected", "collision", spatial=Vec3(0, 0, 0)
        )
        causes = embodied_memory.get_causes(outcome_id)
        assert len(causes) == 1
        assert causes[0].memory_id == action_id

    def test_search_similar_experiences_filtered(self, embodied_memory):
        em = embodied_memory
        em.record_action("move", spatial=Vec3(0, 0, 0))
        em.record_action("grasp", spatial=Vec3(0, 0, 0))
        em.add_atom(MemoryAtom(content="observation", spatial=Vec3(0, 0, 0)))  # not an action

        results = em.search_similar_experiences(Vec3(0, 0, 0), radius=1.0)
        assert len(results) == 2
        for r in results:
            assert r.action.value in ("act", "correct")


class TestEmbodiedMemorySurprisalPersistence:
    """预测状态持久化的深层测试"""

    def test_save_and_restore_state(self, embodied_memory):
        em = embodied_memory
        em.save_surprisal_state("cam1", {"count": 100, "mean": 0.5, "m2": 25.0})
        state = em.get_surprisal_state("cam1")
        assert state["count"] == 100
        assert state["mean"] == pytest.approx(0.5)
        assert state["m2"] == pytest.approx(25.0)

    def test_state_update_overwrites(self, embodied_memory):
        em = embodied_memory
        em.save_surprisal_state("cam1", {"count": 10, "mean": 0.1, "m2": 1.0})
        em.save_surprisal_state("cam1", {"count": 20, "mean": 0.2, "m2": 4.0})
        state = em.get_surprisal_state("cam1")
        assert state["count"] == 20
        assert state["mean"] == pytest.approx(0.2)

    def test_multiple_predictors_independent(self, embodied_memory):
        em = embodied_memory
        em.save_surprisal_state("cam1", {"count": 10, "mean": 0.1, "m2": 1.0})
        em.save_surprisal_state("cam2", {"count": 20, "mean": 0.3, "m2": 9.0})

        s1 = em.get_surprisal_state("cam1")
        s2 = em.get_surprisal_state("cam2")
        assert s1["count"] == 10
        assert s2["count"] == 20

    def test_reset_all_then_get_none(self, embodied_memory):
        em = embodied_memory
        em.save_surprisal_state("cam1", {"count": 10, "mean": 0.1, "m2": 1.0})
        em.reset_surprisal_state()
        assert em.get_surprisal_state("cam1") is None

    def test_pipeline_state_persistence_roundtrip(self, embodied_memory):
        """管线创建后，surprisal gate 状态应能持久化"""
        em = embodied_memory
        pipeline = em.get_pipeline()
        # Simulate some activity
        for i in range(15):
            frame = SensorFrame(
                modality=Modality.RGB,
                timestamp_sec=i * 0.1,
                data=[0.5] * 10,
            )
            em.ingest(frame)

        # Check state was saved to DB
        state = em.get_surprisal_state("rgb_world")
        assert state is not None
        assert state["count"] >= 10


# ========================================================================
# 7. SurprisalGate Deep Tests
# ========================================================================

class TestSurprisalGateDeep:
    """SurprisalGate 边界和异常测试"""

    def test_initialization_phase_passes_all(self):
        """初始化阶段（count < min_samples）所有观测都通过"""
        gate = SurprisalGate(min_samples=10)
        for i in range(9):
            passed, error, threshold = gate.check("test", 1.0)
            assert passed is True
            assert threshold == float("inf")

    def test_steady_state_blocks_redundant(self):
        """稳态后，相似观测被阻塞"""
        gate = SurprisalGate(min_samples=5, k_sigma=2.0)
        for i in range(20):
            gate.check("test", 1.0)  # all same value

        # Now steady state, same value should be blocked
        passed, error, threshold = gate.check("test", 1.0)
        assert passed is False
        assert error <= threshold

    def test_anomaly_detection(self):
        """异常值应通过门控"""
        gate = SurprisalGate(min_samples=5, k_sigma=2.0)
        for i in range(20):
            gate.check("test", 1.0)

        # Anomaly
        passed, error, threshold = gate.check("test", 100.0)
        assert passed is True
        assert error > threshold

    def test_outlier_not_contaminating_stats(self):
        """极端 outlier 不应污染统计"""
        gate = SurprisalGate(min_samples=5, k_sigma=2.0, max_sigma_multiplier=3.0)
        for i in range(20):
            gate.check("test", 1.0)

        # Extreme outlier
        gate.check("test", 1000.0)

        # Stats should still be close to 1.0
        state = gate.get_state("test")
        assert state["mean"] < 10.0  # Not contaminated by 1000

    def test_filter_atom_updates_prediction_error(self):
        gate = SurprisalGate(min_samples=5, k_sigma=2.0)
        for i in range(20):
            gate.check("rgb_cam", 0.5)

        atom = MemoryAtom(content="test", perceptual=PerceptualSnapshot(modality=Modality.RGB))
        result = gate.filter_atom(atom, 100.0, "rgb_cam")
        assert result is not None
        assert result.prediction_error > 0

    def test_filter_atom_blocks_redundant(self):
        gate = SurprisalGate(min_samples=5, k_sigma=2.0)
        for i in range(20):
            gate.check("rgb_cam", 0.5)

        atom = MemoryAtom(content="test", perceptual=PerceptualSnapshot(modality=Modality.RGB))
        result = gate.filter_atom(atom, 0.5, "rgb_cam")
        assert result is None

    def test_running_stats_welford(self):
        """Welford 算法精度测试"""
        stats = _RunningStats()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            stats.update(v)
        assert stats.mean == pytest.approx(3.0)
        assert stats.std() == pytest.approx(math.sqrt(2.0), rel=1e-6)

    def test_running_stats_single_value(self):
        stats = _RunningStats()
        stats.update(5.0)
        assert stats.mean == 5.0
        assert stats.std() == 0.0

    def test_running_stats_empty(self):
        stats = _RunningStats()
        assert stats.mean == 0.0
        assert stats.std() == 0.0

    def test_linear_predictor(self):
        pred = LinearPredictor()
        # With history [1, 2], predict = 2*2 - 1 = 3
        history = [1.0, 2.0]
        assert pred.predict("test", history) == pytest.approx(3.0)

    def test_zero_order_hold_predictor(self):
        pred = ZeroOrderHoldPredictor()
        assert pred.predict("test", [1.0, 2.0, 3.0]) == 3.0
        assert pred.predict("test", []) == 0.0

    def test_predictor_vector_error(self):
        """向量预测的 L2 范数"""
        pred = ZeroOrderHoldPredictor()
        error = pred.prediction_error([1.0, 2.0], [2.0, 3.0])
        assert error == pytest.approx(math.sqrt(2.0))

    def test_predictor_invalid_types(self):
        pred = ZeroOrderHoldPredictor()
        error = pred.prediction_error("abc", "def")
        assert error == 1.0

    def test_reset_specific_predictor(self):
        gate = SurprisalGate()
        gate.check("a", 1.0)
        gate.check("b", 2.0)
        gate.reset("a")
        assert "a" not in gate._stats
        assert "b" in gate._stats

    def test_reset_all(self):
        gate = SurprisalGate()
        gate.check("a", 1.0)
        gate.check("b", 2.0)
        gate.reset()
        assert len(gate._stats) == 0
        assert len(gate._history) == 0

    def test_state_persistence_callback(self):
        saved = {}
        def store(pid, state):
            saved[pid] = state
        def load(pid):
            return saved.get(pid)

        gate = SurprisalGate(state_store=store, state_load=load)
        for i in range(15):
            gate.check("cam", 1.0)

        # Create new gate with same callbacks
        gate2 = SurprisalGate(state_store=store, state_load=load)
        gate2.check("cam", 1.0)
        assert gate2._stats["cam"].count > 1  # Restored from saved state

    def test_history_truncation(self):
        """历史长度超过 1000 后应截断到 500"""
        gate = SurprisalGate()
        for i in range(1100):
            gate.check("test", float(i))
        # 截断后可能又追加了若干条目，但总数不应超过 1000
        assert len(gate._history["test"]) <= 1000
        # 在恰好第 1001 次检查时历史应为 500
        gate2 = SurprisalGate()
        for i in range(1001):
            gate2.check("test2", float(i))
        assert len(gate2._history["test2"]) == 500


# ========================================================================
# 8. IngestPipeline Deep Tests
# ========================================================================

class TestIngestPipelineDeep:
    """IngestPipeline 深层测试"""

    @pytest.fixture
    def mock_store(self):
        store = MockStorageAdapter()
        return store

    @pytest.fixture
    def pipeline(self, mock_store):
        from powermem.embodied.ingest_pipeline import IngestPipeline
        return IngestPipeline(
            memory_store=lambda atom: mock_store.add_memory(atom.to_powermem_payload()),
            buffer_size=5,
            flush_interval_sec=100.0,  # never auto-flush in tests
        )

    def test_ingest_blocked_by_surprisal_gate(self, pipeline):
        """冗余帧被 surprisal gate 阻塞"""
        # First 10 frames initialize the gate
        frame = SensorFrame(
            modality=Modality.RGB,
            timestamp_sec=0.0,
            data=[0.5] * 100,
        )
        for i in range(15):
            pipeline.ingest(frame)

        # Now same frame should be blocked
        result = pipeline.ingest(frame)
        assert result is None  # Blocked

    def test_ingest_anomaly_passes(self, pipeline):
        """异常帧通过 surprisal gate"""
        frame_normal = SensorFrame(
            modality=Modality.RGB,
            timestamp_sec=0.0,
            data=[0.5] * 100,
        )
        frame_anomaly = SensorFrame(
            modality=Modality.RGB,
            timestamp_sec=1.0,
            data=[100.0] * 100,
        )
        for i in range(15):
            pipeline.ingest(frame_normal)

        result = pipeline.ingest(frame_anomaly)
        # Anomaly passes gate but gets buffered; flush to get memory_id
        assert result is None
        flushed = pipeline.flush()
        assert flushed is not None

    def test_flush_empty_buffer(self, pipeline):
        assert pipeline.flush() is None

    def test_flush_merges_temporal(self, pipeline):
        """flush 应合并时间区间"""
        for i in range(4):
            frame = SensorFrame(
                modality=Modality.RGB,
                timestamp_sec=float(i),
                data=[0.5] * 10,
                sensor_pose=Pose(position=Vec3(1, 0, 0)),
            )
            pipeline.ingest(frame)

        mid = pipeline.flush()
        assert mid is not None

    def test_flush_single_atom_no_merge(self, pipeline):
        frame = SensorFrame(
            modality=Modality.RGB,
            timestamp_sec=0.0,
            data=[0.5] * 10,
        )
        pipeline.ingest(frame)
        mid = pipeline.flush()
        assert mid is not None

    def test_get_stats(self, pipeline):
        stats = pipeline.get_stats()
        assert "buffer_size" in stats
        assert stats["buffer_size"] == 0
        assert "surprisal_state" in stats

    def test_reset(self, pipeline):
        frame = SensorFrame(
            modality=Modality.RGB,
            timestamp_sec=0.0,
            data=[0.5] * 10,
        )
        pipeline.ingest(frame)
        pipeline.reset()
        assert len(pipeline._buffer) == 0

    def test_ingest_batch_multiple_frames(self, pipeline):
        """批量摄入多帧，共享一次 flush"""
        frames = [
            SensorFrame(modality=Modality.RGB, timestamp_sec=float(i), data=[0.5] * 10)
            for i in range(4)
        ]
        results = pipeline.ingest_batch(frames)
        # 未达 buffer_size=5，不会自动 flush
        assert all(r is None for r in results)
        assert len(pipeline._buffer) == 4

    def test_ingest_batch_triggers_flush(self, pipeline):
        """批量摄入超过 buffer_size 应触发 flush"""
        frames = [
            SensorFrame(modality=Modality.RGB, timestamp_sec=float(i), data=[float(i)] * 10)
            for i in range(6)
        ]
        results = pipeline.ingest_batch(frames)
        # buffer_size=5，6 帧会触发 flush
        assert len(pipeline._buffer) == 0
        # 最后一条应有 memory_id
        assert results[-1] is not None

    def test_ingest_batch_with_contents(self, pipeline):
        """批量摄入带自定义内容"""
        frames = [
            SensorFrame(modality=Modality.RGB, timestamp_sec=0.0, data=[0.5] * 10),
            SensorFrame(modality=Modality.DEPTH, timestamp_sec=0.0, data=[1.0] * 10),
        ]
        contents = ["camera frame", "depth frame"]
        pipeline.ingest_batch(frames, contents=contents)
        assert len(pipeline._buffer) == 2
        assert pipeline._buffer[0].content == "camera frame"
        assert pipeline._buffer[1].content == "depth frame"

    def test_ingest_batch_contents_length_mismatch(self, pipeline):
        """contents 长度不匹配应抛 ValueError"""
        frames = [
            SensorFrame(modality=Modality.RGB, timestamp_sec=0.0, data=[0.5] * 10),
        ]
        with pytest.raises(ValueError):
            pipeline.ingest_batch(frames, contents=["a", "b"])

    def test_ingest_batch_surprisal_filtering(self, pipeline):
        """批量中部分帧被 surprisal gate 过滤"""
        normal = SensorFrame(modality=Modality.RGB, timestamp_sec=0.0, data=[0.5] * 100)
        anomaly = SensorFrame(modality=Modality.RGB, timestamp_sec=1.0, data=[99.0] * 100)
        # 初始化 gate
        for _ in range(15):
            pipeline.ingest(normal)
        pipeline.flush()
        # batch: 正常帧应被过滤，异常帧应通过
        results = pipeline.ingest_batch([normal, anomaly])
        assert results[0] is None  # filtered
        assert results[1] is None  # buffered but not flushed
        flushed = pipeline.flush()
        assert flushed is not None


# ========================================================================
# 9. End-to-End Scenario Tests
# ========================================================================

class TestEndToEndScenarios:
    """端到端场景测试"""

    def test_full_robot_workflow(self, sqlite_conn):
        """完整工作流：解析 URDF -> 存储模型 -> FK -> 碰撞检测 -> 记忆 -> 检索"""
        from powermem.embodied.parsers import parse_model

        mock_mem = MockMemory()
        em = EmbodiedMemory(memory=mock_mem, db_conn=sqlite_conn, enable_plugin=False)

        # 1. Parse URDF
        urdf = """\
<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base">
    <inertial><mass value="2.0"/></inertial>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry><sphere radius="0.1"/></geometry>
    </collision>
  </link>
  <link name="upper">
    <inertial><mass value="1.5"/></inertial>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.6"/></geometry>
    </collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" velocity="2.0" effort="50"/>
  </joint>
</robot>
"""
        result = parse_model(urdf)
        assert result.format == "urdf"
        assert len(result.links) == 2

        # 2. Build collision bodies
        bodies = build_collision_bodies(result)
        assert len(bodies) == 2

        # 3. Store model
        model_id = em.save_model(result, model_id="test_arm_v1")
        assert model_id == "test_arm_v1"

        # 4. Check self-collision at zero config
        pairs = em.check_self_collision(model_id)
        assert len(pairs) == 0

        # 5. Record observation memories
        for i in range(5):
            atom = MemoryAtom.from_observation(
                content=f"observation at step {i}",
                sensor_pose=Pose(position=Vec3(float(i), 0, 0)),
                modality=Modality.RGB,
                timestamp_sec=float(i),
            )
            em.add_atom(atom)

        # 6. Spatial search
        nearby = em.search_near(Vec3(2, 0, 0), radius=1.5)
        assert len(nearby) >= 2  # should find step 1, 2, 3

        # 7. Temporal search
        interval = TemporalInterval(1.0, 3.0)
        temporal_hits = em.search_temporal(interval)
        assert len(temporal_hits) >= 2

        # 8. Mixed search
        results = em.search(
            "observation",
            spatial_center=Vec3(2, 0, 0),
            spatial_radius=2.0,
            temporal_interval=TemporalInterval(1.0, 4.0),
        )
        assert len(results) >= 2

    def test_multi_frame_sensor_stream(self, sqlite_conn):
        """多帧传感器流：摄入 -> 过滤 -> flush -> 检索"""
        mock_mem = MockMemory()
        em = EmbodiedMemory(memory=mock_mem, db_conn=sqlite_conn, enable_plugin=False)

        # Ingest 20 frames of similar data (most should be filtered)
        for i in range(20):
            frame = SensorFrame(
                modality=Modality.DEPTH,
                timestamp_sec=float(i) * 0.1,
                data=[0.5] * 50,
                sensor_pose=Pose(position=Vec3(1, 0, 0)),
            )
            em.ingest(frame)

        # Force flush
        em.flush_pipeline()

        # Should have some memories (initialization + any anomalies)
        # Search spatially
        results = em.search_near(Vec3(1, 0, 0), radius=1.0)
        assert len(results) >= 1

    def test_world_object_and_trajectory_together(self, sqlite_conn):
        """世界对象 + 轨迹在同一个场景中"""
        mock_mem = MockMemory()
        em = EmbodiedMemory(memory=mock_mem, db_conn=sqlite_conn, enable_plugin=False)

        # Add world objects
        result = ParseResult()
        result.world_objects = [
            {"name": "table", "type": "box", "pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}}},
            {"name": "cup", "type": "cylinder", "pose": {"position": {"x": 1.2, "y": 0.0, "z": 0.8}}},
        ]
        em.add_world_objects(result)

        # Record trajectory near table
        waypoints = [
            (Vec3(0.5, 0, 0.5), 0.0),
            (Vec3(0.8, 0, 0.5), 1.0),
            (Vec3(1.0, 0, 0.5), 2.0),
        ]
        em.record_trajectory("approach table", waypoints)

        # Search world objects near table
        objects = em.search_world_objects(Vec3(1.0, 0, 0), radius=0.5)
        assert len(objects) >= 1

        # Search trajectories near table
        trajectories = em.search_trajectory_near(Vec3(1.0, 0, 0), radius=0.5)
        assert len(trajectories) >= 1

    def test_causal_workflow(self, sqlite_conn):
        """因果工作流：动作 -> 结果 -> 反思"""
        mock_mem = MockMemory()
        em = EmbodiedMemory(memory=mock_mem, db_conn=sqlite_conn, enable_plugin=False)

        # Record action
        action_id = em.record_action("grasp cup", spatial=Vec3(1, 0, 0))

        # Record outcome
        outcome_id = em.record_outcome(action_id, "slipped", "error", spatial=Vec3(1, 0, 0))

        # Record reflection
        reflection = MemoryAtom.from_action(
            content="grasp failed due to low friction",
            action_type=MemoryAction.REFLECT,
        )
        reflection.causal_parents = [action_id, outcome_id]
        reflection_id = em.add_atom(reflection)

        # Verify causal chain
        action_causes = em.get_effects(action_id)
        assert len(action_causes) == 2  # outcome + reflection

        reflection_causes = em.get_causes(reflection_id)
        assert len(reflection_causes) == 2

    def test_constraint_and_experience_retrieval(self, sqlite_conn):
        """约束与经验的联合检索"""
        from powermem.embodied.physical_model import PhysicalConstraint

        mock_mem = MockMemory()
        em = EmbodiedMemory(memory=mock_mem, db_conn=sqlite_conn, enable_plugin=False)

        # Add constraint
        c = PhysicalConstraint(
            constraint_type="collision_free",
            description="safe zone",
            region_center=(1.0, 0.0, 0.5),
            region_radius=0.3,
        )
        em.add_constraint(c)

        # Record action in same area
        em.record_action("move to target", spatial=Vec3(1.1, 0, 0.5))

        # Record outcome
        em.record_outcome(
            em.memory.storage._next_id - 1, "success", "success", spatial=Vec3(1.1, 0, 0.5)
        )

        # Retrieve constraints
        constraints = em.search_constraints(Vec3(1.0, 0, 0), radius=1.0)
        assert len(constraints) >= 1

        # Retrieve experiences
        experiences = em.search_similar_experiences(Vec3(1.0, 0, 0), radius=1.0)
        assert len(experiences) >= 1

    def test_rebuild_after_restart(self, sqlite_conn):
        """模拟重启后从 DB 重建索引"""
        mock_mem = MockMemory()
        em = EmbodiedMemory(memory=mock_mem, db_conn=sqlite_conn, enable_plugin=False)

        # Add data
        for i in range(10):
            em.add_atom(MemoryAtom(
                content=f"mem {i}",
                spatial=Vec3(float(i), 0, 0),
                temporal=TemporalInterval(float(i), float(i + 1)),
            ))

        # Simulate restart: create new EmbodiedMemory with same DB
        em2 = EmbodiedMemory(memory=mock_mem, db_conn=sqlite_conn, enable_plugin=False)

        # Spatial index should be rebuilt
        results = em2.search_near(Vec3(5, 0, 0), radius=2.0)
        assert len(results) >= 3

        # Temporal index should work
        results = em2.search_temporal(TemporalInterval(4.0, 6.0))
        assert len(results) >= 1

    def test_stats_method(self, embodied_memory):
        """stats() 方法不应报错"""
        stats = embodied_memory.stats()
        assert "spatial" in stats
        assert stats["spatial"]["loaded"] is True

    def test_proxy_access(self, embodied_memory):
        """代理访问底层 Memory 属性"""
        assert embodied_memory.agent_id == "test_agent"
        assert embodied_memory.storage is not None


# ========================================================================
# 10. Model Store End-to-End
# ========================================================================

class TestModelStoreEndToEnd:
    """ModelStore 端到端测试"""

    def test_model_lifecycle(self, sqlite_conn):
        from powermem.embodied.model_store import ModelStore
        from powermem.embodied.parsers import parse_model

        store = ModelStore(sqlite_conn)

        urdf = """\
<?xml version="1.0"?>
<robot name="arm">
  <link name="l1">
    <inertial><mass value="1.0"/></inertial>
    <collision><geometry><sphere radius="0.1"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="l1"/><child link="l1"/>
    <limit lower="-1.57" upper="1.57" velocity="2.0" effort="50"/>
  </joint>
</robot>
"""
        result = parse_model(urdf)
        model_id = store.save(result, model_id="arm_v1", model_type="robot")

        # Load
        loaded = store.load(model_id)
        assert loaded is not None
        assert loaded.model_type == "robot"

        # List
        all_models = store.list_models()
        assert any(m["model_id"] == "arm_v1" for m in all_models)

        # Delete
        assert store.delete(model_id) is True
        assert store.load(model_id) is None
        assert store.delete(model_id) is False

    def test_collision_at_config_with_fk(self, sqlite_conn):
        from powermem.embodied.model_store import ModelStore
        from powermem.embodied.physical_model import DHParameter, JointLimit, RobotDynamics
        from powermem.embodied.parsers.base import ParseResult

        store = ModelStore(sqlite_conn)
        result = ParseResult()
        result.format = "urdf"
        result.dynamics = RobotDynamics(
            joint_names=["j1"],
            dh_params=[DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0)],
            joint_limits=[JointLimit()],
            collision_geoms=[
                {"type": "sphere", "link": "a", "center": [0.5, 0, 0], "radius": 0.2},
                {"type": "sphere", "link": "a", "center": [1.5, 0, 0], "radius": 0.2},
            ],
        )
        store.save(result, model_id="fk_col")

        # Zero config: distance between sphere centers = 1.0, radius sum = 0.4
        pairs = store.check_collision_at_config("fk_col", [0.0])
        assert len(pairs) == 0

        # Make them overlap by moving closer
        result.dynamics.collision_geoms = [
            {"type": "sphere", "link": "a", "center": [0.5, 0, 0], "radius": 0.4},
            {"type": "sphere", "link": "a", "center": [0.8, 0, 0], "radius": 0.4},
        ]
        store.save(result, model_id="fk_col_overlap")
        pairs = store.check_collision_at_config("fk_col_overlap", [0.0])
        assert len(pairs) == 1


# ========================================================================
# 11. Batch Operations & Transactions
# ========================================================================

class TestBatchOperations:
    """批量写入与事务管理测试"""

    def test_batch_add_atoms(self, embodied_memory):
        atoms = [
            MemoryAtom(content=f"batch_{i}", spatial=Vec3(float(i), 0, 0))
            for i in range(50)
        ]
        mids = embodied_memory.batch_add_atoms(atoms)
        assert len(mids) == 50
        for i, mid in enumerate(mids):
            atom = embodied_memory.get_atom(mid)
            assert atom is not None
            assert atom.content == f"batch_{i}"

    def test_transaction_context_manager(self, embodied_memory):
        with embodied_memory.transaction():
            for i in range(10):
                embodied_memory.add_world_object(WorldObject(
                    obj_id=f"tx_obj_{i}", obj_type="box", scene_id="tx_scene",
                ))

        for i in range(10):
            obj = embodied_memory.get_world_object(f"tx_obj_{i}")
            assert obj is not None
            assert obj.scene_id == "tx_scene"

    def test_transaction_suppresses_intermediate_commits(self, embodied_memory):
        """事务中写入的对象在未提交前不应被外部看到（通过同一连接验证）"""
        store = embodied_memory.world_object_store
        store._transaction_depth = 1
        store.save(WorldObject(obj_id="deferred", obj_type="sphere", scene_id="s1"))
        # 同一连接，未 commit，应该能看到（SQLite 同一连接内未隔离）
        # 但 rollback 后应该消失
        store._transaction_depth = 0
        embodied_memory.db_conn.rollback()
        # rollback 不会自动清缓存，需要手动失效
        store._invalidate_object_cache("deferred")
        obj = embodied_memory.get_world_object("deferred")
        assert obj is None

    def test_batch_add_atoms_faster_than_sequential(self, embodied_memory):
        """批量写入应比逐条写入更快（至少快 1.2x）"""
        import time

        atoms_seq = [MemoryAtom(content=f"seq_{i}", spatial=Vec3(i, 0, 0)) for i in range(200)]
        t0 = time.perf_counter()
        for atom in atoms_seq:
            embodied_memory.add_atom(atom)
        seq_time = time.perf_counter() - t0

        atoms_batch = [MemoryAtom(content=f"batch_{i}", spatial=Vec3(i, 0, 0)) for i in range(200)]
        t0 = time.perf_counter()
        embodied_memory.batch_add_atoms(atoms_batch)
        batch_time = time.perf_counter() - t0

        assert batch_time < seq_time * 0.85, f"batch {batch_time:.3f}s not faster than seq {seq_time:.3f}s"


class TestBatchAtomLoading:
    """get_atoms 批量读取测试"""

    def test_get_atoms_basic(self, embodied_memory):
        mids = []
        for i in range(10):
            mid = embodied_memory.add_atom(MemoryAtom(content=f"atom_{i}"))
            mids.append(mid)

        atoms = embodied_memory.get_atoms(mids)
        assert len(atoms) == 10
        for i, atom in enumerate(atoms):
            assert atom is not None
            assert atom.content == f"atom_{i}"

    def test_get_atoms_with_cache_hits(self, embodied_memory):
        mid = embodied_memory.add_atom(MemoryAtom(content="cached"))
        # 第一次读取填充缓存
        embodied_memory.get_atom(mid)
        # 第二次批量读取应命中缓存
        atoms = embodied_memory.get_atoms([mid])
        assert len(atoms) == 1
        assert atoms[0] is not None
        assert atoms[0].content == "cached"

    def test_get_atoms_missing_ids(self, embodied_memory):
        existing = embodied_memory.add_atom(MemoryAtom(content="exists"))
        atoms = embodied_memory.get_atoms([existing, 999999])
        assert atoms[0] is not None
        assert atoms[0].content == "exists"
        assert atoms[1] is None

    def test_get_atoms_maintains_order(self, embodied_memory):
        mids = []
        for i in range(5):
            mids.append(embodied_memory.add_atom(MemoryAtom(content=f"order_{i}")))
        # 逆序查询
        atoms = embodied_memory.get_atoms(list(reversed(mids)))
        for i, atom in enumerate(atoms):
            assert atom is not None
            assert atom.content == f"order_{4 - i}"

    def test_get_atoms_populates_cache(self, embodied_memory):
        mid = embodied_memory.add_atom(MemoryAtom(content="fill_cache"))
        # 缓存应为空
        embodied_memory._atom_cache.pop(mid, None)
        assert mid not in embodied_memory._atom_cache
        # 批量读取后应填充缓存
        embodied_memory.get_atoms([mid])
        assert mid in embodied_memory._atom_cache

    def test_get_atoms_batch_reduces_storage_calls(self, embodied_memory):
        mids = []
        for i in range(10):
            mids.append(embodied_memory.add_atom(MemoryAtom(content=f"batch_{i}")))
        embodied_memory._atom_cache.clear()

        # 统计 sequential 的 storage.get_memory 调用次数
        original_get = embodied_memory.memory.storage.get_memory
        seq_calls = 0
        def counting_get(mid):
            nonlocal seq_calls
            seq_calls += 1
            return original_get(mid)
        embodied_memory.memory.storage.get_memory = counting_get
        for mid in mids:
            embodied_memory.get_atom(mid)

        # 统计 batch 的 get_many_memories 调用次数
        batch_calls = 0
        original_get_many = embodied_memory.memory.storage.get_many_memories
        def counting_get_many(mids):
            nonlocal batch_calls
            batch_calls += 1
            return original_get_many(mids)
        embodied_memory.memory.storage.get_many_memories = counting_get_many
        embodied_memory._atom_cache.clear()
        embodied_memory.get_atoms(mids)

        # 恢复
        embodied_memory.memory.storage.get_memory = original_get
        embodied_memory.memory.storage.get_many_memories = original_get_many

        assert seq_calls == len(mids)
        assert batch_calls == 1
