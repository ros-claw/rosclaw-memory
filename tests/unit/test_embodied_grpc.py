
"""
Tests for EmbodiedMemory gRPC service.

Uses an in-process gRPC server + real EmbodiedMemory with mock PowerMem storage.
"""

from __future__ import annotations

import sqlite3
from concurrent import futures
from typing import Any, Dict, Optional

import grpc
import pytest

from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.grpc.client import EmbodiedMemoryClient
from powermem.embodied.grpc.servicer import EmbodiedMemoryServicer
from powermem.embodied.memory_atom import MemoryAtom
from powermem.embodied.proto import embodied_memory_pb2
from powermem.embodied.proto import embodied_memory_pb2_grpc
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import MemoryAction, Modality, TemporalInterval, Vec3


# ---------------------------------------------------------------------------
# Mock PowerMem (same pattern as test_embodied_deep.py)
# ---------------------------------------------------------------------------

class _MockStorageAdapter:
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

    def update_memory(self, memory_id: int, content: str, user_id=None, agent_id=None, metadata=None) -> Dict[str, Any]:
        item = self._store.get(memory_id)
        if item is None:
            raise KeyError(memory_id)
        item["data"] = content
        item["content"] = content
        if metadata is not None:
            item["metadata"] = metadata
        return item


class _MockMemory:
    def __init__(self):
        self.storage = _MockStorageAdapter()
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
        return self.storage.update_memory(memory_id, content, user_id, agent_id, metadata)


@pytest.fixture
def sqlite_conn():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    initialize_embodied_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def grpc_server(sqlite_conn):
    mock_mem = _MockMemory()
    em = EmbodiedMemory(memory=mock_mem, db_conn=sqlite_conn, enable_plugin=False)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    embodied_memory_pb2_grpc.add_EmbodiedMemoryServiceServicer_to_server(
        EmbodiedMemoryServicer(em), server
    )
    port = server.add_insecure_port("localhost:0")
    server.start()
    yield f"localhost:{port}"
    server.stop(grace=1.0)


@pytest.fixture
def client(grpc_server):
    with EmbodiedMemoryClient(target=grpc_server) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGrpcAtomCRUD:
    def test_add_and_get_atom(self, client):
        atom = MemoryAtom(content="test atom", spatial=Vec3(1, 2, 3))
        mid = client.add_atom(atom)
        assert mid >= 1000

        retrieved = client.get_atom(mid)
        assert retrieved is not None
        assert retrieved.content == "test atom"
        assert retrieved.spatial == Vec3(1, 2, 3)

    def test_get_missing_atom(self, client):
        assert client.get_atom(999999) is None

    def test_delete_atom(self, client):
        mid = client.add_atom(MemoryAtom(content="to delete"))
        assert client.delete_atom(mid) is True
        assert client.get_atom(mid) is None

    def test_delete_nonexistent(self, client):
        assert client.delete_atom(999999) is False


class TestGrpcSearch:
    def test_search_near(self, client):
        client.add_atom(MemoryAtom(content="a", spatial=Vec3(0, 0, 0)))
        client.add_atom(MemoryAtom(content="b", spatial=Vec3(1, 0, 0)))
        client.add_atom(MemoryAtom(content="c", spatial=Vec3(10, 0, 0)))

        results = client.search_near(Vec3(0, 0, 0), radius=2.0)
        contents = [r.content for r in results]
        assert "a" in contents
        assert "b" in contents
        assert "c" not in contents

    def test_search_temporal(self, client):
        client.add_atom(MemoryAtom(
            content="morning",
            temporal=TemporalInterval(8.0, 10.0),
        ))
        client.add_atom(MemoryAtom(
            content="night",
            temporal=TemporalInterval(20.0, 22.0),
        ))

        results = client.search(
            query="memory",
            temporal_interval=TemporalInterval(9.0, 11.0),
        )
        contents = [r.content for r in results]
        assert "morning" in contents
        assert "night" not in contents


class TestGrpcTrajectory:
    def test_record_and_search_similar(self, client):
        # Record a straight-line trajectory
        waypoints = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(1, 0, 0), 1.0),
            (Vec3(2, 0, 0), 2.0),
        ]
        mid = client.record_trajectory("straight line", waypoints)
        assert mid is not None

        # Search with identical waypoints -> distance should be 0
        results = client.search_similar_trajectories(
            query_waypoints=waypoints,
            top_k=5,
        )
        assert len(results) >= 1
        atom, dtw = results[0]
        assert atom.content == "straight line"
        assert dtw == pytest.approx(0.0)

    def test_similar_trajectory_dtw_small(self, client):
        # Original
        w1 = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(1, 0, 0), 1.0),
            (Vec3(2, 0, 0), 2.0),
        ]
        client.record_trajectory("original", w1)

        # Slightly perturbed (same shape, small noise)
        w2 = [
            (Vec3(0.1, 0, 0), 0.0),
            (Vec3(1.1, 0, 0), 1.0),
            (Vec3(2.1, 0, 0), 2.0),
        ]
        results = client.search_similar_trajectories(
            query_waypoints=w2,
            top_k=5,
        )
        assert len(results) >= 1
        _, dtw = results[0]
        # Small perturbation should yield small DTW
        assert dtw < 0.5

    def test_different_trajectory_high_dtw(self, client):
        # Original straight line
        w1 = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(1, 0, 0), 1.0),
            (Vec3(2, 0, 0), 2.0),
        ]
        client.record_trajectory("straight", w1)

        # Query is perpendicular line
        w2 = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(0, 5, 0), 1.0),
            (Vec3(0, 10, 0), 2.0),
        ]
        results = client.search_similar_trajectories(
            query_waypoints=w2,
            top_k=5,
        )
        # Signature pre-filter may reject it, or DTW will be large
        if results:
            _, dtw = results[0]
            assert dtw > 1.0

    def test_max_dtw_distance_filter(self, client):
        w1 = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(10, 0, 0), 1.0),
        ]
        client.record_trajectory("far", w1)

        w2 = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(0, 0, 0), 1.0),
        ]
        # With tight threshold, should exclude "far"
        results = client.search_similar_trajectories(
            query_waypoints=w2,
            max_dtw_distance=0.5,
            top_k=5,
        )
        assert len(results) == 0


class TestGrpcIngest:
    def test_ingest_and_get_stats(self, client):
        frame = embodied_memory_pb2.SensorFrame(
            modality="rgb",
            timestamp_sec=0.0,
            data=[0.5] * 10,
            sensor_position=embodied_memory_pb2.Vec3(x=1, y=0, z=0),
        )
        req = embodied_memory_pb2.IngestSensorFrameRequest(frame=frame, content="test ingest")
        resp = client.stub.IngestSensorFrame(req)
        # First frame initializes gate, may or may not store depending on pipeline config
        assert resp.error_message == ""

    def test_get_stats(self, client):
        stats = client.get_stats()
        assert "spatial" in stats


class TestGrpcCausal:
    def test_causes_and_effects(self, client):
        cause_id = client.add_atom(MemoryAtom(content="cause"))
        effect_id = client.add_atom(MemoryAtom(content="effect", causal_parents=[cause_id]))

        # Use low-level stub for causal RPCs (not exposed on client helper yet)
        from powermem.embodied.proto.embodied_memory_pb2 import GetCausesRequest, GetEffectsRequest

        causes_resp = client.stub.GetCauses(GetCausesRequest(memory_id=effect_id))
        assert len(causes_resp.atoms) == 1
        assert causes_resp.atoms[0].content == "cause"

        effects_resp = client.stub.GetEffects(GetEffectsRequest(memory_id=cause_id))
        assert len(effects_resp.atoms) == 1
        assert effects_resp.atoms[0].content == "effect"


class TestGrpcWorldObjects:
    def test_add_and_get_world_object(self, client):
        from powermem.embodied.types import Pose, Vec3, WorldObject

        obj = WorldObject(
            obj_id="grpc_cube",
            obj_type="box",
            name="test cube",
            pose=Pose(position=Vec3(1, 2, 3)),
            size=(0.1, 0.1, 0.1),
            scene_id="grpc_scene",
        )
        oid = client.add_world_object(obj)
        assert oid == "grpc_cube"

        loaded = client.get_world_object("grpc_cube")
        assert loaded is not None
        assert loaded.name == "test cube"
        assert loaded.pose.position == Vec3(1, 2, 3)

    def test_get_missing_world_object(self, client):
        assert client.get_world_object("nonexistent") is None

    def test_update_world_object_pose(self, client):
        from powermem.embodied.types import Pose, Vec3, WorldObject

        client.add_world_object(WorldObject(
            obj_id="movable",
            pose=Pose(position=Vec3(0, 0, 0)),
            scene_id="s1",
        ))
        ok = client.update_world_object_pose(
            "movable", Pose(position=Vec3(5, 5, 5)), state="moved"
        )
        assert ok is True
        loaded = client.get_world_object("movable")
        assert loaded.pose.position == Vec3(5, 5, 5)
        assert loaded.state == "moved"

    def test_search_world_objects(self, client):
        from powermem.embodied.types import Pose, Vec3, WorldObject

        client.add_world_object(WorldObject(
            obj_id="near", pose=Pose(position=Vec3(0, 0, 0)), scene_id="search_scene"
        ))
        client.add_world_object(WorldObject(
            obj_id="far", pose=Pose(position=Vec3(100, 0, 0)), scene_id="search_scene"
        ))
        results = client.search_world_objects(
            Vec3(0, 0, 0), radius=2.0, scene_id="search_scene"
        )
        ids = [r.obj_id for r in results]
        assert "near" in ids
        assert "far" not in ids

    def test_get_scene_graph(self, client):
        from powermem.embodied.types import Pose, Vec3, WorldObject

        client.add_world_object(WorldObject(obj_id="room", scene_id="sg_scene"))
        client.add_world_object(WorldObject(obj_id="table", scene_id="sg_scene", parent_obj_id="room"))
        objects, relations = client.get_scene_graph("sg_scene")
        ids = [o.obj_id for o in objects]
        assert "room" in ids
        assert "table" in ids

    def test_compute_relations(self, client):
        from powermem.embodied.types import Pose, Vec3, WorldObject

        client.add_world_object(WorldObject(
            obj_id="table",
            obj_type="box",
            pose=Pose(position=Vec3(0, 0, 0)),
            size=(1.0, 1.0, 1.0),
            scene_id="rel_scene",
        ))
        client.add_world_object(WorldObject(
            obj_id="cup",
            obj_type="box",
            pose=Pose(position=Vec3(0, 0, 0.6)),
            size=(0.1, 0.1, 0.1),
            scene_id="rel_scene",
        ))
        relations = client.compute_relations("rel_scene", spatial_tolerance=0.05)
        on_rels = [r for r in relations if r.relation == "on"]
        assert len(on_rels) == 1
        assert on_rels[0].subject_id == "cup"
        assert on_rels[0].object_id == "table"


class TestGrpcObjectPermanence:
    def test_sync_scene_objects(self, client):
        from powermem.embodied.types import Pose, Vec3, WorldObject

        # Add pre-existing object
        client.add_world_object(WorldObject(
            obj_id="grpc_mug", obj_type="cylinder", name="mug",
            pose=Pose(position=Vec3(1, 0, 0)), scene_id="grpc_kitchen",
        ))

        # Sync with detection at new position
        report = client.sync_scene_objects(
            scene_id="grpc_kitchen",
            detections=[WorldObject(
                obj_id="grpc_mug", obj_type="cylinder", name="mug",
                pose=Pose(position=Vec3(2, 0, 0)), scene_id="grpc_kitchen",
            )],
            timestamp_sec=5.0,
        )
        assert len(report["transitions"]) == 0  # still visible

        obj = client.get_world_object("grpc_mug")
        assert obj.pose.position == Vec3(2, 0, 0)
        assert obj.occlusion_status == "visible"

    def test_sync_scene_objects_occlusion(self, client):
        from powermem.embodied.types import Pose, Vec3, WorldObject

        client.add_world_object(WorldObject(
            obj_id="grpc_cup", obj_type="box", scene_id="grpc_kitchen",
            pose=Pose(position=Vec3(0, 0, 0)),
        ))

        # Sync with empty detections → object becomes occluded
        report = client.sync_scene_objects(
            scene_id="grpc_kitchen",
            detections=[],
            timestamp_sec=10.0,
        )
        assert len(report["transitions"]) == 1
        assert "visible -> occluded" in report["transitions"][0]


class TestGrpcCognitiveSearch:
    def test_cognitive_search_basic(self, client):
        client.add_atom(MemoryAtom(content="test search", spatial=Vec3(1, 2, 3)))
        results = client.cognitive_search(query="search")
        contents = [r.content for r in results]
        assert "test search" in contents

    def test_cognitive_search_with_spatial(self, client):
        client.add_atom(MemoryAtom(content="nearby", spatial=Vec3(0, 0, 0)))
        client.add_atom(MemoryAtom(content="faraway", spatial=Vec3(100, 0, 0)))

        results = client.cognitive_search(
            query="memory",
            spatial_center=Vec3(0, 0, 0),
            spatial_radius=2.0,
        )
        contents = [r.content for r in results]
        assert "nearby" in contents
        assert "faraway" not in contents

    def test_cognitive_search_with_temporal(self, client):
        client.add_atom(MemoryAtom(
            content="morning event",
            temporal=TemporalInterval(8.0, 10.0),
        ))
        client.add_atom(MemoryAtom(
            content="night event",
            temporal=TemporalInterval(20.0, 22.0),
        ))

        results = client.cognitive_search(
            query="event",
            temporal_interval=TemporalInterval(9.0, 11.0),
        )
        contents = [r.content for r in results]
        assert "morning event" in contents
        assert "night event" not in contents


class TestGrpcConceptAndExperienceGraph:
    def test_index_concept(self, client):
        mid = client.add_atom(MemoryAtom(content="concept test"))
        ok = client.index_concept(
            memory_id=mid,
            dimension="task",
            layer=2,
            concept_id="grasp_mug",
            confidence=0.95,
        )
        assert ok is True

    def test_add_experience_edge(self, client):
        source = client.add_atom(MemoryAtom(content="action"))
        target = client.add_atom(MemoryAtom(content="outcome"))
        ok = client.add_experience_edge(
            source_memory_id=source,
            target_memory_id=target,
            edge_type="causes",
            strength=0.9,
        )
        assert ok is True


class TestGrpcMeditation:
    def test_run_meditation(self, client):
        # Add some observations with entity_id for consolidation
        for i in range(3):
            client.add_atom(MemoryAtom(
                content=f"obs {i}",
                temporal=TemporalInterval(i, i + 1),
                embodied_meta={"entity_id": "grpc_cup"},
            ))

        report = client.run_meditation(phases=["consolidate"])
        assert report["success"] is True
        assert report["consolidated_count"] >= 0
        assert report["elapsed_sec"] >= 0

    def test_run_meditation_all_phases(self, client):
        report = client.run_meditation(phases=["consolidate", "crystallize", "extract"])
        assert report["success"] is True
        assert "errors" in report
