"""
Python client helper for EmbodiedMemory gRPC service.

Thin wrapper around the generated gRPC stubs for convenient use
from Python code or tests.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import grpc

from powermem.embodied.proto import embodied_memory_pb2
from powermem.embodied.proto import embodied_memory_pb2_grpc

from ..memory_atom import MemoryAtom
from ..types import Pose, SpatialRelation, TemporalInterval, Vec3, WorldObject


class EmbodiedMemoryClient:
    """Client for EmbodiedMemoryService."""

    def __init__(self, target: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(target)
        self.stub = embodied_memory_pb2_grpc.EmbodiedMemoryServiceStub(self.channel)

    def close(self) -> None:
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -----------------------------------------------------------------------
    # Atom CRUD
    # -----------------------------------------------------------------------

    def add_atom(self, atom: MemoryAtom, infer: bool = False) -> int:
        """Add a MemoryAtom, return memory_id."""
        from .servicer import _py_atom_to_pb

        req = embodied_memory_pb2.AddAtomRequest(
            atom=_py_atom_to_pb(atom), infer=infer
        )
        resp = self.stub.AddAtom(req)
        return resp.memory_id

    def get_atom(self, memory_id: int) -> Optional[MemoryAtom]:
        """Get atom by ID. Returns None if not found."""
        from .servicer import _pb_atom_to_py

        req = embodied_memory_pb2.GetAtomRequest(memory_id=memory_id)
        resp = self.stub.GetAtom(req)
        if not resp.found:
            return None
        return _pb_atom_to_py(resp.atom)

    def delete_atom(self, memory_id: int) -> bool:
        req = embodied_memory_pb2.DeleteAtomRequest(memory_id=memory_id)
        resp = self.stub.DeleteAtom(req)
        return resp.deleted

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        spatial_center: Optional[Vec3] = None,
        spatial_radius: Optional[float] = None,
        temporal_interval: Optional[TemporalInterval] = None,
        limit: int = 30,
    ) -> List[MemoryAtom]:
        from .servicer import _py_temporal_to_pb, _py_vec3_to_pb

        req = embodied_memory_pb2.SearchRequest(query=query, limit=limit)
        if spatial_center is not None:
            req.spatial_center.CopyFrom(_py_vec3_to_pb(spatial_center))
            req.spatial_radius = spatial_radius or 1.0
        if temporal_interval is not None:
            req.temporal_interval.CopyFrom(_py_temporal_to_pb(temporal_interval))
        resp = self.stub.Search(req)
        from .servicer import _pb_atom_to_py

        return [_pb_atom_to_py(a) for a in resp.atoms]

    def search_near(
        self,
        center: Vec3,
        radius: float,
        frame_id: str = "world",
        limit: int = 30,
    ) -> List[MemoryAtom]:
        from .servicer import _pb_atom_to_py, _py_vec3_to_pb

        req = embodied_memory_pb2.SearchNearRequest(
            center=_py_vec3_to_pb(center),
            radius=radius,
            frame_id=frame_id,
            limit=limit,
        )
        resp = self.stub.SearchNear(req)
        return [_pb_atom_to_py(a) for a in resp.atoms]

    # -----------------------------------------------------------------------
    # Trajectory
    # -----------------------------------------------------------------------

    def record_trajectory(
        self,
        content: str,
        waypoints: List[Tuple[Vec3, float]],
    ) -> int:
        from .servicer import _py_vec3_to_pb

        pb_waypoints = [
            embodied_memory_pb2.TrajectoryWaypoint(
                position=_py_vec3_to_pb(pos), timestamp_sec=ts
            )
            for pos, ts in waypoints
        ]
        req = embodied_memory_pb2.RecordTrajectoryRequest(
            content=content, waypoints=pb_waypoints
        )
        resp = self.stub.RecordTrajectory(req)
        return resp.memory_id

    def search_similar_trajectories(
        self,
        query_waypoints: List[Tuple[Vec3, float]],
        spatial_center: Optional[Vec3] = None,
        spatial_radius: Optional[float] = None,
        temporal_interval: Optional[TemporalInterval] = None,
        top_k: int = 10,
        max_dtw_distance: Optional[float] = None,
    ) -> List[Tuple[MemoryAtom, float]]:
        from .servicer import _py_temporal_to_pb, _py_vec3_to_pb

        pb_waypoints = [
            embodied_memory_pb2.TrajectoryWaypoint(
                position=_py_vec3_to_pb(pos), timestamp_sec=ts
            )
            for pos, ts in query_waypoints
        ]
        req = embodied_memory_pb2.SearchSimilarTrajectoriesRequest(
            query_waypoints=pb_waypoints,
            top_k=top_k,
        )
        if spatial_center is not None:
            req.spatial_center.CopyFrom(_py_vec3_to_pb(spatial_center))
            req.spatial_radius = spatial_radius or 1.0
        if temporal_interval is not None:
            req.temporal_interval.CopyFrom(_py_temporal_to_pb(temporal_interval))
        if max_dtw_distance is not None:
            req.max_dtw_distance = max_dtw_distance

        resp = self.stub.SearchSimilarTrajectories(req)
        from .servicer import _pb_atom_to_py

        return [(_pb_atom_to_py(r.atom), r.dtw_distance) for r in resp.results]

    # -----------------------------------------------------------------------
    # World Objects
    # -----------------------------------------------------------------------

    def add_world_object(self, obj: WorldObject) -> str:
        from .servicer import _py_world_object_to_pb

        req = embodied_memory_pb2.AddWorldObjectRequest(obj=_py_world_object_to_pb(obj))
        resp = self.stub.AddWorldObject(req)
        return resp.obj_id

    def get_world_object(self, obj_id: str) -> Optional[WorldObject]:
        from .servicer import _pb_world_object_to_py

        req = embodied_memory_pb2.GetWorldObjectRequest(obj_id=obj_id)
        resp = self.stub.GetWorldObject(req)
        if not resp.found:
            return None
        return _pb_world_object_to_py(resp.obj)

    def update_world_object_pose(self, obj_id: str, pose: Pose, state: Optional[str] = None) -> bool:
        from .servicer import _py_pose_to_pb

        req = embodied_memory_pb2.UpdateWorldObjectPoseRequest(obj_id=obj_id)
        req.pose.CopyFrom(_py_pose_to_pb(pose))
        if state is not None:
            req.state = state
        resp = self.stub.UpdateWorldObjectPose(req)
        return resp.success

    def search_world_objects(
        self,
        center: Vec3,
        radius: float,
        obj_type: Optional[str] = None,
        scene_id: Optional[str] = None,
        limit: int = 30,
    ) -> List[WorldObject]:
        from .servicer import _pb_world_object_to_py, _py_vec3_to_pb

        req = embodied_memory_pb2.SearchWorldObjectsRequest(
            center=_py_vec3_to_pb(center),
            radius=radius,
            limit=limit,
        )
        if obj_type is not None:
            req.obj_type = obj_type
        if scene_id is not None:
            req.scene_id = scene_id
        resp = self.stub.SearchWorldObjects(req)
        return [_pb_world_object_to_py(o) for o in resp.objects]

    def get_scene_graph(self, scene_id: str) -> Tuple[List[WorldObject], List[SpatialRelation]]:
        from .servicer import _pb_spatial_relation_to_py, _pb_world_object_to_py

        req = embodied_memory_pb2.GetSceneGraphRequest(scene_id=scene_id)
        resp = self.stub.GetSceneGraph(req)
        objects = [_pb_world_object_to_py(o) for o in resp.objects]
        relations = [_pb_spatial_relation_to_py(r) for r in resp.relations]
        return objects, relations

    def compute_relations(self, scene_id: str, spatial_tolerance: float = 0.01) -> List[SpatialRelation]:
        from .servicer import _pb_spatial_relation_to_py

        req = embodied_memory_pb2.ComputeRelationsRequest(
            scene_id=scene_id,
            spatial_tolerance=spatial_tolerance,
        )
        resp = self.stub.ComputeRelations(req)
        return [_pb_spatial_relation_to_py(r) for r in resp.relations]

    def sync_scene_objects(
        self,
        scene_id: str,
        detections: List[WorldObject],
        timestamp_sec: float,
        occlusion_radius: float = 0.5,
    ) -> Dict[str, Any]:
        from .servicer import _py_world_object_to_pb

        pb_detections = [_py_world_object_to_pb(d) for d in detections]
        req = embodied_memory_pb2.SyncSceneObjectsRequest(
            scene_id=scene_id,
            detections=pb_detections,
            timestamp_sec=timestamp_sec,
            occlusion_radius=occlusion_radius,
        )
        resp = self.stub.SyncSceneObjects(req)
        return {
            "updated_objects": [d.to_dict() for d in detections],  # simplified
            "transitions": list(resp.transitions),
        }

    # -----------------------------------------------------------------------
    # Tri-Route Cognitive Search
    # -----------------------------------------------------------------------

    def cognitive_search(
        self,
        query: str,
        spatial_center: Optional[Vec3] = None,
        spatial_radius: Optional[float] = None,
        temporal_interval: Optional[TemporalInterval] = None,
        limit: int = 30,
    ) -> List[MemoryAtom]:
        from .servicer import _py_temporal_to_pb, _py_vec3_to_pb, _pb_atom_to_py

        req = embodied_memory_pb2.CognitiveSearchRequest(query=query, limit=limit)
        if spatial_center is not None:
            req.spatial_center.CopyFrom(_py_vec3_to_pb(spatial_center))
            req.spatial_radius = spatial_radius or 1.0
        if temporal_interval is not None:
            req.temporal_interval.CopyFrom(_py_temporal_to_pb(temporal_interval))
        resp = self.stub.CognitiveSearch(req)
        return [_pb_atom_to_py(a) for a in resp.atoms]

    def index_concept(
        self,
        memory_id: int,
        dimension: str,
        layer: int,
        concept_id: str,
        confidence: float = 1.0,
    ) -> bool:
        req = embodied_memory_pb2.IndexConceptRequest(
            memory_id=memory_id,
            dimension=dimension,
            layer=layer,
            concept_id=concept_id,
            confidence=confidence,
        )
        resp = self.stub.IndexConcept(req)
        return resp.success

    def add_experience_edge(
        self,
        source_memory_id: int,
        target_memory_id: int,
        edge_type: str,
        strength: float = 1.0,
    ) -> bool:
        req = embodied_memory_pb2.AddExperienceEdgeRequest(
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            edge_type=edge_type,
            strength=strength,
        )
        resp = self.stub.AddExperienceEdge(req)
        return resp.success

    def run_meditation(
        self,
        phases: Optional[List[str]] = None,
    ) -> dict:
        req = embodied_memory_pb2.RunMeditationRequest(
            phases=phases or ["consolidate", "crystallize", "extract"]
        )
        resp = self.stub.RunMeditation(req)
        return {
            "success": resp.success,
            "consolidated_count": resp.consolidated_count,
            "crystallized_count": resp.crystallized_count,
            "extracted_patterns": resp.extracted_patterns,
            "elapsed_sec": resp.elapsed_sec,
            "errors": list(resp.errors),
        }

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict:
        resp = self.stub.GetStats(embodied_memory_pb2.GetStatsRequest())
        import json

        return {k: json.loads(v) for k, v in resp.stats_json.items()}
