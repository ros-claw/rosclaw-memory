"""
gRPC Servicer for EmbodiedMemory.

Wraps an EmbodiedMemory instance and exposes it via gRPC.
All DB-mutating operations are protected by an RLock for thread safety.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import grpc

from powermem.embodied.proto import embodied_memory_pb2
from powermem.embodied.proto import embodied_memory_pb2_grpc

from ..embodied_memory import EmbodiedMemory
from ..memory_atom import MemoryAtom
from ..types import IntervalRelation, MemoryAction, Modality, Pose, Quaternion, SpatialRelation, TemporalInterval, Vec3, WorldObject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversions: protobuf <-> Python
# ---------------------------------------------------------------------------

def _pb_vec3_to_py(v: embodied_memory_pb2.Vec3) -> Vec3:
    return Vec3(v.x, v.y, v.z)


def _py_vec3_to_pb(v: Vec3) -> embodied_memory_pb2.Vec3:
    return embodied_memory_pb2.Vec3(x=v.x, y=v.y, z=v.z)


def _pb_temporal_to_py(t: embodied_memory_pb2.TemporalInterval) -> TemporalInterval:
    return TemporalInterval(start_sec=t.start_sec, end_sec=t.end_sec, frame_id=t.frame_id or "wall_clock")


def _py_temporal_to_pb(t: TemporalInterval) -> embodied_memory_pb2.TemporalInterval:
    return embodied_memory_pb2.TemporalInterval(
        start_sec=t.start_sec, end_sec=t.end_sec, frame_id=t.frame_id
    )


def _pb_action_to_py(action_str: str) -> MemoryAction:
    try:
        return MemoryAction(action_str)
    except ValueError:
        return MemoryAction.OBSERVE


def _pb_atom_to_py(payload: embodied_memory_pb2.MemoryAtomPayload) -> MemoryAtom:
    meta: Dict[str, Any] = {}
    for k, v in payload.embodied_meta_json.items():
        try:
            meta[k] = json.loads(v)
        except json.JSONDecodeError:
            meta[k] = v

    atom = MemoryAtom(
        content=payload.content,
        memory_id=payload.memory_id if payload.HasField("memory_id") else None,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
        run_id=payload.run_id,
        spatial=_pb_vec3_to_py(payload.spatial) if payload.HasField("spatial") else None,
        spatial_frame_id=payload.spatial_frame_id or "world",
        temporal=_pb_temporal_to_py(payload.temporal) if payload.HasField("temporal") else None,
        action=_pb_action_to_py(payload.action),
        prediction_error=payload.prediction_error,
        causal_parents=list(payload.causal_parents),
        embodied_meta=meta,
    )
    return atom


def _py_atom_to_pb(atom: MemoryAtom) -> embodied_memory_pb2.MemoryAtomPayload:
    meta_json = {}
    for k, v in atom.embodied_meta.items():
        if isinstance(v, (dict, list)):
            meta_json[k] = json.dumps(v)
        else:
            meta_json[k] = str(v)

    payload = embodied_memory_pb2.MemoryAtomPayload(
        content=atom.content,
        user_id=atom.user_id,
        agent_id=atom.agent_id,
        run_id=atom.run_id,
        spatial_frame_id=atom.spatial_frame_id,
        action=atom.action.value,
        prediction_error=atom.prediction_error,
        causal_parents=atom.causal_parents,
    )
    if atom.memory_id is not None:
        payload.memory_id = atom.memory_id
    if atom.spatial is not None:
        payload.spatial.CopyFrom(_py_vec3_to_pb(atom.spatial))
    if atom.temporal is not None:
        payload.temporal.CopyFrom(_py_temporal_to_pb(atom.temporal))
    payload.embodied_meta_json.update(meta_json)
    return payload


def _py_relation_to_pb(relation: Optional[IntervalRelation]) -> str:
    return relation.value if relation else ""


def _pb_relation_to_py(relation_str: str) -> Optional[IntervalRelation]:
    if not relation_str:
        return None
    try:
        return IntervalRelation(relation_str)
    except ValueError:
        return None


def _pb_pose_to_py(pb_pose: embodied_memory_pb2.Pose) -> Pose:
    return Pose(
        position=_pb_vec3_to_py(pb_pose.position),
        orientation=Quaternion(
            w=pb_pose.orientation.w,
            x=pb_pose.orientation.x,
            y=pb_pose.orientation.y,
            z=pb_pose.orientation.z,
        ),
    )


def _py_pose_to_pb(pose: Pose) -> embodied_memory_pb2.Pose:
    return embodied_memory_pb2.Pose(
        position=_py_vec3_to_pb(pose.position),
        orientation=embodied_memory_pb2.Quaternion(
            w=pose.orientation.w,
            x=pose.orientation.x,
            y=pose.orientation.y,
            z=pose.orientation.z,
        ),
    )


def _pb_world_object_to_py(pb: embodied_memory_pb2.WorldObjectPayload) -> WorldObject:
    size = tuple(pb.size) if pb.size else None
    color = tuple(pb.color) if pb.color else None
    return WorldObject(
        obj_id=pb.obj_id,
        obj_type=pb.obj_type or "box",
        name=pb.name,
        pose=_pb_pose_to_py(pb.pose) if pb.HasField("pose") else Pose(),
        size=size,
        color=color,
        mesh_path=pb.mesh_path if pb.mesh_path else None,
        physics_props=json.loads(pb.physics_props_json) if pb.physics_props_json else {},
        semantic_tags=list(pb.semantic_tags),
        scene_id=pb.scene_id if pb.scene_id else None,
        parent_obj_id=pb.parent_obj_id if pb.parent_obj_id else None,
        state=pb.state or "present",
        memory_id=pb.memory_id if pb.memory_id else None,
    )


def _py_world_object_to_pb(obj: WorldObject) -> embodied_memory_pb2.WorldObjectPayload:
    pb = embodied_memory_pb2.WorldObjectPayload(
        obj_id=obj.obj_id,
        obj_type=obj.obj_type,
        name=obj.name,
        state=obj.state,
        semantic_tags=obj.semantic_tags,
    )
    if obj.pose is not None:
        pb.pose.CopyFrom(_py_pose_to_pb(obj.pose))
    if obj.size is not None:
        pb.size.extend(obj.size)
    if obj.color is not None:
        pb.color.extend(obj.color)
    if obj.mesh_path is not None:
        pb.mesh_path = obj.mesh_path
    if obj.physics_props:
        pb.physics_props_json = json.dumps(obj.physics_props)
    if obj.scene_id is not None:
        pb.scene_id = obj.scene_id
    if obj.parent_obj_id is not None:
        pb.parent_obj_id = obj.parent_obj_id
    if obj.memory_id is not None:
        pb.memory_id = obj.memory_id
    return pb


def _py_spatial_relation_to_pb(rel: SpatialRelation) -> embodied_memory_pb2.SpatialRelationPayload:
    return embodied_memory_pb2.SpatialRelationPayload(
        subject_id=rel.subject_id,
        object_id=rel.object_id,
        relation=rel.relation,
        confidence=rel.confidence,
    )


def _pb_spatial_relation_to_py(pb: embodied_memory_pb2.SpatialRelationPayload) -> SpatialRelation:
    return SpatialRelation(
        subject_id=pb.subject_id,
        object_id=pb.object_id,
        relation=pb.relation,
        confidence=pb.confidence,
    )


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------

class EmbodiedMemoryServicer(embodied_memory_pb2_grpc.EmbodiedMemoryServiceServicer):
    """gRPC servicer wrapping EmbodiedMemory."""

    def __init__(self, embodied_memory: EmbodiedMemory):
        self.em = embodied_memory
        self._lock = threading.RLock()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _handle(self, context, fn):
        """Execute fn under lock and convert exceptions to gRPC statuses."""
        try:
            with self._lock:
                return fn()
        except grpc.RpcError:
            raise
        except Exception as e:
            logger.exception("gRPC handler error")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    # -----------------------------------------------------------------------
    # Atom CRUD
    # -----------------------------------------------------------------------

    def AddAtom(self, request, context):
        def _do():
            atom = _pb_atom_to_py(request.atom)
            mid = self.em.add_atom(atom, infer=request.infer)
            return embodied_memory_pb2.AddAtomResponse(memory_id=mid, success=True)
        return self._handle(context, _do)

    def GetAtom(self, request, context):
        def _do():
            atom = self.em.get_atom(request.memory_id)
            if atom is None:
                return embodied_memory_pb2.GetAtomResponse(found=False)
            return embodied_memory_pb2.GetAtomResponse(
                atom=_py_atom_to_pb(atom), found=True
            )
        return self._handle(context, _do)

    def DeleteAtom(self, request, context):
        def _do():
            ok = self.em.delete_atom(request.memory_id)
            return embodied_memory_pb2.DeleteAtomResponse(deleted=ok)
        return self._handle(context, _do)

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def Search(self, request, context):
        def _do():
            spatial_center = (
                _pb_vec3_to_py(request.spatial_center)
                if request.HasField("spatial_center")
                else None
            )
            spatial_radius = (
                request.spatial_radius if request.HasField("spatial_radius") else None
            )
            temporal_interval = (
                _pb_temporal_to_py(request.temporal_interval)
                if request.HasField("temporal_interval")
                else None
            )
            relation = _pb_relation_to_py(request.temporal_relation)
            atoms = self.em.search(
                query=request.query,
                spatial_center=spatial_center,
                spatial_radius=spatial_radius,
                temporal_interval=temporal_interval,
                temporal_relation=relation,
                limit=request.limit or 30,
            )
            return embodied_memory_pb2.SearchResponse(
                atoms=[_py_atom_to_pb(a) for a in atoms]
            )
        return self._handle(context, _do)

    def SearchNear(self, request, context):
        def _do():
            center = _pb_vec3_to_py(request.center)
            atoms = self.em.search_near(
                center=center,
                radius=request.radius,
                frame_id=request.frame_id or "world",
                limit=request.limit or 30,
            )
            return embodied_memory_pb2.SearchNearResponse(
                atoms=[_py_atom_to_pb(a) for a in atoms]
            )
        return self._handle(context, _do)

    def SearchTemporal(self, request, context):
        def _do():
            interval = _pb_temporal_to_py(request.interval)
            relation = _pb_relation_to_py(request.relation)
            atoms = self.em.search_temporal(
                interval=interval,
                relation=relation,
                frame_id=request.frame_id or None,
                limit=request.limit or 30,
            )
            return embodied_memory_pb2.SearchTemporalResponse(
                atoms=[_py_atom_to_pb(a) for a in atoms]
            )
        return self._handle(context, _do)

    # -----------------------------------------------------------------------
    # Trajectory
    # -----------------------------------------------------------------------

    def RecordTrajectory(self, request, context):
        def _do():
            waypoints: List[Tuple[Vec3, float]] = []
            for wp in request.waypoints:
                waypoints.append((_pb_vec3_to_py(wp.position), wp.timestamp_sec))
            mid = self.em.record_trajectory(request.content, waypoints)
            return embodied_memory_pb2.RecordTrajectoryResponse(
                memory_id=mid, success=True
            )
        return self._handle(context, _do)

    def SearchSimilarTrajectories(self, request, context):
        def _do():
            query_waypoints: List[Tuple[Vec3, float]] = []
            for wp in request.query_waypoints:
                query_waypoints.append((_pb_vec3_to_py(wp.position), wp.timestamp_sec))

            spatial_center = (
                _pb_vec3_to_py(request.spatial_center)
                if request.HasField("spatial_center")
                else None
            )
            spatial_radius = (
                request.spatial_radius if request.HasField("spatial_radius") else None
            )
            temporal_interval = (
                _pb_temporal_to_py(request.temporal_interval)
                if request.HasField("temporal_interval")
                else None
            )
            max_dtw = (
                request.max_dtw_distance
                if request.HasField("max_dtw_distance")
                else None
            )

            results = self.em.search_similar_trajectories(
                query_waypoints=query_waypoints,
                spatial_center=spatial_center,
                spatial_radius=spatial_radius,
                temporal_interval=temporal_interval,
                top_k=request.top_k or 10,
                max_dtw_distance=max_dtw,
            )
            return embodied_memory_pb2.SearchSimilarTrajectoriesResponse(
                results=[
                    embodied_memory_pb2.TrajectorySimilarityResult(
                        atom=_py_atom_to_pb(atom), dtw_distance=dtw
                    )
                    for atom, dtw in results
                ]
            )
        return self._handle(context, _do)

    # -----------------------------------------------------------------------
    # Ingest
    # -----------------------------------------------------------------------

    def IngestSensorFrame(self, request, context):
        def _do():
            from ..ingest_pipeline import SensorFrame
            from ..types import Pose

            pb_frame = request.frame
            frame = SensorFrame(
                modality=Modality(pb_frame.modality),
                timestamp_sec=pb_frame.timestamp_sec,
                data=list(pb_frame.data),
                sensor_pose=Pose(
                    position=_pb_vec3_to_py(pb_frame.sensor_position)
                    if pb_frame.HasField("sensor_position")
                    else Vec3(0, 0, 0)
                ),
                frame_id=pb_frame.frame_id or "world",
            )
            mid = self.em.ingest(frame, content=request.content or None)
            resp = embodied_memory_pb2.IngestSensorFrameResponse(stored=mid is not None)
            if mid is not None:
                resp.memory_id = mid
            return resp
        return self._handle(context, _do)

    # -----------------------------------------------------------------------
    # Model / Collision
    # -----------------------------------------------------------------------

    def SaveModel(self, request, context):
        def _do():
            from ..parsers import parse_model

            result = parse_model(request.model_xml)
            model_id = self.em.save_model(
                result,
                model_id=request.model_id or None,
                model_type=request.model_type or "robot",
            )
            return embodied_memory_pb2.SaveModelResponse(
                model_id=model_id, success=True
            )
        return self._handle(context, _do)

    def CheckSelfCollision(self, request, context):
        def _do():
            pairs = self.em.check_self_collision(request.model_id)
            # pairs may be CollisionBody tuples or dicts depending on implementation
            pb_pairs = []
            for p in pairs:
                if isinstance(p, dict):
                    pb_pairs.append(
                        embodied_memory_pb2.CollisionPair(
                            link_a=p.get("link_a", ""),
                            link_b=p.get("link_b", ""),
                            distance=p.get("distance", 0.0),
                        )
                    )
                else:
                    # fallback: stringify
                    pb_pairs.append(
                        embodied_memory_pb2.CollisionPair(
                            link_a=str(getattr(p, "link_a", "")),
                            link_b=str(getattr(p, "link_b", "")),
                            distance=float(getattr(p, "distance", 0.0)),
                        )
                    )
            return embodied_memory_pb2.CheckSelfCollisionResponse(pairs=pb_pairs)
        return self._handle(context, _do)

    # -----------------------------------------------------------------------
    # Causal graph
    # -----------------------------------------------------------------------

    def GetCauses(self, request, context):
        def _do():
            atoms = self.em.get_causes(request.memory_id, limit=request.limit or 10)
            return embodied_memory_pb2.GetCausesResponse(
                atoms=[_py_atom_to_pb(a) for a in atoms]
            )
        return self._handle(context, _do)

    def GetEffects(self, request, context):
        def _do():
            atoms = self.em.get_effects(request.memory_id, limit=request.limit or 10)
            return embodied_memory_pb2.GetEffectsResponse(
                atoms=[_py_atom_to_pb(a) for a in atoms]
            )
        return self._handle(context, _do)

    # -----------------------------------------------------------------------
    # World Objects
    # -----------------------------------------------------------------------

    def AddWorldObject(self, request, context):
        def _do():
            obj = _pb_world_object_to_py(request.obj)
            obj_id = self.em.add_world_object(obj)
            return embodied_memory_pb2.AddWorldObjectResponse(obj_id=obj_id, success=True)
        return self._handle(context, _do)

    def GetWorldObject(self, request, context):
        def _do():
            obj = self.em.get_world_object(request.obj_id)
            if obj is None:
                return embodied_memory_pb2.GetWorldObjectResponse(found=False)
            return embodied_memory_pb2.GetWorldObjectResponse(
                obj=_py_world_object_to_pb(obj), found=True
            )
        return self._handle(context, _do)

    def UpdateWorldObjectPose(self, request, context):
        def _do():
            pose = _pb_pose_to_py(request.pose) if request.HasField("pose") else None
            state = request.state if request.state else None
            ok = self.em.update_world_object_pose(
                request.obj_id, pose=pose, state=state
            ) if pose else False
            return embodied_memory_pb2.UpdateWorldObjectPoseResponse(success=ok)
        return self._handle(context, _do)

    def SearchWorldObjects(self, request, context):
        def _do():
            center = _pb_vec3_to_py(request.center)
            objects = self.em.search_world_objects(
                center=center,
                radius=request.radius,
                obj_type=request.obj_type or None,
                scene_id=request.scene_id or None,
                limit=request.limit or 30,
            )
            return embodied_memory_pb2.SearchWorldObjectsResponse(
                objects=[_py_world_object_to_pb(o) for o in objects]
            )
        return self._handle(context, _do)

    def GetSceneGraph(self, request, context):
        def _do():
            sg = self.em.get_scene_graph(request.scene_id)
            return embodied_memory_pb2.GetSceneGraphResponse(
                objects=[_py_world_object_to_pb(o) for o in sg.get_objects()],
                relations=[_py_spatial_relation_to_pb(r) for r in sg._relations],
            )
        return self._handle(context, _do)

    def ComputeRelations(self, request, context):
        def _do():
            relations = self.em.auto_compute_relations(
                request.scene_id,
                spatial_tolerance=request.spatial_tolerance or 0.01,
            )
            return embodied_memory_pb2.ComputeRelationsResponse(
                relations=[_py_spatial_relation_to_pb(r) for r in relations]
            )
        return self._handle(context, _do)

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def GetStats(self, request, context):
        def _do():
            stats = self.em.stats()
            stats_json = {}
            for k, v in stats.items():
                stats_json[k] = json.dumps(v) if not isinstance(v, str) else v
            return embodied_memory_pb2.GetStatsResponse(stats_json=stats_json)
        return self._handle(context, _do)
