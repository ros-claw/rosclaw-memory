"""
Protocol definitions for v1.0 type-safe integration

Defines structural typing contracts (PEP 544) that v1.0 MemoryInterface
proxy methods can use as type annotations — without requiring powermem
as a hard dependency.

Usage in v1.0 (rosclaw-v1.0/src/rosclaw/memory/interface.py):
    try:
        from powermem.embodied.protocols import (
            WorldObjectLike, PoseLike, Vec3Like, PermanenceReportLike,
        )
    except ImportError:
        WorldObjectLike = Any  # type: ignore
        PoseLike = Any  # type: ignore
        Vec3Like = Any  # type: ignore
        PermanenceReportLike = Any  # type: ignore

    class MemoryInterface:
        def add_world_object(self, obj: WorldObjectLike) -> Optional[str]: ...
        def get_world_object(self, obj_id: str) -> Optional[WorldObjectLike]: ...

All protocols use structural subtyping (no inheritance required):
any object that has the right attributes/methods satisfies the protocol.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------

@runtime_checkable
class Vec3Like(Protocol):
    """3D vector — must expose x, y, z as floats."""
    x: float
    y: float
    z: float


@runtime_checkable
class QuaternionLike(Protocol):
    """Unit quaternion for 3D orientation."""
    w: float
    x: float
    y: float
    z: float


@runtime_checkable
class PoseLike(Protocol):
    """6-DOF pose: position + orientation."""
    position: Vec3Like
    orientation: QuaternionLike


# ---------------------------------------------------------------------------
# Temporal primitives
# ---------------------------------------------------------------------------

@runtime_checkable
class TemporalIntervalLike(Protocol):
    """Allen-style time interval."""
    start_sec: float
    end_sec: float
    frame_id: str


# ---------------------------------------------------------------------------
# World object
# ---------------------------------------------------------------------------

@runtime_checkable
class WorldObjectLike(Protocol):
    """World object — physical entity in the environment.

    Satisfied by powermem.embodied.types.WorldObject and any dataclass
    with the same structural fields.
    """
    obj_id: str
    obj_type: str
    name: str
    pose: PoseLike
    scene_id: Optional[str]
    state: str
    occlusion_status: str
    confidence: float
    last_seen_sec: float
    semantic_tags: List[str]

    def to_dict(self) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Permanence report (sync_scene_objects result)
# ---------------------------------------------------------------------------

@runtime_checkable
class PermanenceReportLike(Protocol):
    """Result of sync_scene_objects — describes state transitions."""
    transitions: List[Dict[str, Any]]
    added: List[str]
    updated: List[str]
    decayed: List[str]
    missing: List[str]

    def to_dict(self) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

# Waypoint: (Vec3Like position, float timestamp_sec)
WaypointLike = Tuple[Vec3Like, float]


# ---------------------------------------------------------------------------
# SpatialRelation
# ---------------------------------------------------------------------------

@runtime_checkable
class SpatialRelationLike(Protocol):
    """Spatial relation between two world objects."""
    subject_id: str
    object_id: str
    relation: str
    confidence: float


# ---------------------------------------------------------------------------
# SceneGraph (tuple return type for get_scene_graph)
# ---------------------------------------------------------------------------

@runtime_checkable
class SceneGraphLike(Protocol):
    """Scene graph — nodes are WorldObjects, edges are SpatialRelations."""
    scene_id: str
    objects: Dict[str, WorldObjectLike]
    relations: List[SpatialRelationLike]

    def get_objects(self) -> List[WorldObjectLike]: ...
    def get_relations(self) -> List[SpatialRelationLike]: ...


# ---------------------------------------------------------------------------
# MemoryAtom (cognitive_search / trajectory result element)
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryAtomLike(Protocol):
    """Memory atom — the fundamental unit of embodied memory."""
    memory_id: Optional[int]
    content: str
    spatial: Optional[Vec3Like]
    embodied_meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@runtime_checkable
class TelemetryLike(Protocol):
    """Telemetry interface for cache and query metrics."""
    enabled: bool

    def snapshot(self) -> Dict[str, Any]: ...
    def prometheus_metrics(self) -> str: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# EmbodiedMemory protocol (full interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbodiedMemoryLike(Protocol):
    """Full EmbodiedMemory interface — used by v1.0 proxy for type checking.

    Satisfied by powermem.embodied.EmbodiedMemory.
    """

    # World objects
    def add_world_object(self, obj: WorldObjectLike) -> str: ...
    def get_world_object(self, obj_id: str) -> Optional[WorldObjectLike]: ...
    def update_world_object_pose(
        self, obj_id: str, pose: PoseLike, state: Optional[str] = None
    ) -> bool: ...
    def search_world_objects(
        self, center: Vec3Like, radius: float, scene_id: Optional[str] = None
    ) -> List[WorldObjectLike]: ...

    # Scene graph
    def get_scene_graph(self, scene_id: str) -> SceneGraphLike: ...
    def auto_compute_relations(
        self, scene_id: str, spatial_tolerance: float = 0.01
    ) -> List[SpatialRelationLike]: ...
    def sync_scene_objects(
        self,
        scene_id: str,
        detections: List[WorldObjectLike],
        timestamp_sec: float,
    ) -> PermanenceReportLike: ...

    # Trajectories
    def record_trajectory(
        self, content: str, waypoints: Sequence[WaypointLike], **kwargs: Any
    ) -> int: ...
    def search_similar_trajectories(
        self,
        query_waypoints: Sequence[WaypointLike],
        top_k: int = 10,
        max_dtw_distance: Optional[float] = None,
    ) -> List[Tuple[MemoryAtomLike, float]]: ...

    # Cognitive search
    def cognitive_search(
        self,
        query: str,
        spatial_center: Optional[Vec3Like] = None,
        spatial_radius: Optional[float] = None,
        temporal_interval: Optional[TemporalIntervalLike] = None,
        limit: int = 10,
    ) -> List[MemoryAtomLike]: ...

    # Meditation
    def run_meditation(
        self, phases: Optional[List[str]] = None
    ) -> Dict[str, Any]: ...

    # Telemetry
    def get_telemetry(self) -> Optional[Dict[str, Any]]: ...
    def prometheus_metrics(self) -> str: ...

    # Daemon
    def start_background_daemon(self, config: Optional[Dict[str, Any]] = None) -> None: ...
    def stop_background_daemon(self, timeout: float = 5.0) -> None: ...
    def get_daemon_stats(self) -> Optional[Dict[str, Any]]: ...
