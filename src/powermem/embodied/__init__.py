"""
ROSClaw-Memory Embodied Intelligence Package

Phase 0 核心模块入口。所有具身智能扩展通过本包暴露。

使用示例：
    from powermem.embodied import MemoryAtom, IngestPipeline, EmbodiedIntelligencePlugin
    from powermem.embodied.types import Pose, Vec3, Modality

    atom = MemoryAtom.from_observation(
        content="red cup on table",
        sensor_pose=Pose(position=Vec3(1.0, 2.0, 0.5)),
        modality=Modality.RGB,
    )
"""

from __future__ import annotations

# 类型系统
from .types import (
    AffectiveTag,
    IntervalRelation,
    MemoryAction,
    Modality,
    PerceptualSnapshot,
    PhysicalInvariant,
    Pose,
    Quaternion,
    TemporalInterval,
    UncertaintyEstimate,
    UncertaintyType,
    Vec3,
    WorldObject,
)

# 核心数据模型
from .embodied_memory import EmbodiedMemory
from .memory_atom import MemoryAtom

# 索引
from .spatial_index import SpatialIndex, VoxelHash
from .temporal_index import TemporalIndex

# 物理模型
from .physical_model import (
    DHParameter,
    JointLimit,
    PhysicalConstraint,
    RobotDynamics,
)

# 正运动学
from .kinematics import (
    Transform,
    dh_to_transform,
    forward_kinematics,
    forward_kinematics_poses,
    transform_collision_body,
    transform_collision_bodies,
)

# 不确定性与预测编码
from .uncertainty import (
    ExplorationSuggestion,
    SpatialUncertainty,
    fuse_uncertainties_ci,
    fuse_uncertainties_kalman,
    propagate_spatial_uncertainty,
    suggest_exploration,
)
from .surprisal_gate import (
    LinearPredictor,
    Predictor,
    SurprisalGate,
    ZeroOrderHoldPredictor,
)

# 插件与管线
from .embodied_plugin import EmbodiedIntelligencePlugin
from .ingest_pipeline import (
    IngestPipeline,
    SensorFrame,
    get_feature_extractor,
    register_feature_extractor,
)

# 解析器
from .parsers import parse_model, get_parser, ParseResult

# 物理模型存储
from .model_store import ModelStore, StoredModel

# 碰撞检测
from .collision import (
    AABB,
    Capsule,
    CollisionBody,
    CollisionChecker,
    Sphere,
    build_collision_bodies,
)

# Schema
from .schema import initialize_embodied_schema

# 后台维护
from .background_daemon import BackgroundDaemon, DaemonConfig, DaemonStats

# 遥测
from .telemetry import MemoryTelemetry

# Protocol 类型定义（v1.0 集成用）
from .protocols import (
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

__all__ = [
    # types
    "AffectiveTag",
    "IntervalRelation",
    "MemoryAction",
    "Modality",
    "PerceptualSnapshot",
    "PhysicalInvariant",
    "Pose",
    "Quaternion",
    "TemporalInterval",
    "UncertaintyEstimate",
    "UncertaintyType",
    "Vec3",
    "WorldObject",
    # core
    "MemoryAtom",
    # spatial / temporal
    "SpatialIndex",
    "VoxelHash",
    "TemporalIndex",
    # physical
    "DHParameter",
    "JointLimit",
    "PhysicalConstraint",
    "RobotDynamics",
    # kinematics
    "Transform",
    "dh_to_transform",
    "forward_kinematics",
    "forward_kinematics_poses",
    "transform_collision_body",
    "transform_collision_bodies",
    # uncertainty
    "ExplorationSuggestion",
    "SpatialUncertainty",
    "fuse_uncertainties_ci",
    "fuse_uncertainties_kalman",
    "propagate_spatial_uncertainty",
    "suggest_exploration",
    # surprisal
    "LinearPredictor",
    "Predictor",
    "SurprisalGate",
    "ZeroOrderHoldPredictor",
    # plugin / pipeline
    "EmbodiedIntelligencePlugin",
    "IngestPipeline",
    "SensorFrame",
    "get_feature_extractor",
    "register_feature_extractor",
    # parsers
    "parse_model",
    "get_parser",
    "ParseResult",
    # model store
    "ModelStore",
    "StoredModel",
    # collision
    "Sphere",
    "Capsule",
    "AABB",
    "CollisionBody",
    "CollisionChecker",
    "build_collision_bodies",
    # schema
    "initialize_embodied_schema",
    # background daemon
    "BackgroundDaemon",
    "DaemonConfig",
    "DaemonStats",
    # telemetry
    "MemoryTelemetry",
    # protocols (v1.0 type-safe integration)
    "EmbodiedMemoryLike",
    "MemoryAtomLike",
    "PermanenceReportLike",
    "PoseLike",
    "QuaternionLike",
    "SceneGraphLike",
    "SpatialRelationLike",
    "TelemetryLike",
    "TemporalIntervalLike",
    "Vec3Like",
    "WorldObjectLike",
]
