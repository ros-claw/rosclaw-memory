"""
具身智能核心类型定义 — 物理 AI 记忆系统的原子类型

所有类型均为不可变值对象（frozen dataclass），支持 JSON 序列化，
可直接嵌入 PowerMem 的 metadata payload。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 基础几何类型
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Vec3:
    """三维向量 / 点坐标"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance_to(self, other: Vec3) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Vec3:
        return cls(float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("z", 0.0)))


@dataclass(frozen=True, slots=True)
class Quaternion:
    """四元数旋转表示"""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"w": self.w, "x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Quaternion:
        return cls(
            float(d.get("w", 1.0)),
            float(d.get("x", 0.0)),
            float(d.get("y", 0.0)),
            float(d.get("z", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class Pose:
    """6-DOF 位姿：位置 + 旋转"""
    position: Vec3 = field(default_factory=Vec3)
    orientation: Quaternion = field(default_factory=Quaternion)

    def to_dict(self) -> Dict[str, Any]:
        return {"position": self.position.to_dict(), "orientation": self.orientation.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Pose:
        return cls(
            Vec3.from_dict(d.get("position", {})),
            Quaternion.from_dict(d.get("orientation", {})),
        )


# ---------------------------------------------------------------------------
# 时间类型（Allen Interval Algebra 子集）
# ---------------------------------------------------------------------------

class IntervalRelation(Enum):
    """Allen 区间代数：13 种拓扑关系的核心子集"""
    BEFORE = "before"          # A 在 B 之前结束
    MEETS = "meets"            # A 结束 = B 开始
    OVERLAPS = "overlaps"      # A 开始 < B 开始，A 结束 > B 开始，A 结束 < B 结束
    DURING = "during"          # A 完全在 B 内
    STARTS = "starts"          # A 开始 = B 开始，A 结束 < B 结束
    FINISHES = "finishes"      # A 结束 = B 结束，A 开始 > B 开始
    EQUALS = "equals"          # A = B
    CONTAINS = "contains"      # B during A
    STARTED_BY = "started_by"  # B starts A
    FINISHED_BY = "finished_by" # B finishes A
    OVERLAPPED_BY = "overlapped_by"
    MET_BY = "met_by"
    AFTER = "after"


@dataclass(frozen=True, slots=True)
class TemporalInterval:
    """时间区间 — 物理 AI 的一等公民"""
    start_sec: float
    end_sec: float
    frame_id: str = "wall_clock"  # 时间参考系：wall_clock, sim_time, episode_N

    def __post_init__(self):
        if self.end_sec < self.start_sec:
            object.__setattr__(self, "end_sec", self.start_sec)

    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def relation_to(self, other: TemporalInterval) -> IntervalRelation:
        """计算与另一个区间的 Allen 关系"""
        a1, a2 = self.start_sec, self.end_sec
        b1, b2 = other.start_sec, other.end_sec
        eps = 1e-6

        if abs(a1 - b1) < eps and abs(a2 - b2) < eps:
            return IntervalRelation.EQUALS
        if abs(a2 - b1) < eps and a1 < b1:
            return IntervalRelation.MEETS
        if abs(a1 - b2) < eps and a2 > b2:
            return IntervalRelation.MET_BY
        if a2 < b1 - eps:
            return IntervalRelation.BEFORE
        if a1 > b2 + eps:
            return IntervalRelation.AFTER
        if abs(a1 - b1) < eps and a2 < b2:
            return IntervalRelation.STARTS
        if abs(a1 - b1) < eps and a2 > b2:
            return IntervalRelation.STARTED_BY
        if abs(a2 - b2) < eps and a1 > b1:
            return IntervalRelation.FINISHES
        if abs(a2 - b2) < eps and a1 < b1:
            return IntervalRelation.FINISHED_BY
        if a1 > b1 + eps and a2 < b2 - eps:
            return IntervalRelation.DURING
        if a1 < b1 - eps and a2 > b2 + eps:
            return IntervalRelation.CONTAINS
        if a1 < b1 - eps and a2 > b1 + eps and a2 < b2 - eps:
            return IntervalRelation.OVERLAPS
        return IntervalRelation.OVERLAPPED_BY

    def to_dict(self) -> Dict[str, Any]:
        return {"start_sec": self.start_sec, "end_sec": self.end_sec, "frame_id": self.frame_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TemporalInterval:
        return cls(float(d["start_sec"]), float(d["end_sec"]), str(d.get("frame_id", "wall_clock")))


# ---------------------------------------------------------------------------
# 不确定性类型
# ---------------------------------------------------------------------------

class UncertaintyType(Enum):
    """不确定性来源分类"""
    ALEATORIC = "aleatoric"      # 偶然不确定性：传感器噪声，不可学习
    EPISTEMIC = "epistemic"      # 认知不确定性：知识盲区，可通过探索降低
    MODEL = "model"              # 模型不确定性：近似误差
    OUT_OF_DISTRIBUTION = "ood"  # 分布外


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate:
    """不确定性估计 — 每个感知测量必须携带"""
    type: UncertaintyType = UncertaintyType.ALEATORIC
    # 主不确定性度量（协方差矩阵的特征值或标量 std）
    std: float = 0.0
    # 3x3 协方差矩阵（扁平化存储），可选
    covariance: Tuple[float, ...] = field(default_factory=tuple)
    # 熵或置信度得分 [0, 1]
    confidence: float = 1.0
    # 采样次数（MC Dropout 等）
    sample_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "std": self.std,
            "covariance": list(self.covariance) if self.covariance else [],
            "confidence": self.confidence,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> UncertaintyEstimate:
        return cls(
            type=UncertaintyType(d.get("type", "aleatoric")),
            std=float(d.get("std", 0.0)),
            covariance=tuple(d.get("covariance", [])),
            confidence=float(d.get("confidence", 1.0)),
            sample_count=int(d.get("sample_count", 1)),
        )


# ---------------------------------------------------------------------------
# 感知与物理类型
# ---------------------------------------------------------------------------

class Modality(Enum):
    """传感器模态"""
    RGB = "rgb"
    DEPTH = "depth"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    JOINT_STATE = "joint_state"
    TACTILE = "tactile"
    AUDIO = "audio"
    PROPRIOCEPTION = "proprioception"  # 本体感知：关节角、速度
    SEMANTIC = "semantic"              # 语义分割
    TEXT = "text"                      # 语言描述
    FUSION = "fusion"                  # 多模态融合


@dataclass(frozen=True, slots=True)
class PerceptualSnapshot:
    """感知快照 — 传感器的原始或特征级测量"""
    modality: Modality = Modality.RGB
    # 特征向量（替代原始像素/点云）
    feature_vec: Tuple[float, ...] = field(default_factory=tuple)
    # 原始数据的哈希（用于去重和溯源）
    raw_data_hash: str = ""
    # 传感器外参：传感器坐标系 -> 世界坐标系
    sensor_pose: Pose = field(default_factory=Pose)
    # 测量不确定性
    uncertainty: UncertaintyEstimate = field(default_factory=UncertaintyEstimate)
    # 元数据：分辨率、帧率、曝光等
    sensor_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "feature_vec": list(self.feature_vec) if self.feature_vec else [],
            "raw_data_hash": self.raw_data_hash,
            "sensor_pose": self.sensor_pose.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "sensor_meta": self.sensor_meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PerceptualSnapshot:
        return cls(
            modality=Modality(d.get("modality", "rgb")),
            feature_vec=tuple(d.get("feature_vec", [])),
            raw_data_hash=str(d.get("raw_data_hash", "")),
            sensor_pose=Pose.from_dict(d.get("sensor_pose", {})),
            uncertainty=UncertaintyEstimate.from_dict(d.get("uncertainty", {})),
            sensor_meta=dict(d.get("sensor_meta", {})),
        )


@dataclass(frozen=True, slots=True)
class PhysicalInvariant:
    """物理不变量 — 世界骨架（确定性部分）

    与 PerceptualSnapshot 相对：PhysicalInvariant 描述的是物理定律保证的、
    不随观测变化的部分。例如物体的质量、关节的 DH 参数、摩擦系数等。
    """
    # 物体/连杆标识
    entity_id: str = ""
    # 物理属性
    mass_kg: Optional[float] = None
    center_of_mass: Vec3 = field(default_factory=Vec3)
    # 惯性矩阵（3x3 扁平化）
    inertia_matrix: Tuple[float, ...] = field(default_factory=tuple)
    # DH 参数或 URDF 关节定义（字典形式，避免依赖 ROS）
    kinematic_params: Dict[str, Any] = field(default_factory=dict)
    # 动力学参数
    dynamics_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "mass_kg": self.mass_kg,
            "center_of_mass": self.center_of_mass.to_dict(),
            "inertia_matrix": list(self.inertia_matrix) if self.inertia_matrix else [],
            "kinematic_params": self.kinematic_params,
            "dynamics_params": self.dynamics_params,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PhysicalInvariant:
        return cls(
            entity_id=str(d.get("entity_id", "")),
            mass_kg=float(d["mass_kg"]) if d.get("mass_kg") is not None else None,
            center_of_mass=Vec3.from_dict(d.get("center_of_mass", {})),
            inertia_matrix=tuple(d.get("inertia_matrix", [])),
            kinematic_params=dict(d.get("kinematic_params", {})),
            dynamics_params=dict(d.get("dynamics_params", {})),
        )


# ---------------------------------------------------------------------------
# 情感/显著性标记
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AffectiveTag:
    """情感标记 — 类杏仁核功能

    不是情绪本身，而是对记忆显著性的快速评估。
    """
    # 显著性得分 [0, 1]，1.0 = 极端重要（碰撞、故障、意外发现）
    salience: float = 0.5
    # 效价：-1（负面）到 +1（正面）
    valence: float = 0.0
    # 唤醒度：0（平静）到 1（高度警觉）
    arousal: float = 0.0
    # 触发源
    trigger: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"salience": self.salience, "valence": self.valence, "arousal": self.arousal, "trigger": self.trigger}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AffectiveTag:
        return cls(
            salience=float(d.get("salience", 0.5)),
            valence=float(d.get("valence", 0.0)),
            arousal=float(d.get("arousal", 0.0)),
            trigger=str(d.get("trigger", "")),
        )


# ---------------------------------------------------------------------------
# 记忆动作类型
# ---------------------------------------------------------------------------

class MemoryAction(Enum):
    """记忆生命周期动作"""
    OBSERVE = "observe"      # 被动观测
    ACT = "act"              # 执行动作
    PREDICT = "predict"      # 内部预测
    CORRECT = "correct"      # 预测校正
    REFLECT = "reflect"      # 反思/蒸馏
    FORGET = "forget"        # 主动遗忘


# ---------------------------------------------------------------------------
# 世界对象与空间关系
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WorldObject:
    """世界对象 —— 物理环境中可感知、可交互的实体"""
    obj_id: str
    obj_type: str = "box"           # box | sphere | cylinder | capsule | mesh
    name: str = ""
    pose: Pose = field(default_factory=Pose)
    size: Optional[Tuple[float, ...]] = None       # [w, h, d] or [radius]
    color: Optional[Tuple[float, ...]] = None      # [r, g, b, a]
    mesh_path: Optional[str] = None
    physics_props: Dict[str, Any] = field(default_factory=dict)   # mass, friction, restitution
    semantic_tags: List[str] = field(default_factory=list)        # ["graspable", "furniture"]
    scene_id: Optional[str] = None
    parent_obj_id: Optional[str] = None            # scene graph parent
    state: str = "present"                          # present | moved | removed | occluded
    memory_id: Optional[int] = None                # link to embodied_memories
    # 对象恒存（遮挡感知）
    occlusion_status: str = "visible"               # visible | occluded | missing
    last_confirmed_position: Optional[Vec3] = None  # 最后一次确认的位置
    confidence: float = 1.0                         # 存在置信度 [0, 1]
    last_seen_sec: float = 0.0                     # 最后一次检测到的时间戳

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obj_id": self.obj_id,
            "obj_type": self.obj_type,
            "name": self.name,
            "pose": self.pose.to_dict(),
            "size": list(self.size) if self.size else [],
            "color": list(self.color) if self.color else [],
            "mesh_path": self.mesh_path,
            "physics_props": self.physics_props,
            "semantic_tags": self.semantic_tags,
            "scene_id": self.scene_id,
            "parent_obj_id": self.parent_obj_id,
            "state": self.state,
            "memory_id": self.memory_id,
            "occlusion_status": self.occlusion_status,
            "last_confirmed_position": self.last_confirmed_position.to_dict() if self.last_confirmed_position else None,
            "confidence": self.confidence,
            "last_seen_sec": self.last_seen_sec,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> WorldObject:
        size = d.get("size", [])
        color = d.get("color", [])
        return cls(
            obj_id=str(d["obj_id"]),
            obj_type=str(d.get("obj_type", "box")),
            name=str(d.get("name", "")),
            pose=Pose.from_dict(d.get("pose", {})),
            size=tuple(size) if size else None,
            color=tuple(color) if color else None,
            mesh_path=d.get("mesh_path") if d.get("mesh_path") else None,
            physics_props=dict(d.get("physics_props", {})),
            semantic_tags=list(d.get("semantic_tags", [])),
            scene_id=d.get("scene_id") if d.get("scene_id") else None,
            parent_obj_id=d.get("parent_obj_id") if d.get("parent_obj_id") else None,
            state=str(d.get("state", "present")),
            memory_id=int(d["memory_id"]) if d.get("memory_id") is not None else None,
            occlusion_status=str(d.get("occlusion_status", "visible")),
            last_confirmed_position=Vec3.from_dict(d["last_confirmed_position"]) if d.get("last_confirmed_position") else None,
            confidence=float(d.get("confidence", 1.0)),
            last_seen_sec=float(d.get("last_seen_sec", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class SpatialRelation:
    """空间关系 —— 两个世界对象之间的拓扑关系"""
    subject_id: str
    object_id: str
    relation: str = "next_to"      # on | in | next_to | above | below | touching | contained_by
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "relation": self.relation,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SpatialRelation:
        return cls(
            subject_id=str(d["subject_id"]),
            object_id=str(d["object_id"]),
            relation=str(d.get("relation", "next_to")),
            confidence=float(d.get("confidence", 1.0)),
        )


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------

def _serialize_value(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    if hasattr(v, "to_dict"):
        return v.to_dict()
    if isinstance(v, (list, tuple)):
        return [_serialize_value(i) for i in v]
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    return v


def to_jsonable(obj: Any) -> Any:
    """将具身类型递归转为 JSON 可序列化对象"""
    return _serialize_value(obj)
