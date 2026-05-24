"""
MemoryAtom — 具身智能的统一记忆原子单元

一个 MemoryAtom = PowerMem memory + 具身扩展字段。
它是 ROSClaw-Memory 与 PowerMem 之间的适配层：
- 对外（ROSClaw）：强类型、具身感知
- 对内（PowerMem）：通过 to_metadata() / from_metadata() 无缝映射到 metadata payload
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .physical_model import PhysicalConstraint
from .types import (
    AffectiveTag,
    MemoryAction,
    Modality,
    PerceptualSnapshot,
    PhysicalInvariant,
    Pose,
    TemporalInterval,
    UncertaintyEstimate,
    UncertaintyType,
    Vec3,
    to_jsonable,
)


@dataclass(slots=True)
class MemoryAtom:
    """具身记忆原子 — ROSClaw-Memory 的核心数据单元

    Fields:
        content: 人类可读的文本描述（由 LLM 生成或人工标注）
        memory_id: PowerMem 分配的 Snowflake ID（写入后填充）
        spatial: 3D 世界坐标（可选）
        temporal: 时间区间（可选）
        perceptual: 感知快照（传感器测量）
        physical: 物理不变量（确定性骨架）
        uncertainty: 不确定性估计
        affective: 情感/显著性标记
        action: 记忆产生的动作类型
        prediction_error: 预测编码误差（Surprisal Gate 用）
        causal_parents: 因果父记忆 ID 列表
        embodied_meta: 任意扩展 JSON 元数据
        created_at: ISO-8601 时间戳
    """

    # --- 必需字段 ---
    content: str = ""

    # --- PowerMem 兼容字段 ---
    memory_id: Optional[int] = None
    user_id: str = ""
    agent_id: str = ""
    run_id: str = ""
    actor_id: str = ""

    # --- 具身扩展字段 ---
    spatial: Optional[Vec3] = None
    spatial_frame_id: str = "world"
    spatial_voxel_key: Optional[str] = None
    temporal: Optional[TemporalInterval] = None
    perceptual: Optional[PerceptualSnapshot] = None
    physical: Optional[PhysicalInvariant] = None
    uncertainty: Optional[UncertaintyEstimate] = None
    affective: Optional[AffectiveTag] = None
    action: MemoryAction = MemoryAction.OBSERVE
    prediction_error: float = 0.0
    causal_parents: List[int] = field(default_factory=list)
    embodied_meta: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # --- 内部缓存 ---
    _content_hash: Optional[str] = field(default=None, repr=False)
    _embedding: Optional[List[float]] = field(default=None, repr=False)

    # ========================================================================
    # 工厂方法
    # ========================================================================

    @classmethod
    def from_observation(
        cls,
        content: str,
        sensor_pose: Pose,
        modality: Modality = Modality.RGB,
        feature_vec: Optional[Tuple[float, ...]] = None,
        uncertainty: Optional[UncertaintyEstimate] = None,
        timestamp_sec: Optional[float] = None,
        **kwargs,
    ) -> MemoryAtom:
        """从传感器观测快速创建 MemoryAtom"""
        perceptual = PerceptualSnapshot(
            modality=modality,
            feature_vec=feature_vec or (),
            sensor_pose=sensor_pose,
            uncertainty=uncertainty or UncertaintyEstimate(),
        )
        temporal = None
        if timestamp_sec is not None:
            temporal = TemporalInterval(start_sec=timestamp_sec, end_sec=timestamp_sec)
        return cls(
            content=content,
            spatial=sensor_pose.position if sensor_pose else None,
            temporal=temporal,
            perceptual=perceptual,
            uncertainty=uncertainty,
            action=MemoryAction.OBSERVE,
            **kwargs,
        )

    @classmethod
    def from_physical_model(
        cls,
        content: str,
        physical: PhysicalInvariant,
        **kwargs,
    ) -> MemoryAtom:
        """从物理模型/约束创建 MemoryAtom"""
        return cls(
            content=content,
            physical=physical,
            action=MemoryAction.PREDICT,
            **kwargs,
        )

    @classmethod
    def from_action(
        cls,
        content: str,
        action_type: MemoryAction,
        spatial: Optional[Vec3] = None,
        temporal: Optional[TemporalInterval] = None,
        outcome_status: Optional[str] = None,
        **kwargs,
    ) -> MemoryAtom:
        """从动作执行记录创建 MemoryAtom

        Args:
            outcome_status: 动作结果，如 "success" | "collision" | "timeout" | "error"
        """
        meta = dict(kwargs.pop("embodied_meta", {}))
        if outcome_status is not None:
            meta["outcome_status"] = outcome_status
        return cls(
            content=content,
            action=action_type,
            spatial=spatial,
            temporal=temporal,
            embodied_meta=meta,
            **kwargs,
        )

    @classmethod
    def from_constraint(
        cls,
        content: str,
        constraint: "PhysicalConstraint",
        **kwargs,
    ) -> MemoryAtom:
        """从物理约束创建 MemoryAtom"""
        from .physical_model import PhysicalConstraint
        meta = dict(kwargs.pop("embodied_meta", {}))
        meta["constraint"] = constraint.to_dict()
        spatial = None
        if constraint.region_center is not None:
            spatial = Vec3(*constraint.region_center)
        return cls(
            content=content,
            spatial=spatial,
            embodied_meta=meta,
            action=MemoryAction.PREDICT,
            **kwargs,
        )

    @classmethod
    def from_world_object(
        cls,
        content: str,
        obj: Dict[str, Any],
        **kwargs,
    ) -> MemoryAtom:
        """从世界对象（ParseResult.world_objects）创建 MemoryAtom

        Args:
            obj: 解析器输出的世界对象 dict，含 name, type, pose, size 等字段
        """
        meta = dict(kwargs.pop("embodied_meta", {}))
        meta["world_object"] = obj
        meta["physical_type"] = "world_object"

        spatial = None
        pose = obj.get("pose")
        if pose:
            pos = pose.get("position") if isinstance(pose, dict) else None
            if pos:
                spatial = Vec3.from_dict(pos)
            elif isinstance(pose, (list, tuple)) and len(pose) >= 3:
                spatial = Vec3(float(pose[0]), float(pose[1]), float(pose[2]))

        return cls(
            content=content,
            spatial=spatial,
            embodied_meta=meta,
            action=MemoryAction.OBSERVE,
            **kwargs,
        )

    @classmethod
    def from_trajectory(
        cls,
        content: str,
        waypoints: List[Tuple[Vec3, float]],
        **kwargs,
    ) -> MemoryAtom:
        """从轨迹路点创建 MemoryAtom

        Args:
            waypoints: [(Vec3 position, float timestamp_sec), ...] 按时间排序

        Returns:
            MemoryAtom，spatial 为轨迹起点，temporal 为起止时间区间，
            embodied_meta 含完整路点列表 + 预计算签名（用于快速预过滤）
        """
        if not waypoints:
            return cls(
                content=content,
                embodied_meta={"trajectory": {"waypoints": [], "signature": [0.0] * 8}},
                action=MemoryAction.ACT,
                **kwargs,
            )

        meta = dict(kwargs.pop("embodied_meta", {}))
        wp_data = []
        for pos, ts in waypoints:
            wp_data.append({"position": pos.to_dict(), "timestamp_sec": ts})

        # 预计算轨迹特征签名，避免查询时从 JSON 重建路点再计算
        from .trajectory_similarity import trajectory_feature_signature
        sig = trajectory_feature_signature(waypoints)

        meta["trajectory"] = {
            "waypoints": wp_data,
            "duration": waypoints[-1][1] - waypoints[0][1],
            "waypoint_count": len(waypoints),
            "signature": list(sig),
        }
        # 标记 physical_type 以便 DB 层快速索引查询（替代慢速 LIKE '%trajectory%'）
        meta["physical_type"] = "trajectory"

        # 空间：轨迹中点（比起点更利于空间索引覆盖）
        mid_idx = len(waypoints) // 2
        spatial = waypoints[mid_idx][0]

        # 时间：起止区间
        temporal = TemporalInterval(
            start_sec=waypoints[0][1],
            end_sec=waypoints[-1][1],
        )

        return cls(
            content=content,
            spatial=spatial,
            temporal=temporal,
            embodied_meta=meta,
            action=MemoryAction.ACT,
            **kwargs,
        )

    # ========================================================================
    # PowerMem 互操作：metadata <-> MemoryAtom
    # ========================================================================

    def to_metadata(self) -> Dict[str, Any]:
        """将具身字段序列化为 PowerMem 的 metadata dict

        PowerMem 的 StorageAdapter 会将 payload["metadata"] 整体存储，
        因此所有具身字段嵌套在 metadata 内部，对 PowerMem 完全透明。
        """
        meta: Dict[str, Any] = {
            "rosclaw_version": "0.1.0",
            "action": self.action.value,
            "prediction_error": self.prediction_error,
            "spatial_frame_id": self.spatial_frame_id,
        }
        if self.spatial is not None:
            meta["spatial"] = self.spatial.to_dict()
        if self.spatial_voxel_key is not None:
            meta["spatial_voxel_key"] = self.spatial_voxel_key
        if self.temporal is not None:
            meta["temporal"] = self.temporal.to_dict()
        if self.perceptual is not None:
            meta["perceptual"] = self.perceptual.to_dict()
        if self.physical is not None:
            meta["physical"] = self.physical.to_dict()
        if self.uncertainty is not None:
            meta["uncertainty"] = self.uncertainty.to_dict()
        if self.affective is not None:
            meta["affective"] = self.affective.to_dict()
        if self.causal_parents:
            meta["causal_parents"] = self.causal_parents
        if self.embodied_meta:
            meta["embodied_meta"] = self.embodied_meta
        return meta

    @classmethod
    def from_metadata(cls, content: str, metadata: Dict[str, Any], **top_fields) -> MemoryAtom:
        """从 PowerMem 的 metadata dict 还原 MemoryAtom"""
        meta = metadata or {}

        def _get_spatial() -> Optional[Vec3]:
            s = meta.get("spatial")
            return Vec3.from_dict(s) if s else None

        def _get_temporal() -> Optional[TemporalInterval]:
            t = meta.get("temporal")
            return TemporalInterval.from_dict(t) if t else None

        def _get_perceptual() -> Optional[PerceptualSnapshot]:
            p = meta.get("perceptual")
            return PerceptualSnapshot.from_dict(p) if p else None

        def _get_physical() -> Optional[PhysicalInvariant]:
            p = meta.get("physical")
            return PhysicalInvariant.from_dict(p) if p else None

        def _get_uncertainty() -> Optional[UncertaintyEstimate]:
            u = meta.get("uncertainty")
            return UncertaintyEstimate.from_dict(u) if u else None

        def _get_affective() -> Optional[AffectiveTag]:
            a = meta.get("affective")
            return AffectiveTag.from_dict(a) if a else None

        return cls(
            content=content,
            memory_id=top_fields.get("memory_id") or top_fields.get("id"),
            user_id=top_fields.get("user_id", ""),
            agent_id=top_fields.get("agent_id", ""),
            run_id=top_fields.get("run_id", ""),
            actor_id=top_fields.get("actor_id", ""),
            spatial=_get_spatial(),
            spatial_frame_id=meta.get("spatial_frame_id", "world"),
            spatial_voxel_key=meta.get("spatial_voxel_key"),
            temporal=_get_temporal(),
            perceptual=_get_perceptual(),
            physical=_get_physical(),
            uncertainty=_get_uncertainty(),
            affective=_get_affective(),
            action=MemoryAction(meta.get("action", "observe")),
            prediction_error=float(meta.get("prediction_error", 0.0)),
            causal_parents=list(meta.get("causal_parents", [])),
            embodied_meta=dict(meta.get("embodied_meta", {})),
            created_at=top_fields.get("created_at"),
            updated_at=top_fields.get("updated_at"),
        )

    def to_powermem_payload(self) -> Dict[str, Any]:
        """生成可直接传给 StorageAdapter.add_memory() 的 payload dict"""
        payload = {
            "content": self.content,
            "metadata": self.to_metadata(),
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "actor_id": self.actor_id,
            "category": f"embodied_{self.action.value}",
            "hash": self.content_hash,
        }
        if self._embedding is not None:
            payload["embedding"] = self._embedding
        return payload

    # ========================================================================
    # 辅助属性
    # ========================================================================

    @property
    def content_hash(self) -> str:
        if self._content_hash is None:
            self._content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]
        return self._content_hash

    @property
    def embedding(self) -> Optional[List[float]]:
        return self._embedding

    @embedding.setter
    def embedding(self, value: Optional[List[float]]) -> None:
        self._embedding = value

    @property
    def is_significant(self) -> bool:
        """是否显著（用于快速过滤）"""
        if self.affective and self.affective.salience > 0.8:
            return True
        if abs(self.prediction_error) > 1.0:
            return True
        return False

    @property
    def is_high_uncertainty(self) -> bool:
        """是否高不确定性（需要优先探索）"""
        if self.uncertainty is None:
            return False
        return self.uncertainty.confidence < 0.3 or self.uncertainty.std > 1.0

    def compute_voxel_key(self, voxel_size: float = 0.1) -> Optional[str]:
        """计算空间体素键（用于 Voxel Hash）"""
        if self.spatial is None:
            return None
        vx = int(self.spatial.x / voxel_size)
        vy = int(self.spatial.y / voxel_size)
        vz = int(self.spatial.z / voxel_size)
        key = f"{vx}:{vy}:{vz}:{self.spatial_frame_id}"
        self.spatial_voxel_key = key
        return key

    def __repr__(self) -> str:
        parts = [f"MemoryAtom(id={self.memory_id}, action={self.action.value}"]
        if self.spatial:
            parts.append(f"pos=({self.spatial.x:.2f}, {self.spatial.y:.2f}, {self.spatial.z:.2f})")
        if self.temporal:
            parts.append(f"t=[{self.temporal.start_sec:.2f}, {self.temporal.end_sec:.2f}]")
        if self.perceptual:
            parts.append(f"modality={self.perceptual.modality.value}")
        if self.affective:
            parts.append(f"salience={self.affective.salience:.2f}")
        parts.append(f"pe={self.prediction_error:.3f})")
        return " ".join(parts)
