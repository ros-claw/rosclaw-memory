"""
物理模型 — 纯数学描述的机器人/环境动力学

设计原则：
1. 零 ROS 依赖：所有参数从 URDF/SDF 解析后存入纯数学结构
2. 运行时独立：存储的是参数化公式，不是 ROS 消息
3. 可验证：通过物理不变量验证一致性（如质量 > 0）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 关节定义
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class JointLimit:
    """关节限制"""
    min_rad: float = -3.14159
    max_rad: float = 3.14159
    max_vel: float = 1.0       # rad/s
    max_torque: float = 10.0   # Nm

    def to_dict(self) -> Dict[str, float]:
        return {"min_rad": self.min_rad, "max_rad": self.max_rad, "max_vel": self.max_vel, "max_torque": self.max_torque}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> JointLimit:
        return cls(
            min_rad=float(d.get("min_rad", -3.14159)),
            max_rad=float(d.get("max_rad", 3.14159)),
            max_vel=float(d.get("max_vel", 1.0)),
            max_torque=float(d.get("max_torque", 10.0)),
        )


@dataclass(frozen=True, slots=True)
class DHParameter:
    """Denavit-Hartenberg 参数"""
    d: float = 0.0      # 连杆偏距
    theta: float = 0.0  # 关节角（变量）
    a: float = 0.0      # 连杆长度
    alpha: float = 0.0  # 连杆扭角

    def to_dict(self) -> Dict[str, float]:
        return {"d": self.d, "theta": self.theta, "a": self.a, "alpha": self.alpha}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DHParameter:
        return cls(
            d=float(d.get("d", 0.0)),
            theta=float(d.get("theta", 0.0)),
            a=float(d.get("a", 0.0)),
            alpha=float(d.get("alpha", 0.0)),
        )


# ---------------------------------------------------------------------------
# 机器人动力学模型
# ---------------------------------------------------------------------------

@dataclass
class RobotDynamics:
    """机器人动力学参数化表示

    标准形式：M(q) * qddot + C(q, qdot) * qdot + G(q) = tau
    存储的是参数化公式系数，不是运行时矩阵值。
    """
    # 关节列表
    joint_names: List[str] = field(default_factory=list)
    dh_params: List[DHParameter] = field(default_factory=list)
    joint_limits: List[JointLimit] = field(default_factory=list)

    # 质量矩阵 M(q) 的符号/参数化表示（JSON 字符串，供符号计算库使用）
    mass_matrix_expr: str = "{}"
    coriolis_matrix_expr: str = "{}"
    gravity_vector_expr: str = "{}"

    # 连杆动力学参数
    link_masses: List[float] = field(default_factory=list)
    link_inertias: List[List[float]] = field(default_factory=list)  # 每个 3x3 扁平化

    # 碰撞体（简化表示：球/胶囊/盒）
    collision_geoms: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> List[str]:
        """验证模型一致性，返回错误列表（空 = 有效）"""
        errors = []
        n = len(self.joint_names)
        if n == 0:
            errors.append("至少需要一个关节")
        if len(self.dh_params) != n:
            errors.append(f"DH 参数数量 ({len(self.dh_params)}) 与关节数 ({n}) 不匹配")
        if len(self.joint_limits) != n:
            errors.append(f"关节限制数量 ({len(self.joint_limits)}) 与关节数 ({n}) 不匹配")
        for i, m in enumerate(self.link_masses):
            if m <= 0:
                errors.append(f"连杆 {i} 质量必须 > 0")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_names": self.joint_names,
            "dh_params": [dh.to_dict() for dh in self.dh_params],
            "joint_limits": [jl.to_dict() for jl in self.joint_limits],
            "mass_matrix_expr": self.mass_matrix_expr,
            "coriolis_matrix_expr": self.coriolis_matrix_expr,
            "gravity_vector_expr": self.gravity_vector_expr,
            "link_masses": self.link_masses,
            "link_inertias": self.link_inertias,
            "collision_geoms": self.collision_geoms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RobotDynamics:
        return cls(
            joint_names=list(d.get("joint_names", [])),
            dh_params=[DHParameter.from_dict(x) for x in d.get("dh_params", [])],
            joint_limits=[JointLimit.from_dict(x) for x in d.get("joint_limits", [])],
            mass_matrix_expr=str(d.get("mass_matrix_expr", "{}")),
            coriolis_matrix_expr=str(d.get("coriolis_matrix_expr", "{}")),
            gravity_vector_expr=str(d.get("gravity_vector_expr", "{}")),
            link_masses=list(d.get("link_masses", [])),
            link_inertias=[list(x) for x in d.get("link_inertias", [])],
            collision_geoms=list(d.get("collision_geoms", [])),
        )

    @classmethod
    def from_urdf_dict(cls, urdf_dict: Dict[str, Any]) -> RobotDynamics:
        """从解析后的 URDF dict 创建动力学模型

        urdf_dict 格式（不依赖 ROS，由外部解析器提供）：
        {
            "joints": [{"name": ..., "type": ..., "axis": ..., "limit": {...}}],
            "links": [{"name": ..., "mass": ..., "inertia": {...}}],
        }
        """
        joints = urdf_dict.get("joints", [])
        links = urdf_dict.get("links", [])

        joint_names = [j["name"] for j in joints]
        joint_limits = []
        for j in joints:
            lim = j.get("limit", {})
            joint_limits.append(JointLimit(
                min_rad=float(lim.get("lower", -3.14159)),
                max_rad=float(lim.get("upper", 3.14159)),
                max_vel=float(lim.get("velocity", 1.0)),
                max_torque=float(lim.get("effort", 10.0)),
            ))

        # 简化 DH 参数提取（实际应由外部 kinematics 库提供）
        dh_params = [DHParameter() for _ in joint_names]

        link_masses = []
        link_inertias = []
        for link in links:
            mass = float(link.get("mass", 1.0))
            link_masses.append(mass)
            inertia = link.get("inertia", {})
            # 3x3 惯性矩阵扁平化
            inertia_flat = [
                float(inertia.get("ixx", 0.001)), float(inertia.get("ixy", 0.0)), float(inertia.get("ixz", 0.0)),
                float(inertia.get("ixy", 0.0)), float(inertia.get("iyy", 0.001)), float(inertia.get("iyz", 0.0)),
                float(inertia.get("ixz", 0.0)), float(inertia.get("iyz", 0.0)), float(inertia.get("izz", 0.001)),
            ]
            link_inertias.append(inertia_flat)

        return cls(
            joint_names=joint_names,
            dh_params=dh_params,
            joint_limits=joint_limits,
            link_masses=link_masses,
            link_inertias=link_inertias,
        )


# ---------------------------------------------------------------------------
# 物理约束
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PhysicalConstraint:
    """物理约束 — 存储在记忆中的确定性知识"""
    constraint_type: str  # "reachability" | "stability" | "collision_free" | "graspable"
    description: str = ""
    # 约束参数（类型特定）
    params: Dict[str, Any] = field(default_factory=dict)
    # 适用空间区域
    region_center: Optional[Tuple[float, float, float]] = None
    region_radius: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "description": self.description,
            "params": self.params,
            "region_center": self.region_center,
            "region_radius": self.region_radius,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PhysicalConstraint:
        return cls(
            constraint_type=str(d.get("constraint_type", "")),
            description=str(d.get("description", "")),
            params=dict(d.get("params", {})),
            region_center=tuple(d["region_center"]) if d.get("region_center") else None,
            region_radius=float(d.get("region_radius", 0.0)),
        )
