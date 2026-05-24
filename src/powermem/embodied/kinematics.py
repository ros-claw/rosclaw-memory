"""
正运动学 — 纯 Python，零外部依赖

基于标准 DH 参数（Denavit-Hartenberg）计算连杆位姿，
支持将碰撞体从连杆局部坐标系变换到世界坐标系。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .collision import AABB, Capsule, CollisionBody, Sphere
from .physical_model import DHParameter
from .types import Pose, Quaternion, Vec3


@dataclass(frozen=True)
class Transform:
    """4x4 齐次变换矩阵（行优先扁平存储）"""
    m00: float = 1.0
    m01: float = 0.0
    m02: float = 0.0
    m03: float = 0.0
    m10: float = 0.0
    m11: float = 1.0
    m12: float = 0.0
    m13: float = 0.0
    m20: float = 0.0
    m21: float = 0.0
    m22: float = 1.0
    m23: float = 0.0

    def __matmul__(self, other: "Transform") -> "Transform":
        """矩阵乘法：self * other"""
        return Transform(
            m00=self.m00 * other.m00 + self.m01 * other.m10 + self.m02 * other.m20,
            m01=self.m00 * other.m01 + self.m01 * other.m11 + self.m02 * other.m21,
            m02=self.m00 * other.m02 + self.m01 * other.m12 + self.m02 * other.m22,
            m03=self.m00 * other.m03 + self.m01 * other.m13 + self.m02 * other.m23 + self.m03,
            m10=self.m10 * other.m00 + self.m11 * other.m10 + self.m12 * other.m20,
            m11=self.m10 * other.m01 + self.m11 * other.m11 + self.m12 * other.m21,
            m12=self.m10 * other.m02 + self.m11 * other.m12 + self.m12 * other.m22,
            m13=self.m10 * other.m03 + self.m11 * other.m13 + self.m12 * other.m23 + self.m13,
            m20=self.m20 * other.m00 + self.m21 * other.m10 + self.m22 * other.m20,
            m21=self.m20 * other.m01 + self.m21 * other.m11 + self.m22 * other.m21,
            m22=self.m20 * other.m02 + self.m21 * other.m12 + self.m22 * other.m22,
            m23=self.m20 * other.m03 + self.m21 * other.m13 + self.m22 * other.m23 + self.m23,
        )

    def apply(self, v: Vec3) -> Vec3:
        """对三维点应用旋转+平移（假设 w=1）"""
        return Vec3(
            self.m00 * v.x + self.m01 * v.y + self.m02 * v.z + self.m03,
            self.m10 * v.x + self.m11 * v.y + self.m12 * v.z + self.m13,
            self.m20 * v.x + self.m21 * v.y + self.m22 * v.z + self.m23,
        )

    def to_pose(self) -> Pose:
        """提取位姿（位置 + 旋转四元数）"""
        # 位置
        pos = Vec3(self.m03, self.m13, self.m23)
        # 旋转矩阵 → 四元数（Shepperd 方法）
        q = _rotation_matrix_to_quaternion(self)
        return Pose(position=pos, orientation=q)

    @classmethod
    def identity(cls) -> "Transform":
        return cls()

    @classmethod
    def from_translation(cls, x: float, y: float, z: float) -> "Transform":
        return cls(m03=x, m13=y, m23=z)


def _rotation_matrix_to_quaternion(T: Transform) -> Quaternion:
    """将旋转矩阵转为四元数"""
    # 旋转矩阵 3x3:
    # [m00 m01 m02]
    # [m10 m11 m12]
    # [m20 m21 m22]
    trace = T.m00 + T.m11 + T.m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (T.m21 - T.m12) * s
        y = (T.m02 - T.m20) * s
        z = (T.m10 - T.m01) * s
    elif T.m00 > T.m11 and T.m00 > T.m22:
        s = 2.0 * math.sqrt(1.0 + T.m00 - T.m11 - T.m22)
        w = (T.m21 - T.m12) / s
        x = 0.25 * s
        y = (T.m01 + T.m10) / s
        z = (T.m02 + T.m20) / s
    elif T.m11 > T.m22:
        s = 2.0 * math.sqrt(1.0 + T.m11 - T.m00 - T.m22)
        w = (T.m02 - T.m20) / s
        x = (T.m01 + T.m10) / s
        y = 0.25 * s
        z = (T.m12 + T.m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + T.m22 - T.m00 - T.m11)
        w = (T.m10 - T.m01) / s
        x = (T.m02 + T.m20) / s
        y = (T.m12 + T.m21) / s
        z = 0.25 * s
    return Quaternion(w=w, x=x, y=y, z=z)


# ---------------------------------------------------------------------------
# DH 参数 → 变换矩阵
# ---------------------------------------------------------------------------

def dh_to_transform(dh: DHParameter, joint_angle: float) -> Transform:
    """标准 DH 参数转齐次变换矩阵

    参数：
        dh: DH 参数 (d, theta, a, alpha)
        joint_angle: 当前关节角（加到 theta 上）

    标准 DH 约定（Craig）：
        T = Rot(z, θ) * Trans(z, d) * Trans(x, a) * Rot(x, α)
    """
    theta = dh.theta + joint_angle
    d = dh.d
    a = dh.a
    alpha = dh.alpha

    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    return Transform(
        m00=ct,
        m01=-st * ca,
        m02=st * sa,
        m03=a * ct,
        m10=st,
        m11=ct * ca,
        m12=-ct * sa,
        m13=a * st,
        m20=0.0,
        m21=sa,
        m22=ca,
        m23=d,
    )


# ---------------------------------------------------------------------------
# 正运动学
# ---------------------------------------------------------------------------

def forward_kinematics(
    dh_params: List[DHParameter],
    joint_angles: List[float],
) -> List[Transform]:
    """正运动学：计算每个关节/连杆到世界坐标系的变换

    Args:
        dh_params: N 个 DH 参数
        joint_angles: N 个关节角（弧度）

    Returns:
        transforms: N 个 Transform，第 i 个表示连杆 i 在世界坐标系下的位姿
    """
    n = len(dh_params)
    if n == 0:
        return []

    # 补齐 joint_angles
    angles = list(joint_angles) + [0.0] * (n - len(joint_angles))
    angles = angles[:n]

    transforms: List[Transform] = []
    T_world = Transform.identity()

    for i in range(n):
        T_i = dh_to_transform(dh_params[i], angles[i])
        T_world = T_world @ T_i
        transforms.append(T_world)

    return transforms


def forward_kinematics_poses(
    dh_params: List[DHParameter],
    joint_angles: List[float],
) -> List[Pose]:
    """正运动学，返回 Pose 列表"""
    return [T.to_pose() for T in forward_kinematics(dh_params, joint_angles)]


# ---------------------------------------------------------------------------
# 碰撞体坐标变换
# ---------------------------------------------------------------------------

def transform_collision_body(body: CollisionBody, T: Transform) -> CollisionBody:
    """将碰撞体从局部坐标系变换到世界坐标系"""
    geom = body.geometry
    if isinstance(geom, Sphere):
        new_geom = Sphere(
            center=T.apply(geom.center),
            radius=geom.radius,
        )
    elif isinstance(geom, Capsule):
        new_geom = Capsule(
            a=T.apply(geom.a),
            b=T.apply(geom.b),
            radius=geom.radius,
        )
    elif isinstance(geom, AABB):
        # AABB 旋转后不再是轴对齐，变换 8 个角点后重新包围
        corners = [
            Vec3(geom.min.x, geom.min.y, geom.min.z),
            Vec3(geom.min.x, geom.min.y, geom.max.z),
            Vec3(geom.min.x, geom.max.y, geom.min.z),
            Vec3(geom.min.x, geom.max.y, geom.max.z),
            Vec3(geom.max.x, geom.min.y, geom.min.z),
            Vec3(geom.max.x, geom.min.y, geom.max.z),
            Vec3(geom.max.x, geom.max.y, geom.min.z),
            Vec3(geom.max.x, geom.max.y, geom.max.z),
        ]
        transformed = [T.apply(c) for c in corners]
        min_x = min(v.x for v in transformed)
        min_y = min(v.y for v in transformed)
        min_z = min(v.z for v in transformed)
        max_x = max(v.x for v in transformed)
        max_y = max(v.y for v in transformed)
        max_z = max(v.z for v in transformed)
        new_geom = AABB(
            min=Vec3(min_x, min_y, min_z),
            max=Vec3(max_x, max_y, max_z),
        )
    else:
        # 未知几何体，直接复制
        new_geom = geom  # type: ignore[assignment]

    return CollisionBody(
        entity_id=body.entity_id,
        geom_type=body.geom_type,
        geometry=new_geom,
        link_name=body.link_name,
        frame_id="world",
    )


def transform_collision_bodies(
    bodies: List[CollisionBody],
    transforms: List[Transform],
    link_index_map: dict[str, int],
) -> List[CollisionBody]:
    """批量变换碰撞体到世界坐标系

    Args:
        bodies: 局部坐标系下的碰撞体列表
        transforms: 每个连杆到世界的变换（由 forward_kinematics 产出）
        link_index_map: link_name -> transforms 索引的映射

    Returns:
        世界坐标系下的碰撞体列表
    """
    result: List[CollisionBody] = []
    for body in bodies:
        idx = link_index_map.get(body.link_name, -1)
        if idx < 0 or idx >= len(transforms):
            # 找不到对应连杆，保持原样（通常不应该发生）
            result.append(body)
            continue
        result.append(transform_collision_body(body, transforms[idx]))
    return result
