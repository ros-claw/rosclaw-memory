"""
轻量级碰撞检测 — 纯 Python，零外部依赖

支持几何体：
- Sphere：球体（关节末端、球形碰撞体）
- Capsule：胶囊体（圆柱+半球，最常用的连杆碰撞体）
- AABB：轴对齐包围盒（快速粗检测）

碰撞查询：
- point_in_geom：点是否在几何体内
- geom_distance：几何体到点的最近距离
- geom_geom_intersect：两个几何体是否相交
- ray_intersect：射线与几何体相交

与空间索引结合：碰撞体中心坐标通过 Voxel Hash 索引。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .types import Vec3


# ---------------------------------------------------------------------------
# 几何体定义
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Sphere:
    """球体：中心 + 半径"""
    center: Vec3
    radius: float

    def contains(self, point: Vec3) -> bool:
        return self.center.distance_to(point) <= self.radius

    def distance_to(self, point: Vec3) -> float:
        return max(0.0, self.center.distance_to(point) - self.radius)

    def aabb(self) -> AABB:
        r = self.radius
        return AABB(
            min=Vec3(self.center.x - r, self.center.y - r, self.center.z - r),
            max=Vec3(self.center.x + r, self.center.y + r, self.center.z + r),
        )


@dataclass(frozen=True, slots=True)
class Capsule:
    """胶囊体：线段 + 半径（圆柱体两端加半球）

    最常用的机器人连杆碰撞表示。
    """
    a: Vec3      # 线段起点
    b: Vec3      # 线段终点
    radius: float

    def contains(self, point: Vec3) -> bool:
        return self.distance_to(point) <= 0.0

    def distance_to(self, point: Vec3) -> float:
        """点到胶囊体的有符号距离（负 = 内部）"""
        # 投影点在线段上的最近点
        ab = self.b - self.a
        ab_len_sq = ab.x**2 + ab.y**2 + ab.z**2
        if ab_len_sq < 1e-12:
            # 退化为球体
            return self.a.distance_to(point) - self.radius

        t = max(0.0, min(1.0,
            ((point.x - self.a.x) * ab.x +
             (point.y - self.a.y) * ab.y +
             (point.z - self.a.z) * ab.z) / ab_len_sq
        ))
        closest = Vec3(
            self.a.x + t * ab.x,
            self.a.y + t * ab.y,
            self.a.z + t * ab.z,
        )
        return closest.distance_to(point) - self.radius

    def aabb(self) -> AABB:
        r = self.radius
        return AABB(
            min=Vec3(min(self.a.x, self.b.x) - r, min(self.a.y, self.b.y) - r, min(self.a.z, self.b.z) - r),
            max=Vec3(max(self.a.x, self.b.x) + r, max(self.a.y, self.b.y) + r, max(self.a.z, self.b.z) + r),
        )


@dataclass(frozen=True, slots=True)
class AABB:
    """轴对齐包围盒 — 用于快速粗检测"""
    min: Vec3
    max: Vec3

    def contains(self, point: Vec3) -> bool:
        return (
            self.min.x <= point.x <= self.max.x
            and self.min.y <= point.y <= self.max.y
            and self.min.z <= point.z <= self.max.z
        )

    def intersects(self, other: AABB) -> bool:
        return (
            self.min.x <= other.max.x and self.max.x >= other.min.x
            and self.min.y <= other.max.y and self.max.y >= other.min.y
            and self.min.z <= other.max.z and self.max.z >= other.min.z
        )

    def distance_to(self, point: Vec3) -> float:
        """点到 AABB 的最近距离（内部 = 0）"""
        dx = max(self.min.x - point.x, 0.0, point.x - self.max.x)
        dy = max(self.min.y - point.y, 0.0, point.y - self.max.y)
        dz = max(self.min.z - point.z, 0.0, point.z - self.max.z)
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def center(self) -> Vec3:
        return Vec3(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )

    def diagonal(self) -> float:
        return self.min.distance_to(self.max)


# ---------------------------------------------------------------------------
# 统一碰撞体
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CollisionBody:
    """统一碰撞体 — 包装任意几何体 + 元数据"""
    entity_id: str            # 所属物体/连杆 ID
    geom_type: str            # "sphere" | "capsule" | "aabb"
    geometry: Sphere | Capsule | AABB
    link_name: str = ""       # 所属连杆名
    frame_id: str = "world"   # 坐标系

    def contains(self, point: Vec3) -> bool:
        return self.geometry.contains(point)

    def distance_to(self, point: Vec3) -> float:
        return self.geometry.distance_to(point)

    def aabb(self) -> AABB:
        return self.geometry.aabb()


# ---------------------------------------------------------------------------
# 碰撞检测函数
# ---------------------------------------------------------------------------

def bodies_intersect(a: CollisionBody, b: CollisionBody) -> bool:
    """两个碰撞体是否相交 — 使用 AABB 粗检测 + 精确检测"""
    # 1. AABB 粗检测
    if not a.aabb().intersects(b.aabb()):
        return False

    # 2. 精确检测（简化：采样距离）
    # 对于球体/胶囊体组合，使用解析方法
    if isinstance(a.geometry, Sphere) and isinstance(b.geometry, Sphere):
        return _sphere_sphere_intersect(a.geometry, b.geometry)
    if isinstance(a.geometry, Sphere) and isinstance(b.geometry, Capsule):
        return _sphere_capsule_intersect(a.geometry, b.geometry)
    if isinstance(a.geometry, Capsule) and isinstance(b.geometry, Sphere):
        return _sphere_capsule_intersect(b.geometry, a.geometry)
    if isinstance(a.geometry, Capsule) and isinstance(b.geometry, Capsule):
        return _capsule_capsule_intersect(a.geometry, b.geometry)

    # 通用回退：检查中心点
    return a.contains(b.geometry.center()) or b.contains(a.geometry.center())


def _sphere_sphere_intersect(a: Sphere, b: Sphere) -> bool:
    return a.center.distance_to(b.center) <= (a.radius + b.radius)


def _sphere_capsule_intersect(sphere: Sphere, capsule: Capsule) -> bool:
    return capsule.distance_to(sphere.center) <= sphere.radius


def _capsule_capsule_intersect(a: Capsule, b: Capsule) -> bool:
    # 线段间最近距离
    dist = _segment_segment_distance(a.a, a.b, b.a, b.b)
    return dist <= (a.radius + b.radius)


def _segment_segment_distance(
    p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3
) -> float:
    """两条线段间的最短距离"""
    # 使用向量叉积计算线段间距离
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3

    a = u.x*u.x + u.y*u.y + u.z*u.z
    b = u.x*v.x + u.y*v.y + u.z*v.z
    c = v.x*v.x + v.y*v.y + v.z*v.z
    d = u.x*w.x + u.y*w.y + u.z*w.z
    e = v.x*w.x + v.y*w.y + v.z*w.z
    D = a*c - b*b

    sc = sN = sD = D
    tc = tN = tD = D

    if D < 1e-12:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < 1e-12 else sN / sD
    tc = 0.0 if abs(tN) < 1e-12 else tN / tD

    dP = w + Vec3(
        sc * u.x - tc * v.x,
        sc * u.y - tc * v.y,
        sc * u.z - tc * v.z,
    )
    return math.sqrt(dP.x*dP.x + dP.y*dP.y + dP.z*dP.z)


# ---------------------------------------------------------------------------
# 从 ParseResult 构建碰撞体
# ---------------------------------------------------------------------------

def build_collision_bodies(
    parse_result: "ParseResult",
    default_radius: float = 0.05,
) -> List[CollisionBody]:
    """从 ParseResult 的碰撞几何体构建 CollisionBody 列表

    策略（优先级从高到低）：
    1. 如果 link 已解析出 collision_geoms，直接还原为对应几何体
    2. 否则回退到基于质量的近似球体
    """
    from .parsers.base import ParseResult
    bodies: List[CollisionBody] = []
    source_id = parse_result.source_hash or "model"

    for link in parse_result.links:
        link_name = link.get("name", "")
        collision_geoms = link.get("collision_geoms", [])

        if collision_geoms:
            for g in collision_geoms:
                gtype = g.get("type", "sphere")
                entity_id = f"{source_id}:{link_name}"
                if gtype == "sphere":
                    bodies.append(CollisionBody(
                        entity_id=entity_id,
                        geom_type="sphere",
                        geometry=Sphere(
                            center=Vec3(*g.get("center", [0.0, 0.0, 0.0])),
                            radius=float(g.get("radius", default_radius)),
                        ),
                        link_name=link_name,
                    ))
                elif gtype == "capsule":
                    bodies.append(CollisionBody(
                        entity_id=entity_id,
                        geom_type="capsule",
                        geometry=Capsule(
                            a=Vec3(*g.get("a", [0.0, 0.0, 0.0])),
                            b=Vec3(*g.get("b", [0.0, 0.0, 0.0])),
                            radius=float(g.get("radius", default_radius)),
                        ),
                        link_name=link_name,
                    ))
                elif gtype == "aabb":
                    bodies.append(CollisionBody(
                        entity_id=entity_id,
                        geom_type="aabb",
                        geometry=AABB(
                            min=Vec3(*g.get("min", [0.0, 0.0, 0.0])),
                            max=Vec3(*g.get("max", [0.0, 0.0, 0.0])),
                        ),
                        link_name=link_name,
                    ))
        else:
            # 回退：基于质量的近似球体
            com = link.get("com", {"x": 0.0, "y": 0.0, "z": 0.0})
            center = Vec3(com["x"], com["y"], com["z"])
            mass = link.get("mass", 1.0)
            radius = (mass ** (1.0 / 3.0)) * default_radius
            bodies.append(CollisionBody(
                entity_id=f"{source_id}:{link_name}",
                geom_type="sphere",
                geometry=Sphere(center=center, radius=radius),
                link_name=link_name,
            ))

    return bodies


# ---------------------------------------------------------------------------
# 碰撞查询管理器
# ---------------------------------------------------------------------------

class CollisionChecker:
    """碰撞查询管理器 — 批量检测 + AABB 加速"""

    def __init__(self):
        self.bodies: List[CollisionBody] = []

    def add_body(self, body: CollisionBody) -> None:
        self.bodies.append(body)

    def add_bodies(self, bodies: List[CollisionBody]) -> None:
        self.bodies.extend(bodies)

    def check_point(self, point: Vec3) -> List[CollisionBody]:
        """查询包含给定点的所有碰撞体"""
        return [b for b in self.bodies if b.contains(point)]

    def check_intersections(self) -> List[Tuple[CollisionBody, CollisionBody]]:
        """检测所有碰撞体对之间的相交"""
        collisions: List[Tuple[CollisionBody, CollisionBody]] = []
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                if bodies_intersect(self.bodies[i], self.bodies[j]):
                    collisions.append((self.bodies[i], self.bodies[j]))
        return collisions

    def nearest_body(self, point: Vec3) -> Optional[Tuple[CollisionBody, float]]:
        """查找离点最近的碰撞体"""
        if not self.bodies:
            return None
        nearest = min(self.bodies, key=lambda b: b.distance_to(point))
        return nearest, nearest.distance_to(point)

    def clear(self) -> None:
        self.bodies.clear()
