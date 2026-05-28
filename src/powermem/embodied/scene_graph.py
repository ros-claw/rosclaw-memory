"""
场景图 —— 世界对象的层级结构与空间关系推理

从 WorldObjectStore 加载对象与关系，提供：
- 层级遍历（父子关系）
- 关系查询（on / in / next_to）
- 自动关系计算（基于 AABB 启发式）
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .types import Pose, SpatialRelation, Vec3, WorldObject
from .world_object_store import WorldObjectStore

logger = logging.getLogger(__name__)


class AABB:
    """轴对齐包围盒"""

    def __init__(self, min_point: Vec3, max_point: Vec3):
        self.min = min_point
        self.max = max_point

    @classmethod
    def from_world_object(cls, obj: WorldObject) -> Optional[AABB]:
        """从 WorldObject 的 pose + size 构建 AABB"""
        if obj.size is None:
            return None
        cx, cy, cz = obj.pose.position.x, obj.pose.position.y, obj.pose.position.z
        if obj.obj_type == "sphere":
            r = obj.size[0] if len(obj.size) >= 1 else 0.05
            return cls(Vec3(cx - r, cy - r, cz - r), Vec3(cx + r, cy + r, cz + r))
        # box / cylinder / capsule / mesh: size = [w, h, d] or [radius, length]
        sx = obj.size[0] if len(obj.size) >= 1 else 0.1
        sy = obj.size[1] if len(obj.size) >= 2 else sx
        sz = obj.size[2] if len(obj.size) >= 3 else sx
        return cls(
            Vec3(cx - sx / 2, cy - sy / 2, cz - sz / 2),
            Vec3(cx + sx / 2, cy + sy / 2, cz + sz / 2),
        )

    def intersects(self, other: AABB) -> bool:
        return (
            self.min.x <= other.max.x and self.max.x >= other.min.x
            and self.min.y <= other.max.y and self.max.y >= other.min.y
            and self.min.z <= other.max.z and self.max.z >= other.min.z
        )

    def contains(self, other: AABB) -> bool:
        return (
            self.min.x <= other.min.x and self.max.x >= other.max.x
            and self.min.y <= other.min.y and self.max.y >= other.max.y
            and self.min.z <= other.min.z and self.max.z >= other.max.z
        )

    def bottom(self) -> float:
        return self.min.z

    def top(self) -> float:
        return self.max.z

    def horizontal_center(self) -> Tuple[float, float]:
        return ((self.min.x + self.max.x) / 2, (self.min.y + self.max.y) / 2)

    def horizontal_overlap_ratio(self, other: AABB) -> float:
        """计算水平投影重叠面积比例"""
        ix_min = max(self.min.x, other.min.x)
        ix_max = min(self.max.x, other.max.x)
        iy_min = max(self.min.y, other.min.y)
        iy_max = min(self.max.y, other.max.y)
        if ix_max <= ix_min or iy_max <= iy_min:
            return 0.0
        inter = (ix_max - ix_min) * (iy_max - iy_min)
        a1 = (self.max.x - self.min.x) * (self.max.y - self.min.y)
        a2 = (other.max.x - other.min.x) * (other.max.y - other.min.y)
        min_area = min(a1, a2)
        return inter / min_area if min_area > 0 else 0.0

    def horizontal_distance(self, other: AABB) -> float:
        """水平投影中心距离"""
        c1 = self.horizontal_center()
        c2 = other.horizontal_center()
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


class SceneGraph:
    """场景图 —— 内存中的对象层级与关系视图"""

    def __init__(self, scene_id: str, store: WorldObjectStore):
        self.scene_id = scene_id
        self.store = store
        self._objects: Dict[str, WorldObject] = {}
        self._relations: List[SpatialRelation] = []
        self._by_parent: Dict[str, List[WorldObject]] = {}
        self._by_relation: Dict[str, Dict[str, List[SpatialRelation]]] = {}
        self._built = False

    def build(self) -> None:
        """从 store 加载所有对象与关系（单次批量查询）"""
        objs = self.store.list_by_scene(self.scene_id, limit=10000)
        self._objects = {o.obj_id: o for o in objs}

        # 批量加载场景内所有关系（一次 IN 查询，避免 N 次单点查询）
        scene_rels_map = self.store.get_scene_graph(self.scene_id)
        seen = set()
        unique_rels = []
        for rel_list in scene_rels_map.values():
            for rel in rel_list:
                key = (rel.subject_id, rel.object_id, rel.relation)
                if key not in seen:
                    seen.add(key)
                    unique_rels.append(rel)
        self._relations = unique_rels

        # 按父对象分组（层级）
        self._by_parent = {}
        for o in objs:
            pid = o.parent_obj_id
            if pid:
                self._by_parent.setdefault(pid, []).append(o)

        # 按关系类型索引
        self._by_relation = {}
        for rel in self._relations:
            self._by_relation.setdefault(rel.relation, {}).setdefault(rel.subject_id, []).append(rel)

        self._built = True
        logger.debug("SceneGraph built for %s with %d objects, %d relations", self.scene_id, len(objs), len(self._relations))

    def get_objects(self) -> List[WorldObject]:
        if not self._built:
            self.build()
        return list(self._objects.values())

    def get_object(self, obj_id: str) -> Optional[WorldObject]:
        if not self._built:
            self.build()
        return self._objects.get(obj_id)

    def get_children(self, obj_id: str) -> List[WorldObject]:
        """获取 scene graph 中的子对象（parent_obj_id 匹配）"""
        if not self._built:
            self.build()
        return self._by_parent.get(obj_id, [])

    def get_parent(self, obj_id: str) -> Optional[WorldObject]:
        if not self._built:
            self.build()
        obj = self._objects.get(obj_id)
        if obj is None or obj.parent_obj_id is None:
            return None
        return self._objects.get(obj.parent_obj_id)

    def get_objects_with_relation(self, obj_id: str, relation: str) -> List[WorldObject]:
        """获取与指定对象有某种关系的所有对象（作为 subject）"""
        if not self._built:
            self.build()
        result = []
        rel_map = self._by_relation.get(relation, {})
        for rel in rel_map.get(obj_id, []):
            target = self._objects.get(rel.object_id)
            if target:
                result.append(target)
        return result

    def get_objects_on(self, obj_id: str) -> List[WorldObject]:
        return self.get_objects_with_relation(obj_id, "on")

    def get_objects_in(self, obj_id: str) -> List[WorldObject]:
        return self.get_objects_with_relation(obj_id, "in")

    def get_objects_next_to(self, obj_id: str) -> List[WorldObject]:
        return self.get_objects_with_relation(obj_id, "next_to")

    @staticmethod
    def _compute_pair_relations(
        id_a: str,
        id_b: str,
        aabbs: Dict[str, AABB],
        tol: float,
    ) -> Optional[SpatialRelation]:
        """Compute the dominant spatial relation for a single pair (if any)."""
        a, b = aabbs[id_a], aabbs[id_b]

        a_bottom, a_top = a.bottom(), a.top()
        b_bottom, b_top = b.bottom(), b.top()
        vertical_gap_bottom = abs(a_bottom - b_top)
        vertical_gap_top = abs(a_top - b_bottom)
        overlap_ratio = a.horizontal_overlap_ratio(b)

        # on: a on b
        if vertical_gap_bottom <= tol and overlap_ratio > 0.5 and a_top > b_top:
            return SpatialRelation(id_a, id_b, "on", confidence=overlap_ratio)
        # on: b on a
        if vertical_gap_top <= tol and overlap_ratio > 0.5 and b_top > a_top:
            return SpatialRelation(id_b, id_a, "on", confidence=overlap_ratio)

        # in: a in b
        if b.contains(a) or (b.min.x <= a.min.x + tol and b.max.x >= a.max.x - tol
                              and b.min.y <= a.min.y + tol and b.max.y >= a.max.y - tol
                              and b.min.z <= a.min.z + tol and b.max.z >= a.max.z - tol):
            return SpatialRelation(id_a, id_b, "in", confidence=1.0)
        # in: b in a
        if a.contains(b) or (a.min.x <= b.min.x + tol and a.max.x >= b.max.x - tol
                              and a.min.y <= b.min.y + tol and a.max.y >= b.max.y - tol
                              and a.min.z <= b.min.z + tol and a.max.z >= b.max.z - tol):
            return SpatialRelation(id_b, id_a, "in", confidence=1.0)

        # touching
        if not a.intersects(b):
            dx = max(a.min.x - b.max.x, b.min.x - a.max.x, 0)
            dy = max(a.min.y - b.max.y, b.min.y - a.max.y, 0)
            dz = max(a.min.z - b.max.z, b.min.z - a.max.z, 0)
            dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if dist <= tol:
                return SpatialRelation(id_a, id_b, "touching", confidence=1.0 - dist / (tol + 1e-9))

        # above / below（有水平重叠但未接触）
        if overlap_ratio > 0.1:
            if a_bottom > b_top + tol:
                return SpatialRelation(id_a, id_b, "above", confidence=overlap_ratio)
            if b_bottom > a_top + tol:
                return SpatialRelation(id_b, id_a, "above", confidence=overlap_ratio)

        # next_to
        h_dist = a.horizontal_distance(b)
        max_span = max(
            a.max.x - a.min.x + b.max.x - b.min.x,
            a.max.y - a.min.y + b.max.y - b.min.y,
        ) / 2
        if h_dist < max_span * 1.5 + tol:
            return SpatialRelation(
                id_a, id_b, "next_to",
                confidence=max(0.0, 1.0 - h_dist / (max_span * 1.5 + 1e-9)),
            )

        return None

    @staticmethod
    def _build_spatial_buckets(
        aabbs: Dict[str, AABB],
        cell_size: float,
    ) -> Dict[Tuple[int, int], List[str]]:
        """按 AABB 中心在 2D 水平面分桶。"""
        buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for obj_id, aabb in aabbs.items():
            cx = (aabb.min.x + aabb.max.x) / 2
            cy = (aabb.min.y + aabb.max.y) / 2
            key = (int(math.floor(cx / cell_size)), int(math.floor(cy / cell_size)))
            buckets[key].append(obj_id)
        return buckets

    def compute_relations(self, spatial_tolerance: float = 0.01) -> List[SpatialRelation]:
        """基于 AABB 启发式自动计算空间关系（空间分桶优化 O(n*k)）

        规则：
        - on: A 的底部接近 B 的顶部，且水平投影重叠 > 50%
        - in: A 的包围盒完全在 B 的包围盒内（容差内）
        - next_to: 水平中心距离 < max(A_size, B_size) * 1.5，且不满足 on/in
        - above: A 的底部 > B 的顶部，水平有重叠但不满足 on
        - below: A 的顶部 < B 的底部，水平有重叠
        - touching: AABB 刚好接触（距离 < tolerance）
        """
        if not self._built:
            self.build()

        aabbs: Dict[str, AABB] = {}
        for obj in self._objects.values():
            aabb = AABB.from_world_object(obj)
            if aabb:
                aabbs[obj.obj_id] = aabb

        if len(aabbs) < 2:
            return []

        # 小场景直接暴力比较，避免分桶开销
        if len(aabbs) <= 50:
            return self._compute_relations_brute_force(aabbs, spatial_tolerance)

        # 空间分桶：按水平面 2D 网格，桶大小根据最大可能关系距离设定
        # next_to 最大阈值约为 max_span * 1.5（大物件 ~3m），取 1m 桶保证 3x3 邻域覆盖
        cell_size = max(1.0, spatial_tolerance * 20)
        buckets = self._build_spatial_buckets(aabbs, cell_size)
        sorted_cells = sorted(buckets.keys())

        relations: List[SpatialRelation] = []
        tol = spatial_tolerance

        for i, cell_key in enumerate(sorted_cells):
            cell_objs = buckets[cell_key]
            # 同桶内全比较（i < j 天然去重）
            for j in range(len(cell_objs)):
                for k in range(j + 1, len(cell_objs)):
                    rel = self._compute_pair_relations(cell_objs[j], cell_objs[k], aabbs, tol)
                    if rel:
                        relations.append(rel)

            # 邻桶比较：只与排序后更靠后的 cell 比较，避免重复
            cx, cy = cell_key
            for later_cell in sorted_cells[i + 1 :]:
                if abs(later_cell[0] - cx) > 1 or abs(later_cell[1] - cy) > 1:
                    continue
                neighbor_objs = buckets[later_cell]
                for id_a in cell_objs:
                    for id_b in neighbor_objs:
                        rel = self._compute_pair_relations(id_a, id_b, aabbs, tol)
                        if rel:
                            relations.append(rel)

        return relations

    @staticmethod
    def _compute_relations_brute_force(
        aabbs: Dict[str, AABB],
        spatial_tolerance: float,
    ) -> List[SpatialRelation]:
        """小场景暴力比较（n <= 50 时更快，无分桶/去重开销）"""
        obj_ids = list(aabbs.keys())
        relations: List[SpatialRelation] = []
        tol = spatial_tolerance
        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                rel = SceneGraph._compute_pair_relations(obj_ids[i], obj_ids[j], aabbs, tol)
                if rel:
                    relations.append(rel)
        return relations
