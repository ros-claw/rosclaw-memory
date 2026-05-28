"""
空间索引 — Voxel Hash + SeekDB B-Tree 混合索引

设计目标：
1. 写性能：Voxel Hash O(1) 插入，无需 R-Tree 复杂分裂
2. 读性能：精确点查用 Voxel Hash，范围查用 SeekDB B-Tree
3. 持久化：Voxel Hash 在内存，SeekDB 在磁盘，启动时从 DB 重建

坐标系：
- 所有坐标存储在 world 坐标系（默认）
- frame_id 字段支持多坐标系（如 "map", "base_link", "camera_left"）
"""

from __future__ import annotations

import logging
import math
import heapq
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .types import Vec3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voxel Hash
# ---------------------------------------------------------------------------

class VoxelHash:
    """内存中的三维体素哈希表

    每个体素是一个 "桶"，存储落在此区域内的 memory_id 集合。
    查询时：先计算目标体素键，再查桶内精确距离过滤。

    内部使用 (vx, vy, vz, frame_id) 元组作为键，避免字符串格式化/解析开销。
    """

    def __init__(self, voxel_size: float = 0.1):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = voxel_size
        # cell_key -> {memory_id}
        self._buckets: Dict[Tuple[int, int, int, str], Set[int]] = defaultdict(set)
        # memory_id -> cell_key（反向索引，用于删除）
        self._id_to_cell: Dict[int, Tuple[int, int, int, str]] = {}

    def insert(self, memory_id: int, position: Vec3, frame_id: str = "world") -> Tuple[int, int, int, str]:
        """插入一个记忆点到 Voxel Hash，返回 cell_key"""
        vs = self.voxel_size
        cell = (int(math.floor(position.x / vs)), int(math.floor(position.y / vs)), int(math.floor(position.z / vs)), frame_id)
        # 如果已存在不同 key，先从旧桶移除
        old = self._id_to_cell.get(memory_id)
        if old is not None and old != cell:
            self._buckets[old].discard(memory_id)

        self._buckets[cell].add(memory_id)
        self._id_to_cell[memory_id] = cell
        return cell

    def remove(self, memory_id: int) -> bool:
        """删除记忆点，返回是否成功"""
        cell = self._id_to_cell.pop(memory_id, None)
        if cell is None:
            return False
        self._buckets[cell].discard(memory_id)
        if not self._buckets[cell]:
            del self._buckets[cell]
        return True

    def query_near(
        self,
        center: Vec3,
        radius: float,
        frame_id: str = "world",
    ) -> Set[int]:
        """球形范围查询 — 返回候选 memory_id 集合（含体素近似，需二次精确过滤）"""
        vs = self.voxel_size
        r_cells = int(math.ceil(radius / vs))
        cx = int(math.floor(center.x / vs))
        cy = int(math.floor(center.y / vs))
        cz = int(math.floor(center.z / vs))

        candidates: Set[int] = set()
        total_voxels = (2 * r_cells + 1) ** 3
        n_items = len(self._id_to_cell)

        # 稀疏数据优化：当存储对象数远少于需扫描体素数时，直接遍历对象而非空体素
        if n_items < total_voxels:
            for memory_id, (vx, vy, vz, kf) in self._id_to_cell.items():
                if kf != frame_id:
                    continue
                if abs(vx - cx) <= r_cells and abs(vy - cy) <= r_cells and abs(vz - cz) <= r_cells:
                    candidates.add(memory_id)
        else:
            _buckets = self._buckets
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    for dz in range(-r_cells, r_cells + 1):
                        key = (cx + dx, cy + dy, cz + dz, frame_id)
                        candidates.update(_buckets.get(key, set()))
        return candidates

    def query_exact(
        self,
        center: Vec3,
        radius: float,
        frame_id: str = "world",
        id_to_position: Optional[Dict[int, Vec3]] = None,
        limit: int = 100,
    ) -> List[Tuple[int, float]]:
        """精确球形范围查询 — 返回 [(memory_id, distance), ...] 按距离升序

        使用平方距离过滤避免 sqrt，并用堆实现 O(N log K) 部分排序。
        """
        candidates = self.query_near(center, radius, frame_id)
        radius_sq = radius * radius

        # Collect valid hits with squared distance first (no sqrt)
        hits: List[Tuple[int, float]] = []
        for mid in candidates:
            if id_to_position is not None:
                pos = id_to_position.get(mid)
                if pos is None:
                    continue
            else:
                hits.append((mid, 0.0))
                continue
            dist_sq = center.distance_to_sq(pos)
            if dist_sq <= radius_sq:
                hits.append((mid, math.sqrt(dist_sq)))

        # Partial sort with heapq — O(N log K) instead of O(N log N)
        if len(hits) <= limit:
            hits.sort(key=lambda x: x[1])
            return hits
        return heapq.nsmallest(limit, hits, key=lambda x: x[1])

    def get_all_ids(self) -> Set[int]:
        return set(self._id_to_cell.keys())

    def stats(self) -> Dict[str, Any]:
        total_ids = len(self._id_to_cell)
        total_buckets = len(self._buckets)
        avg_load = total_ids / max(total_buckets, 1)
        max_load = max((len(ids) for ids in self._buckets.values()), default=0)
        return {
            "voxel_size": self.voxel_size,
            "total_ids": total_ids,
            "total_buckets": total_buckets,
            "avg_load": round(avg_load, 2),
            "max_load": max_load,
        }


# ---------------------------------------------------------------------------
# 空间索引管理器（Voxel Hash + SeekDB）
# ---------------------------------------------------------------------------

class SpatialIndex:
    """混合空间索引管理器

    职责：
    - 维护内存 Voxel Hash 用于快速点查和范围查
    - 通过 SeekDB 的 B-Tree 索引持久化空间坐标
    - 启动时从 SeekDB 重建 Voxel Hash
    """

    def __init__(
        self,
        db_conn: Any,
        voxel_size: float = 0.1,
        table_name: str = "embodied_memories",
    ):
        self.db_conn = db_conn
        self.voxel = VoxelHash(voxel_size=voxel_size)
        self.table_name = table_name
        self._id_to_position: Dict[int, Vec3] = {}
        self._loaded = False

    def rebuild_from_db(self) -> None:
        """从 SeekDB 重建 Voxel Hash — 在初始化或崩溃恢复时调用"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            f"""
            SELECT memory_id, spatial_x, spatial_y, spatial_z, spatial_frame_id
            FROM {self.table_name}
            WHERE spatial_x IS NOT NULL
            """
        )
        count = 0
        for row in cursor.fetchall():
            memory_id, x, y, z, frame_id = row
            pos = Vec3(float(x), float(y), float(z))
            self.voxel.insert(int(memory_id), pos, str(frame_id or "world"))
            self._id_to_position[int(memory_id)] = pos
            count += 1
        self._loaded = True
        logger.info("SpatialIndex rebuilt from DB: %d entries", count)

    def add(
        self,
        memory_id: int,
        position: Vec3,
        frame_id: str = "world",
        voxel_key: Optional[str] = None,
    ) -> Tuple[int, int, int, str]:
        """添加空间索引条目（内存 + 异步持久化建议）

        注意：SeekDB 的 UPDATE 应由调用方在事务中执行，
        本方法只负责内存 Voxel Hash 和返回 cell_key。
        voxel_key 参数保留以兼容旧调用方，实际由 VoxelHash 内部重新计算。
        """
        cell = self.voxel.insert(memory_id, position, frame_id)
        self._id_to_position[memory_id] = position
        return cell

    def remove(self, memory_id: int) -> bool:
        """删除空间索引条目"""
        self._id_to_position.pop(memory_id, None)
        return self.voxel.remove(memory_id)

    def query_radius(
        self,
        center: Vec3,
        radius: float,
        frame_id: str = "world",
        limit: int = 100,
    ) -> List[Tuple[int, float]]:
        """球形范围查询，返回 [(memory_id, distance), ...] 按距离升序"""
        if not self._loaded:
            self.rebuild_from_db()
        results = self.voxel.query_exact(center, radius, frame_id, self._id_to_position, limit=limit)
        return results

    def get_all_ids(self) -> Set[int]:
        return self.voxel.get_all_ids()

    def query_nearest(
        self,
        center: Vec3,
        k: int = 10,
        frame_id: str = "world",
        max_radius: float = 10.0,
    ) -> List[Tuple[int, float]]:
        """k 近邻查询 — 在 max_radius 范围内找最近的 k 个"""
        candidates = self.query_radius(center, max_radius, frame_id, limit=10000)
        return candidates[:k]

    def get_position(self, memory_id: int) -> Optional[Vec3]:
        return self._id_to_position.get(memory_id)

    def stats(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            **self.voxel.stats(),
        }
