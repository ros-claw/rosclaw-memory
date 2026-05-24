"""
EmbodiedMemory — ROSClaw-Memory 的核心封装类

在 PowerMem `Memory` 之上提供具身智能接口：
- `add_atom()`: 写入 MemoryAtom（自动更新空间/时间/具身扩展表）
- `search()`: 语义 + 空间 + 时间的混合检索
- `ingest()`: 传感器流接入（通过 IngestPipeline）
- `get_atom()`: 读取并还原为 MemoryAtom

向后兼容：所有 PowerMem.Memory API 通过 `.memory` 属性直接暴露。
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from powermem.core.memory import Memory

from .embodied_plugin import EmbodiedIntelligencePlugin
from .ingest_pipeline import IngestPipeline, SensorFrame
from .memory_atom import MemoryAtom
from .model_store import ModelStore
from .schema import initialize_embodied_schema
from .spatial_index import SpatialIndex
from .surprisal_gate import SurprisalGate
from .temporal_index import TemporalIndex
from .trajectory_similarity import (
    dtw_distance_normalized,
    signature_compatible,
    trajectory_feature_signature,
)
from .types import IntervalRelation, TemporalInterval, Vec3

logger = logging.getLogger(__name__)


class EmbodiedMemory:
    """具身记忆管理器

    Args:
        memory: PowerMem Memory 实例
        db_conn: 底层数据库连接（用于空间/时间索引和具身扩展表）
        voxel_size: 空间体素大小（米，默认 0.1）
        enable_plugin: 是否启用 EmbodiedIntelligencePlugin（默认 True）
        plugin_config: 插件配置 dict
    """

    def __init__(
        self,
        memory: Memory,
        db_conn: Any,
        voxel_size: float = 0.1,
        enable_plugin: bool = True,
        plugin_config: Optional[Dict[str, Any]] = None,
    ):
        self.memory = memory
        self.db_conn = db_conn

        # 初始化具身扩展 schema
        initialize_embodied_schema(db_conn)

        # 空间/时间索引
        self.spatial_index = SpatialIndex(db_conn, voxel_size=voxel_size)
        self.spatial_index.rebuild_from_db()
        self.temporal_index = TemporalIndex(db_conn)

        # 插件
        self._plugin: Optional[EmbodiedIntelligencePlugin] = None
        if enable_plugin:
            cfg = plugin_config or {"enabled": True}
            self._plugin = EmbodiedIntelligencePlugin(cfg)

        # 物理模型存储
        self.model_store = ModelStore(db_conn)

        # 接入管线（延迟初始化，需要 memory_store 回调）
        self._pipeline: Optional[IngestPipeline] = None

    # ========================================================================
    # 写入接口
    # ========================================================================

    def add_atom(self, atom: MemoryAtom, infer: bool = False) -> int:
        """写入一个 MemoryAtom

        流程：
        1. 通过 EmbodiedIntelligencePlugin.on_add 预处理
        2. 写入 PowerMem 主表（获得 memory_id）
        3. 写入 embodied_memories 扩展表
        4. 更新空间索引（Voxel Hash）
        5. 写入因果边

        Args:
            atom: 记忆原子
            infer: 是否启用 PowerMem 的 LLM 推理（默认 False，具身记忆通常已有内容）

        Returns:
            memory_id: Snowflake ID
        """
        # 1. 插件预处理
        if self._plugin and self._plugin.enabled:
            plugin_result = self._plugin.on_add(
                content=atom.content,
                metadata=atom.to_metadata(),
            )
            if plugin_result and "metadata" in plugin_result:
                # 将插件增强后的 metadata 回填到 atom
                atom = MemoryAtom.from_metadata(
                    atom.content,
                    plugin_result["metadata"],
                    memory_id=atom.memory_id,
                    user_id=atom.user_id or self.memory.agent_id,
                    agent_id=atom.agent_id or self.memory.agent_id,
                )

        # 2. 写入 PowerMem
        payload = atom.to_powermem_payload()
        if atom.memory_id is not None:
            # 已存在 memory_id，尝试更新
            if hasattr(self.memory, 'update'):
                self.memory.update(
                    memory_id=atom.memory_id,
                    content=payload['content'],
                    metadata=payload.get('metadata'),
                    user_id=payload.get('user_id'),
                    agent_id=payload.get('agent_id'),
                )
                memory_id = atom.memory_id
            else:
                memory_id = self.memory.storage.add_memory(payload)
        else:
            memory_id = self.memory.storage.add_memory(payload)
        atom.memory_id = memory_id

        # 3. 写入具身扩展表
        self._insert_embodied_record(atom)

        # 4. 空间索引
        if atom.spatial is not None:
            voxel_key = atom.spatial_voxel_key or atom.compute_voxel_key(
                self.spatial_index.voxel.voxel_size
            )
            self.spatial_index.add(
                memory_id=memory_id,
                position=atom.spatial,
                frame_id=atom.spatial_frame_id,
                voxel_key=voxel_key,
            )

        # 5. 因果边
        if atom.causal_parents:
            self._insert_causal_edges(memory_id, atom.causal_parents)

        logger.debug("Added MemoryAtom id=%s spatial=%s", memory_id, atom.spatial)
        return memory_id

    def _insert_embodied_record(self, atom: MemoryAtom) -> None:
        """将 MemoryAtom 的具身字段写入 embodied_memories 表"""
        cursor = self.db_conn.cursor()
        sql = """
            INSERT INTO embodied_memories (
                memory_id, spatial_x, spatial_y, spatial_z, spatial_voxel_key, spatial_frame_id,
                temporal_start, temporal_end, temporal_frame_id,
                modality, feature_vec_hash, raw_data_hash,
                entity_id, physical_type,
                uncertainty_type, uncertainty_std, uncertainty_confidence,
                salience, valence, arousal,
                action_type, prediction_error, embodied_meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                spatial_x = excluded.spatial_x,
                spatial_y = excluded.spatial_y,
                spatial_z = excluded.spatial_z,
                spatial_voxel_key = excluded.spatial_voxel_key,
                updated_at = CURRENT_TIMESTAMP
        """
        params = self._atom_to_sql_params(atom)
        try:
            cursor.execute(sql, params)
            self.db_conn.commit()
        except Exception as e:
            logger.warning("Failed to insert embodied record for id=%s: %s", atom.memory_id, e)

    def _atom_to_sql_params(self, atom: MemoryAtom) -> Tuple[Any, ...]:
        """将 MemoryAtom 转为 SQL 参数元组"""
        spatial = atom.spatial
        temporal = atom.temporal
        perceptual = atom.perceptual
        physical = atom.physical
        uncertainty = atom.uncertainty
        affective = atom.affective

        # physical_type 优先级：meta 显式指定 > physical 存在 > perceptual 存在 > unknown
        physical_type = atom.embodied_meta.get("physical_type")
        if physical_type is None:
            physical_type = "invariant" if physical else ("snapshot" if perceptual else "unknown")

        return (
            atom.memory_id,
            spatial.x if spatial else None,
            spatial.y if spatial else None,
            spatial.z if spatial else None,
            atom.spatial_voxel_key,
            atom.spatial_frame_id,
            temporal.start_sec if temporal else None,
            temporal.end_sec if temporal else None,
            temporal.frame_id if temporal else "wall_clock",
            perceptual.modality.value if perceptual else None,
            None,  # feature_vec_hash — 可由应用层计算
            perceptual.raw_data_hash if perceptual else None,
            physical.entity_id if physical else atom.embodied_meta.get("entity_id"),
            physical_type,
            uncertainty.type.value if uncertainty else None,
            uncertainty.std if uncertainty else None,
            uncertainty.confidence if uncertainty else None,
            affective.salience if affective else None,
            affective.valence if affective else None,
            affective.arousal if affective else None,
            atom.action.value,
            atom.prediction_error,
            json.dumps(atom.embodied_meta) if atom.embodied_meta else "{}",
        )

    def _insert_causal_edges(self, effect_id: int, cause_ids: List[int]) -> None:
        cursor = self.db_conn.cursor()
        sql = "INSERT INTO embodied_causal_edges (cause_memory_id, effect_memory_id) VALUES (?, ?)"
        for cause_id in cause_ids:
            try:
                cursor.execute(sql, (cause_id, effect_id))
            except Exception as e:
                logger.warning("Failed to insert causal edge %s -> %s: %s", cause_id, effect_id, e)
        self.db_conn.commit()

    # ========================================================================
    # 读取接口
    # ========================================================================

    def get_atom(self, memory_id: int) -> Optional[MemoryAtom]:
        """读取记忆并还原为 MemoryAtom"""
        raw = self.memory.storage.get_memory(memory_id)
        if raw is None:
            return None
        return MemoryAtom.from_metadata(
            content=raw.get("data", raw.get("content", "")),
            metadata=raw.get("metadata", {}),
            memory_id=memory_id,
            user_id=raw.get("user_id"),
            agent_id=raw.get("agent_id"),
            run_id=raw.get("run_id"),
            created_at=raw.get("created_at"),
            updated_at=raw.get("updated_at"),
        )

    # ========================================================================
    # 检索接口
    # ========================================================================

    def search(
        self,
        query: str,
        spatial_center: Optional[Vec3] = None,
        spatial_radius: Optional[float] = None,
        temporal_interval: Optional[TemporalInterval] = None,
        temporal_relation: Optional[IntervalRelation] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 30,
    ) -> List[MemoryAtom]:
        """混合检索：语义 + 空间 + 时间

        策略：
        1. 先用 PowerMem 语义检索获取候选集
        2. 如有空间约束，用空间索引过滤
        3. 如有时间约束，用时间索引过滤
        4. 取交集，按综合得分排序
        """
        # 1. 语义检索
        semantic_result = self.memory.search(query, filters=filters, limit=limit * 3)
        candidates: Dict[int, Dict[str, Any]] = {}
        for item in semantic_result.get("results", []):
            mid = item.get("id")
            if mid is not None:
                candidates[int(mid)] = {"semantic_score": item.get("score", 0.0), "raw": item}

        # 2. 空间过滤
        if spatial_center is not None and spatial_radius is not None:
            spatial_hits = self.spatial_index.query_radius(
                spatial_center, spatial_radius, limit=limit * 3
            )
            spatial_ids = {mid for mid, _ in spatial_hits}
            candidates = {k: v for k, v in candidates.items() if k in spatial_ids}

        # 3. 时间过滤
        if temporal_interval is not None:
            if temporal_relation is None:
                temporal_hits = self.temporal_index.query_overlapping(
                    temporal_interval, limit=limit * 3
                )
            else:
                temporal_hits = self.temporal_index.query(
                    temporal_interval, temporal_relation, limit=limit * 3
                )
            temporal_ids = {mid for mid, _ in temporal_hits}
            candidates = {k: v for k, v in candidates.items() if k in temporal_ids}

        # 4. 加载为 MemoryAtom 并排序（简单策略：语义分降序）
        results: List[Tuple[MemoryAtom, float]] = []
        for mid, info in candidates.items():
            atom = self.get_atom(mid)
            if atom:
                results.append((atom, info["semantic_score"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return [atom for atom, _ in results[:limit]]

    def search_near(
        self,
        center: Vec3,
        radius: float,
        frame_id: str = "world",
        limit: int = 30,
    ) -> List[MemoryAtom]:
        """纯空间范围查询"""
        hits = self.spatial_index.query_radius(center, radius, frame_id, limit=limit)
        atoms: List[MemoryAtom] = []
        for mid, dist in hits:
            atom = self.get_atom(mid)
            if atom:
                atom.embodied_meta["_spatial_distance"] = dist
                atoms.append(atom)
        return atoms

    def search_temporal(
        self,
        interval: TemporalInterval,
        relation: Optional[IntervalRelation] = None,
        frame_id: Optional[str] = None,
        limit: int = 30,
    ) -> List[MemoryAtom]:
        """纯时间区间查询（默认任意重叠）"""
        if relation is None:
            hits = self.temporal_index.query_overlapping(interval, frame_id, limit=limit)
        else:
            hits = self.temporal_index.query(interval, relation, frame_id, limit=limit)
        atoms: List[MemoryAtom] = []
        for mid, _ in hits:
            atom = self.get_atom(mid)
            if atom:
                atoms.append(atom)
        return atoms

    # ========================================================================
    # 传感器接入
    # ========================================================================

    def get_pipeline(self) -> IngestPipeline:
        """获取（或创建）传感器接入管线

        管线中的 SurprisalGate 会自动绑定数据库持久化回调，
        实现预测编码状态跨会话恢复。
        """
        if self._pipeline is None:
            gate = SurprisalGate(
                state_store=lambda pid, state: self.save_surprisal_state(pid, state),
                state_load=lambda pid: self.get_surprisal_state(pid),
            )
            self._pipeline = IngestPipeline(
                memory_store=self.add_atom,
                surprisal_gate=gate,
            )
        return self._pipeline

    def ingest(self, frame: SensorFrame, content: Optional[str] = None) -> Optional[int]:
        """便捷方法：单帧摄入"""
        return self.get_pipeline().ingest(frame, content)

    def flush_pipeline(self) -> Optional[int]:
        """强制刷新传感器缓冲"""
        return self.get_pipeline().flush()

    # ========================================================================
    # 物理模型管理
    # ========================================================================

    def save_model(
        self,
        result: "ParseResult",
        model_id: Optional[str] = None,
        model_type: str = "robot",
    ) -> str:
        """保存物理模型到数据库"""
        return self.model_store.save(result, model_id=model_id, model_type=model_type)

    def get_model(self, model_id: str) -> Optional[Any]:
        """读取存储的物理模型"""
        return self.model_store.load(model_id)

    def list_models(
        self,
        model_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """枚举存储的模型"""
        return self.model_store.list_models(model_type=model_type, limit=limit)

    def delete_model(self, model_id: str) -> bool:
        """删除物理模型"""
        return self.model_store.delete(model_id)

    def check_self_collision(self, model_id: str) -> List[Any]:
        """检测模型自碰撞"""
        return self.model_store.check_self_collision(model_id)

    # ========================================================================
    # 物理约束记忆
    # ========================================================================

    def add_constraint(self, constraint: "PhysicalConstraint", content: str = "") -> int:
        """将物理约束存储为 MemoryAtom

        Args:
            constraint: 物理约束
            content: 可选描述文本

        Returns:
            memory_id
        """
        from .memory_atom import MemoryAtom
        from .physical_model import PhysicalConstraint

        atom = MemoryAtom.from_constraint(
            content=content or constraint.description,
            constraint=constraint,
        )
        # 标记 physical_type 为 constraint，便于 SQL 查询
        atom.embodied_meta["physical_type"] = "constraint"
        atom.embodied_meta["entity_id"] = constraint.constraint_type
        return self.add_atom(atom)

    def search_constraints(
        self,
        center: Vec3,
        radius: float,
        constraint_type: Optional[str] = None,
        limit: int = 30,
    ) -> List[MemoryAtom]:
        """搜索指定空间区域内的物理约束

        策略：先用空间索引缩小范围，再过滤 physical_type="constraint"
        """
        hits = self.spatial_index.query_radius(center, radius, limit=limit * 3)
        atoms: List[MemoryAtom] = []
        for mid, _ in hits:
            atom = self.get_atom(mid)
            if atom is None:
                continue
            # 过滤约束类型
            if atom.embodied_meta.get("physical_type") != "constraint":
                continue
            if constraint_type is not None:
                c = atom.embodied_meta.get("constraint", {})
                if c.get("constraint_type") != constraint_type:
                    continue
            atoms.append(atom)
        return atoms[:limit]

    # ========================================================================
    # 动作/轨迹经验记忆
    # ========================================================================

    def record_action(
        self,
        content: str,
        action_type: str = "act",
        spatial: Optional[Vec3] = None,
        **kwargs,
    ) -> int:
        """记录动作执行（返回 memory_id，后续用于关联结果）"""
        from .memory_atom import MemoryAtom
        from .types import MemoryAction

        atom = MemoryAtom.from_action(
            content=content,
            action_type=MemoryAction.ACT,
            spatial=spatial,
            **kwargs,
        )
        return self.add_atom(atom)

    def record_outcome(
        self,
        action_id: int,
        content: str,
        outcome_status: str,
        spatial: Optional[Vec3] = None,
        **kwargs,
    ) -> int:
        """记录动作结果，并建立因果边 action_id -> outcome_id

        Args:
            action_id: 动作记忆的 memory_id
            content: 结果描述
            outcome_status: "success" | "collision" | "timeout" | "error"
            spatial: 结果发生的空间位置

        Returns:
            outcome memory_id
        """
        from .memory_atom import MemoryAtom
        from .types import MemoryAction

        atom = MemoryAtom.from_action(
            content=content,
            action_type=MemoryAction.CORRECT,  # 结果是对动作的校正/反馈
            spatial=spatial,
            outcome_status=outcome_status,
            **kwargs,
        )
        atom.causal_parents = [action_id]
        outcome_id = self.add_atom(atom)
        return outcome_id

    def search_similar_experiences(
        self,
        center: Vec3,
        radius: float,
        action_type: Optional[str] = None,
        limit: int = 30,
    ) -> List[MemoryAtom]:
        """检索空间区域内的历史动作经验（含结果）"""
        hits = self.spatial_index.query_radius(center, radius, limit=limit * 3)
        atoms: List[MemoryAtom] = []
        for mid, _ in hits:
            atom = self.get_atom(mid)
            if atom is None:
                continue
            if atom.action.value not in ("act", "correct"):
                continue
            if action_type is not None and atom.action.value != action_type:
                continue
            atoms.append(atom)
        return atoms[:limit]

    # ========================================================================
    # 世界对象记忆
    # ========================================================================

    def add_world_objects(self, parse_result: "ParseResult") -> List[int]:
        """将 ParseResult 中的 world_objects 批量存储为 MemoryAtom

        Args:
            parse_result: 解析结果，含 world_objects 列表

        Returns:
            所有新增的 memory_id 列表
        """
        ids: List[int] = []
        for obj in parse_result.world_objects:
            name = obj.get("name", "unknown_object")
            obj_type = obj.get("type", "unknown")
            content = f"World object: {name} ({obj_type})"
            atom = MemoryAtom.from_world_object(content=content, obj=obj)
            mid = self.add_atom(atom)
            ids.append(mid)
        return ids

    def search_world_objects(
        self,
        center: Vec3,
        radius: float,
        obj_type: Optional[str] = None,
        limit: int = 30,
    ) -> List[MemoryAtom]:
        """搜索指定空间区域内的世界对象

        Args:
            center: 查询中心
            radius: 查询半径（米）
            obj_type: 可选过滤对象类型（如 "box", "sphere", "mesh"）
            limit: 最大返回数

        Returns:
            MemoryAtom 列表
        """
        hits = self.spatial_index.query_radius(center, radius, limit=limit * 3)
        atoms: List[MemoryAtom] = []
        for mid, dist in hits:
            atom = self.get_atom(mid)
            if atom is None:
                continue
            if atom.embodied_meta.get("physical_type") != "world_object":
                continue
            if obj_type is not None:
                wo = atom.embodied_meta.get("world_object", {})
                if wo.get("type") != obj_type:
                    continue
            atom.embodied_meta["_spatial_distance"] = dist
            atoms.append(atom)
        return atoms[:limit]

    # ========================================================================
    # 轨迹记忆
    # ========================================================================

    def record_trajectory(
        self,
        content: str,
        waypoints: List[Tuple[Vec3, float]],
        **kwargs,
    ) -> int:
        """记录一条轨迹（运动/操作轨迹）

        Args:
            content: 轨迹描述，如 "move from A to B"
            waypoints: [(Vec3 position, float timestamp_sec), ...]

        Returns:
            memory_id
        """
        atom = MemoryAtom.from_trajectory(
            content=content,
            waypoints=waypoints,
            **kwargs,
        )
        return self.add_atom(atom)

    def search_trajectory_near(
        self,
        center: Vec3,
        radius: float,
        temporal_interval: Optional[TemporalInterval] = None,
        temporal_relation: Optional[IntervalRelation] = None,
        limit: int = 30,
    ) -> List[MemoryAtom]:
        """检索经过指定空间区域的轨迹

        策略：
        1. 先用空间索引粗筛（基于轨迹中点）
        2. 对候选轨迹，逐路点精确判断是否有路点落入查询范围
        3. 如有时间约束，再用时间索引过滤

        Args:
            center: 空间查询中心
            radius: 查询半径（米）
            temporal_interval: 可选时间约束
            temporal_relation: Allen 区间关系
            limit: 最大返回数

        Returns:
            MemoryAtom 列表（按最近路点距离排序）
        """
        # 1. 空间粗筛（扩大半径以覆盖轨迹两端可能偏离中点的情况）
        coarse_radius = radius * 2.0
        hits = self.spatial_index.query_radius(center, coarse_radius, limit=limit * 5)

        results: List[Tuple[MemoryAtom, float]] = []
        for mid, _ in hits:
            atom = self.get_atom(mid)
            if atom is None:
                continue
            traj = atom.embodied_meta.get("trajectory")
            if not traj:
                continue

            # 逐路点精确判断
            min_dist: Optional[float] = None
            for wp in traj.get("waypoints", []):
                pos_dict = wp.get("position")
                if not pos_dict:
                    continue
                pos = Vec3.from_dict(pos_dict)
                dist = center.distance_to(pos)
                if min_dist is None or dist < min_dist:
                    min_dist = dist

            if min_dist is not None and min_dist <= radius:
                results.append((atom, min_dist))

        # 2. 时间过滤（默认任意重叠）
        if temporal_interval is not None:
            if temporal_relation is None:
                temporal_hits = self.temporal_index.query_overlapping(
                    temporal_interval, limit=limit * 5
                )
            else:
                temporal_hits = self.temporal_index.query(
                    temporal_interval, temporal_relation, limit=limit * 5
                )
            temporal_ids = {mid for mid, _ in temporal_hits}
            results = [(atom, dist) for atom, dist in results if atom.memory_id in temporal_ids]

        # 3. 按最近路点距离排序
        results.sort(key=lambda x: x[1])
        for atom, dist in results:
            atom.embodied_meta["_nearest_waypoint_distance"] = dist
        return [atom for atom, _ in results[:limit]]

    def search_trajectory_temporal(
        self,
        interval: TemporalInterval,
        relation: Optional[IntervalRelation] = None,
        limit: int = 30,
    ) -> List[MemoryAtom]:
        """纯时间区间检索轨迹

        默认使用"任意重叠"语义（query_overlapping），即只要轨迹时间区间
        与查询区间有交集即命中。可通过 relation 参数指定具体的 Allen 关系。
        """
        if relation is None:
            hits = self.temporal_index.query_overlapping(interval, limit=limit * 3)
        else:
            hits = self.temporal_index.query(interval, relation, limit=limit * 3)
        atoms: List[MemoryAtom] = []
        for mid, _ in hits:
            atom = self.get_atom(mid)
            if atom is None:
                continue
            if "trajectory" not in atom.embodied_meta:
                continue
            atoms.append(atom)
        return atoms[:limit]

    def search_similar_trajectories(
        self,
        query_waypoints: List[Tuple[Vec3, float]],
        spatial_center: Optional[Vec3] = None,
        spatial_radius: Optional[float] = None,
        temporal_interval: Optional[TemporalInterval] = None,
        top_k: int = 10,
        max_dtw_distance: Optional[float] = None,
    ) -> List[Tuple[MemoryAtom, float]]:
        """检索与查询轨迹形状相似的轨迹记忆

        策略：
        1. 空间/时间粗筛（利用现有索引）
        2. 轨迹特征签名预过滤
        3. DTW 精排

        Args:
            query_waypoints: 查询轨迹路点 [(Vec3, timestamp_sec), ...]
            spatial_center: 可选空间查询中心（粗筛轨迹中点）
            spatial_radius: 可选空间查询半径
            temporal_interval: 可选时间区间（粗筛）
            top_k: 返回最大数量
            max_dtw_distance: 可选 DTW 距离上限

        Returns:
            [(MemoryAtom, dtw_distance), ...] 按距离升序
        """
        if not query_waypoints:
            return []

        query_positions = [wp[0] for wp in query_waypoints]
        query_sig = trajectory_feature_signature(query_waypoints)

        # 1. 粗筛候选集
        candidate_ids: Set[int] = set()

        # 空间粗筛
        if spatial_center is not None and spatial_radius is not None:
            spatial_hits = self.spatial_index.query_radius(
                spatial_center, spatial_radius, limit=top_k * 20
            )
            candidate_ids = {mid for mid, _ in spatial_hits}

        # 时间粗筛
        if temporal_interval is not None:
            temporal_hits = self.temporal_index.query_overlapping(
                temporal_interval, limit=top_k * 20
            )
            temporal_ids = {mid for mid, _ in temporal_hits}
            if candidate_ids:
                candidate_ids &= temporal_ids
            else:
                candidate_ids = temporal_ids

        # 如果没有任何粗筛条件，从 DB 拉取所有轨迹记忆（带警告）
        if not candidate_ids:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT memory_id FROM embodied_memories WHERE embodied_meta LIKE '%trajectory%' LIMIT 1000"
            )
            candidate_ids = {int(row[0]) for row in cursor.fetchall()}
            if len(candidate_ids) >= 1000:
                logger.warning(
                    "search_similar_trajectories: no coarse filters applied, "
                    "DTW computed against %d candidates. This is expensive.",
                    len(candidate_ids),
                )

        # 2. 签名预过滤 + DTW 精排
        results: List[Tuple[MemoryAtom, float]] = []
        for mid in candidate_ids:
            atom = self.get_atom(mid)
            if atom is None:
                continue
            traj_meta = atom.embodied_meta.get("trajectory")
            if not traj_meta:
                continue

            cand_waypoints = []
            for wp in traj_meta.get("waypoints", []):
                pos = wp.get("position")
                if pos:
                    cand_waypoints.append((Vec3.from_dict(pos), wp.get("timestamp_sec", 0.0)))

            if not cand_waypoints:
                continue

            # 签名预过滤
            cand_sig = trajectory_feature_signature(cand_waypoints)
            if not signature_compatible(query_sig, cand_sig):
                continue

            cand_positions = [wp[0] for wp in cand_waypoints]
            dtw = dtw_distance_normalized(query_positions, cand_positions)

            if max_dtw_distance is not None and dtw > max_dtw_distance:
                continue

            results.append((atom, dtw))

        results.sort(key=lambda x: x[1])
        for atom, dtw in results:
            atom.embodied_meta["_dtw_distance"] = dtw
        return results[:top_k]

    # ========================================================================
    # 预测编码持久化
    # ========================================================================

    def get_surprisal_state(self, predictor_id: str) -> Optional[Dict[str, Any]]:
        """从数据库读取指定预测器的持久化状态（兼容 SurprisalGate _RunningStats）

        SurprisalGate 期望的格式：{"count": int, "mean": float, "m2": float}
        """
        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT window_mean, window_std, window_count "
            "FROM embodied_predictive_state WHERE predictor_id = ?",
            (predictor_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        mean, std, count = row
        count = int(count or 0)
        mean = float(mean or 0.0)
        std = float(std or 0.0)
        # m2 = std^2 * count（Welford 算法的反向推导）
        m2 = (std ** 2) * count if count > 0 else 0.0
        return {"count": count, "mean": mean, "m2": m2}

    def save_surprisal_state(self, predictor_id: str, state: Dict[str, Any]) -> None:
        """将预测器状态写入数据库

        state 来自 SurprisalGate _RunningStats.to_dict()：
        {"count": int, "mean": float, "m2": float}
        """
        cursor = self.db_conn.cursor()
        count = int(state.get("count", 0))
        mean = float(state.get("mean", 0.0))
        m2 = float(state.get("m2", 0.0))
        std = math.sqrt(m2 / count) if count > 0 else 0.0
        threshold = mean + 3.0 * std if count > 0 else 0.0

        sql = """
            INSERT INTO embodied_predictive_state (
                predictor_id, last_prediction, last_update_sec,
                dynamic_threshold, window_mean, window_std,
                window_count, update_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(predictor_id) DO UPDATE SET
                last_prediction = excluded.last_prediction,
                last_update_sec = excluded.last_update_sec,
                dynamic_threshold = excluded.dynamic_threshold,
                window_mean = excluded.window_mean,
                window_std = excluded.window_std,
                window_count = excluded.window_count,
                update_count = excluded.update_count,
                updated_at = CURRENT_TIMESTAMP
        """
        params = (
            predictor_id,
            None,  # last_prediction — 暂不支持 JSON 序列化
            0.0,
            threshold,
            mean,
            std,
            count,
            count,
        )
        try:
            cursor.execute(sql, params)
            self.db_conn.commit()
        except Exception as e:
            logger.warning("Failed to save surprisal state for %s: %s", predictor_id, e)

    def reset_surprisal_state(self, predictor_id: Optional[str] = None) -> None:
        """重置指定或全部预测器持久化状态"""
        cursor = self.db_conn.cursor()
        try:
            if predictor_id is None:
                cursor.execute("DELETE FROM embodied_predictive_state")
            else:
                cursor.execute(
                    "DELETE FROM embodied_predictive_state WHERE predictor_id = ?",
                    (predictor_id,),
                )
            self.db_conn.commit()
        except Exception as e:
            logger.warning("Failed to reset surprisal state: %s", e)

    # ========================================================================
    # 因果图
    # ========================================================================

    def get_causes(self, memory_id: int, limit: int = 10) -> List[MemoryAtom]:
        """获取指定记忆的原因记忆"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT cause_memory_id FROM embodied_causal_edges WHERE effect_memory_id = ? LIMIT ?",
            (memory_id, limit),
        )
        atoms: List[MemoryAtom] = []
        for (cause_id,) in cursor.fetchall():
            atom = self.get_atom(int(cause_id))
            if atom:
                atoms.append(atom)
        return atoms

    def get_effects(self, memory_id: int, limit: int = 10) -> List[MemoryAtom]:
        """获取指定记忆的结果记忆"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT effect_memory_id FROM embodied_causal_edges WHERE cause_memory_id = ? LIMIT ?",
            (memory_id, limit),
        )
        atoms: List[MemoryAtom] = []
        for (effect_id,) in cursor.fetchall():
            atom = self.get_atom(int(effect_id))
            if atom:
                atoms.append(atom)
        return atoms

    # ========================================================================
    # 生命周期
    # ========================================================================

    def delete_atom(self, memory_id: int) -> bool:
        """删除记忆（PowerMem + 具身扩展 + 索引）"""
        # 清理空间索引
        self.spatial_index.remove(memory_id)
        # PowerMem 删除（级联删除 embodied_memories via FK）
        try:
            return self.memory.delete(memory_id)
        except Exception as e:
            logger.warning("Failed to delete memory %s: %s", memory_id, e)
            return False

    def stats(self) -> Dict[str, Any]:
        """获取具身记忆系统统计"""
        return {
            "spatial": self.spatial_index.stats(),
            "pipeline": self._pipeline.get_stats() if self._pipeline else None,
            "plugin": self._plugin.get_surprisal_state() if self._plugin else None,
        }

    # ========================================================================
    # 向后兼容代理
    # ========================================================================

    def __getattr__(self, name: str) -> Any:
        """将未识别的属性访问代理到底层 PowerMem Memory"""
        return getattr(self.memory, name)
