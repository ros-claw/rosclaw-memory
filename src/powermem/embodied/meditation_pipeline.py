"""
MeditationPipeline — 机器人离线记忆抽象管道

三阶段熵减：
1. Entity Consolidation: 同一实体短时观测聚合为事件
2. Relation Crystallization: 增量式空间关系固化
3. Pattern Extraction: 高频动作-结果模式提取为抽象概念

在机器人空闲（充电/待机）时运行，把原始感知流转化为结构化经验。
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .types import SpatialRelation, Vec3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 报告类型
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MeditationReport:
    """冥想管道执行报告"""

    success: bool = True
    elapsed_sec: float = 0.0
    consolidated_count: int = 0
    crystallized_count: int = 0
    extracted_patterns: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "consolidated_count": self.consolidated_count,
            "crystallized_count": self.crystallized_count,
            "extracted_patterns": self.extracted_patterns,
            "errors": self.errors,
        }


@dataclass(slots=True)
class EntityEvent:
    """聚合后的实体事件"""

    entity_id: str
    representative_memory_id: int
    start_sec: float
    end_sec: float
    avg_position: Vec3
    observation_count: int
    action_types: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedPattern:
    """提取出的模式"""

    pattern_id: str
    dimension: str
    layer: int
    representative_memory_id: int
    support: int
    description: str


# ---------------------------------------------------------------------------
# Phase 1: Entity Consolidation
# ---------------------------------------------------------------------------

class EntityConsolidator:
    """实体观测聚合器

    把同一 entity_id 在短时间窗口内的多次 MemoryAtom 观测聚合为一个 EntityEvent。
    """

    def __init__(self, db_conn: Any):
        self.db_conn = db_conn

    def consolidate(self, window_sec: float = 30.0) -> List[EntityEvent]:
        """执行实体聚合

        策略：
        1. 查询所有带 entity_id 的 memory atoms
        2. 按 entity_id + 时间桶（window_sec）分组
        3. 每组生成一个 EntityEvent
        4. 将 EntityEvent 写入 concept_index
        """
        cursor = self.db_conn.cursor()
        # 获取所有带 entity_id 的 atoms，按 entity_id 和 created_at 排序
        cursor.execute(
            """
            SELECT memory_id, entity_id, spatial_x, spatial_y, spatial_z,
                   temporal_start, temporal_end, action_type, created_at
            FROM embodied_memories
            WHERE entity_id IS NOT NULL
            ORDER BY entity_id, temporal_start
            """
        )
        rows = cursor.fetchall()

        events: List[EntityEvent] = []
        current_group: List[Tuple] = []
        current_entity: Optional[str] = None

        def _flush_group():
            if not current_group or current_entity is None:
                return
            event = self._aggregate_group(current_entity, current_group)
            events.append(event)
            self._index_event(event)

        for row in rows:
            mid, eid, sx, sy, sz, tstart, tend, action, _ = row
            if eid != current_entity:
                _flush_group()
                current_entity = eid
                current_group = []
            elif current_group and tstart is not None:
                # 检查时间窗口是否超过阈值
                last_tstart = current_group[-1][5]  # temporal_start of last row
                if last_tstart is not None and abs(tstart - last_tstart) > window_sec:
                    _flush_group()
                    current_group = []
            current_group.append(row)

        _flush_group()
        logger.info("EntityConsolidator: %d events from %d observations", len(events), len(rows))
        return events

    def _aggregate_group(self, entity_id: str, group: List[Tuple]) -> EntityEvent:
        """将一组观测聚合为 EntityEvent"""
        rep_mid = int(group[0][0])
        starts = [r[5] for r in group if r[5] is not None]
        ends = [r[6] for r in group if r[6] is not None]
        xs = [r[2] for r in group if r[2] is not None]
        ys = [r[3] for r in group if r[3] is not None]
        zs = [r[4] for r in group if r[4] is not None]
        actions = list({r[7] for r in group if r[7] is not None})

        start_sec = min(starts) if starts else 0.0
        end_sec = max(ends) if ends else start_sec
        avg_x = sum(xs) / len(xs) if xs else 0.0
        avg_y = sum(ys) / len(ys) if ys else 0.0
        avg_z = sum(zs) / len(zs) if zs else 0.0

        return EntityEvent(
            entity_id=entity_id,
            representative_memory_id=rep_mid,
            start_sec=start_sec,
            end_sec=end_sec,
            avg_position=Vec3(avg_x, avg_y, avg_z),
            observation_count=len(group),
            action_types=actions,
        )

    def _index_event(self, event: EntityEvent) -> None:
        """将 EntityEvent 写入 concept_index"""
        cursor = self.db_conn.cursor()
        concept_id = f"{event.entity_id}_event_{int(event.start_sec)}"
        try:
            cursor.execute(
                """
                INSERT INTO embodied_concept_index (memory_id, dimension, layer, concept_id, confidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, dimension, concept_id) DO UPDATE SET
                    layer = excluded.layer,
                    confidence = excluded.confidence
                """,
                (event.representative_memory_id, "entity", 1, concept_id, 1.0),
            )
            self.db_conn.commit()
        except Exception as e:
            logger.warning("Failed to index entity event %s: %s", concept_id, e)


# ---------------------------------------------------------------------------
# Phase 2: Relation Crystallization
# ---------------------------------------------------------------------------

class RelationCrystallizer:
    """关系固化器

    对发生变化的世界对象，增量计算空间关系，避免全局 O(n²)。
    """

    def __init__(self, embodied_memory: Any):
        self.em = embodied_memory

    def crystallize(self, changed_obj_ids: Optional[List[str]] = None) -> List[SpatialRelation]:
        """执行关系固化

        Args:
            changed_obj_ids: 指定发生变化的对象 ID。为 None 时自动检测最近移动的对象。
        """
        if changed_obj_ids is None:
            changed_obj_ids = self._detect_changed_objects()

        if not changed_obj_ids:
            logger.info("RelationCrystallizer: no changed objects detected")
            return []

        crystallized: List[SpatialRelation] = []
        store = self.em.world_object_store

        for obj_id in changed_obj_ids:
            obj = store.load(obj_id)
            if obj is None or obj.pose is None:
                continue

            # 查询该对象 3m 内的邻居（用 store 的场景列表 + 距离过滤）
            if obj.scene_id:
                candidates = store.list_by_scene(obj.scene_id, limit=1000)
            else:
                candidates = []

            neighbors = []
            for other in candidates:
                if other.obj_id == obj_id:
                    continue
                dist = obj.pose.position.distance_to(other.pose.position)
                if dist <= 3.0:
                    neighbors.append(other)

            # 计算关系（复用 SceneGraph 的单对逻辑）
            from .scene_graph import AABB, SceneGraph
            for other in neighbors:
                aabb_a = AABB.from_world_object(obj)
                aabb_b = AABB.from_world_object(other)
                if aabb_a is None or aabb_b is None:
                    continue
                aabbs = {obj.obj_id: aabb_a, other.obj_id: aabb_b}
                rel = SceneGraph._compute_pair_relations(obj.obj_id, other.obj_id, aabbs, 0.01)
                if rel is not None:
                    store.add_relation(rel)
                    crystallized.append(rel)
                    # 同时写入 experience_graph
                    self._add_experience_relation(obj_id, other.obj_id, rel.relation)

        logger.info("RelationCrystallizer: %d relations for %d changed objects", len(crystallized), len(changed_obj_ids))
        return crystallized

    def _detect_changed_objects(self) -> List[str]:
        """检测最近有 pose 变化记录的世界对象"""
        cursor = self.em.db_conn.cursor()
        # 查询最近 1 小时内有 world_object_change 记录的对象
        cursor.execute(
            """
            SELECT DISTINCT embodied_meta
            FROM embodied_memories
            WHERE physical_type = 'world_object_change'
              AND (temporal_start > (strftime('%s', 'now') - 3600) OR temporal_start IS NULL)
            ORDER BY memory_id DESC
            """
        )
        changed: set = set()
        for (meta_json,) in cursor.fetchall():
            try:
                meta = json.loads(meta_json or "{}")
                oid = meta.get("world_object_id")
                if oid:
                    changed.add(oid)
            except Exception:
                continue
        return list(changed)

    def _add_experience_relation(self, subject_id: str, object_id: str, relation: str) -> None:
        """将空间关系也映射为 experience_graph 边"""
        # 查找这两个对象对应的最新 memory_id
        cursor = self.em.db_conn.cursor()
        cursor.execute(
            "SELECT memory_id FROM embodied_world_objects WHERE obj_id = ?",
            (subject_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return
        source_mid = row[0]

        cursor.execute(
            "SELECT memory_id FROM embodied_world_objects WHERE obj_id = ?",
            (object_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return
        target_mid = row[0]

        edge_type = self._relation_to_edge_type(relation)
        try:
            self.em.add_experience_edge(source_mid, target_mid, edge_type, 1.0)
        except Exception as e:
            logger.warning("Failed to add experience relation %s-%s: %s", subject_id, object_id, e)

    @staticmethod
    def _relation_to_edge_type(relation: str) -> str:
        mapping = {
            "on": "supports",
            "in": "contains",
            "next_to": "adjacent_to",
            "touching": "adjacent_to",
            "above": "supports",
            "below": "supports",
        }
        return mapping.get(relation, "adjacent_to")


# ---------------------------------------------------------------------------
# Phase 3: Pattern Extraction
# ---------------------------------------------------------------------------

class PatternExtractor:
    """模式提取器

    从 experience_graph 和 causal_edges 中挖掘高频模式，创建抽象概念节点。
    """

    def __init__(self, db_conn: Any):
        self.db_conn = db_conn

    def extract_patterns(self, lookback_hours: int = 24, min_support: int = 3) -> List[ExtractedPattern]:
        """提取高频模式

        策略：
        1. 查询最近 lookback_hours 内的 causes 边
        2. 按 (source_action_type, outcome_status) 聚合
        3. 支持度 >= min_support 的模式创建 concept_index 条目
        """
        cursor = self.db_conn.cursor()
        since_sec = time.time() - lookback_hours * 3600

        # 1. 获取 recent causes 边及其关联的 action_type 和 outcome_status
        cursor.execute(
            """
            SELECT e.cause_memory_id, e.effect_memory_id,
                   m1.action_type, m1.embodied_meta,
                   m2.action_type, m2.embodied_meta
            FROM embodied_causal_edges e
            JOIN embodied_memories m1 ON m1.memory_id = e.cause_memory_id
            JOIN embodied_memories m2 ON m2.memory_id = e.effect_memory_id
            WHERE e.created_at > datetime(?, 'unixepoch')
               OR m1.temporal_start > ?
            """,
            (since_sec, since_sec),
        )
        rows = cursor.fetchall()

        # 2. 聚合模式
        pattern_counts: Dict[Tuple[str, str], List[int]] = {}
        for cause_mid, effect_mid, cause_action, cause_meta_raw, effect_action, effect_meta_raw in rows:
            outcome_status = self._extract_outcome_status(effect_meta_raw)
            key = (cause_action or "unknown", outcome_status or "unknown")
            if key not in pattern_counts:
                pattern_counts[key] = []
            pattern_counts[key].append(cause_mid)

        # 3. 筛选高频模式并写入 concept_index
        patterns: List[ExtractedPattern] = []
        for (action, outcome), mids in pattern_counts.items():
            if len(mids) < min_support:
                continue

            rep_mid = mids[0]
            pattern_id = f"pattern_{action}_{outcome}"
            description = f"{action} -> {outcome} (support={len(mids)})"

            # 判断维度：如果是物理相关 action/outcome，归为 physics；否则 task
            dimension = "physics" if self._is_physics_pattern(action, outcome) else "task"

            self._index_pattern(rep_mid, pattern_id, dimension, description, len(mids))
            patterns.append(
                ExtractedPattern(
                    pattern_id=pattern_id,
                    dimension=dimension,
                    layer=2,
                    representative_memory_id=rep_mid,
                    support=len(mids),
                    description=description,
                )
            )

        # 4. 同时从 experience_graph 中提取 frequent edge_type 模式
        cursor.execute(
            """
            SELECT source_memory_id, target_memory_id, edge_type
            FROM embodied_experience_graph
            WHERE created_at > datetime(?, 'unixepoch')
            """,
            (since_sec,),
        )
        edge_rows = cursor.fetchall()
        edge_type_counts: Dict[str, List[Tuple[int, int]]] = {}
        for s, t, et in edge_rows:
            edge_type_counts.setdefault(et, []).append((s, t))

        for et, pairs in edge_type_counts.items():
            if len(pairs) < min_support:
                continue
            rep_s, rep_t = pairs[0]
            pattern_id = f"edge_pattern_{et}"
            description = f"Frequent edge type: {et} (count={len(pairs)})"
            self._index_pattern(rep_s, pattern_id, "task", description, len(pairs))
            patterns.append(
                ExtractedPattern(
                    pattern_id=pattern_id,
                    dimension="task",
                    layer=2,
                    representative_memory_id=rep_s,
                    support=len(pairs),
                    description=description,
                )
            )

        logger.info("PatternExtractor: %d patterns extracted", len(patterns))
        return patterns

    def _extract_outcome_status(self, meta_json: Optional[str]) -> Optional[str]:
        if not meta_json:
            return None
        try:
            meta = json.loads(meta_json)
            return meta.get("outcome_status") or meta.get("world_object", {}).get("state")
        except Exception:
            return None

    def _is_physics_pattern(self, action: str, outcome: str) -> bool:
        physics_keywords = ["collision", "slip", "friction", "force", "grasp", "torque"]
        text = f"{action} {outcome}".lower()
        return any(kw in text for kw in physics_keywords)

    def _index_pattern(self, memory_id: int, pattern_id: str, dimension: str, description: str, support: int) -> None:
        cursor = self.db_conn.cursor()
        confidence = min(support / 10.0, 1.0)
        try:
            cursor.execute(
                """
                INSERT INTO embodied_concept_index (memory_id, dimension, layer, concept_id, confidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, dimension, concept_id) DO UPDATE SET
                    layer = excluded.layer,
                    confidence = excluded.confidence
                """,
                (memory_id, dimension, 2, pattern_id, confidence),
            )
            self.db_conn.commit()
        except Exception as e:
            logger.warning("Failed to index pattern %s: %s", pattern_id, e)


# ---------------------------------------------------------------------------
# MeditationPipeline 主控
# ---------------------------------------------------------------------------

class MeditationPipeline:
    """冥想管道主控

    顺序执行三阶段离线抽象，生成报告。
    """

    def __init__(self, embodied_memory: Any):
        self.em = embodied_memory
        self.db_conn = embodied_memory.db_conn

    def run(self, phases: Optional[List[str]] = None) -> MeditationReport:
        """运行冥想管道

        Args:
            phases: 要执行的阶段列表，默认 ["consolidate", "crystallize", "extract"]

        Returns:
            MeditationReport
        """
        if phases is None:
            phases = ["consolidate", "crystallize", "extract"]

        report = MeditationReport()
        t0 = time.perf_counter()

        for phase in phases:
            try:
                if phase == "consolidate":
                    consolidator = EntityConsolidator(self.db_conn)
                    events = consolidator.consolidate(window_sec=30.0)
                    report.consolidated_count = len(events)

                elif phase == "crystallize":
                    crystallizer = RelationCrystallizer(self.em)
                    relations = crystallizer.crystallize()
                    report.crystallized_count = len(relations)

                elif phase == "extract":
                    extractor = PatternExtractor(self.db_conn)
                    patterns = extractor.extract_patterns(lookback_hours=24, min_support=3)
                    report.extracted_patterns = len(patterns)

                else:
                    report.errors.append(f"Unknown phase: {phase}")

            except Exception as e:
                logger.exception("Meditation phase %s failed", phase)
                report.errors.append(f"{phase}: {str(e)}")
                report.success = False

        report.elapsed_sec = time.perf_counter() - t0
        logger.info(
            "MeditationPipeline finished in %.2fs: consolidated=%d, crystallized=%d, patterns=%d",
            report.elapsed_sec,
            report.consolidated_count,
            report.crystallized_count,
            report.extracted_patterns,
        )
        return report
