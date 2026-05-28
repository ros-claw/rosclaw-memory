"""
CognitiveRouter — Tri-Route 认知检索引擎

专家二（专家二）架构哲学的实现：
- System-1 Associative Route: 向量相似度检索（直觉）
- System-2 Global Selection Route: 层级图遍历（推理）
- System-3 SpatioTemporal Route: 时空查询（定位）

三路正交，交集生成，带时空衰减的综合重排。
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ._json import fast_dumps
from .memory_atom import MemoryAtom
from .types import IntervalRelation, TemporalInterval, Vec3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 查询意图解析
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class QueryIntent:
    """查询意图 —— 决定激活哪些路由"""

    query: str = ""
    needs_associative: bool = True      # System-1: 默认总是需要语义召回
    needs_global: bool = False          # System-2: 全局模式/抽象推理
    needs_spatiotemporal: bool = False  # System-3: 时空约束

    # 解析出的实体提示（供各路由使用）
    suggested_concepts: List[str] = field(default_factory=list)
    suggested_dimensions: List[str] = field(default_factory=list)


# 触发 System-2 的抽象/模式关键词（中英双语）
_GLOBAL_KEYWORDS = frozenset([
    # 英语
    "why", "pattern", "usually", "often", "most", "trend", "always",
    "generally", "typically", "frequently", "summary", "overview",
    "common", "habit", "routine", "fails", "failures", "degradation",
    # 中文
    "为什么", "模式", "规律", "习惯", "通常", "总是", "经常",
    "一般", "总结", "概述", "常见", "失败", "退化", "趋势",
    "越来越差", "越来越", "哪里容易", "哪个区域", "哪些",
])

# 触发 System-2 的疑问词（需要全局推理的问题类型）
_GLOBAL_QUESTION_PATTERNS = [
    re.compile(r"what.*(?:pattern|trend|change|difference)"),
    re.compile(r"which.*(?:most|least|best|worst|often)"),
    re.compile(r"why.*(?:always|usually|often|frequently)"),
    re.compile(r"(?:how many|how much).*(?:overall|in total|on average)"),
]


# ---------------------------------------------------------------------------
# 候选记忆包装（带多路得分）
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Candidate:
    memory_id: int
    atom: Optional[MemoryAtom] = None
    semantic_score: float = 0.0
    structural_score: float = 0.0
    spatial_distance: Optional[float] = None
    temporal_delta: Optional[float] = None


# ---------------------------------------------------------------------------
# CognitiveRouter
# ---------------------------------------------------------------------------

class CognitiveRouter:
    """Tri-Route 认知路由器

    Args:
        embodied_memory: EmbodiedMemory 实例（用于访问索引和存储）
        alpha: System-1 语义得分权重
        beta: System-2 结构得分权重
        gamma: 因果/经验加成权重
        sigma_spatial: 空间衰减特征长度（米，默认 3.0）
        sigma_temporal: 时间衰减特征长度（秒，默认 3600）
    """

    def __init__(
        self,
        embodied_memory: Any,
        alpha: float = 1.0,
        beta: float = 0.8,
        gamma: float = 0.3,
        sigma_spatial: float = 3.0,
        sigma_temporal: float = 3600.0,
        assoc_cache_ttl: float = 5.0,
    ):
        self.em = embodied_memory
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma_spatial = sigma_spatial
        self.sigma_temporal = sigma_temporal
        self._assoc_cache_ttl = assoc_cache_ttl
        self._associative_cache: Dict[str, Tuple[Dict[int, _Candidate], float]] = {}
        self._MAX_ASSOC_CACHE = 64

    # ========================================================================
    # 公共入口
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
        """Tri-Route 认知检索入口

        流程：
        1. 解析查询意图（决定激活哪些路由）
        2. 并行执行各路由（当前顺序执行，未来可改 async）
        3. 取候选集交集
        4. 综合重排（带时空衰减）
        """
        intent = self._parse_intent(query, spatial_center, temporal_interval)
        logger.debug(
            "CognitiveRouter intent: assoc=%s global=%s st=%s",
            intent.needs_associative,
            intent.needs_global,
            intent.needs_spatiotemporal,
        )

        # 收集各路由的候选集
        st_candidates: Dict[int, _Candidate] = {}
        assoc_candidates: Dict[int, _Candidate] = {}
        global_candidates: Dict[int, _Candidate] = {}

        # System-3: SpatioTemporal Route
        if intent.needs_spatiotemporal:
            st_candidates = self._route_spatiotemporal(
                spatial_center, spatial_radius, temporal_interval, temporal_relation, limit
            )

        # System-1: Associative Route
        if intent.needs_associative:
            assoc_candidates = self._route_associative(query, filters, limit)

        # System-2: Global Selection Route
        if intent.needs_global:
            global_candidates = self._route_global_selection(intent, limit)

        # 如果没有激活任何路由，回退到纯语义检索
        if not st_candidates and not assoc_candidates and not global_candidates:
            assoc_candidates = self._route_associative(query, filters, limit)

        # 交集：候选必须出现在所有激活的路由中
        active_sets: List[Set[int]] = []
        if intent.needs_spatiotemporal and st_candidates:
            active_sets.append(set(st_candidates.keys()))
        if intent.needs_associative and assoc_candidates:
            active_sets.append(set(assoc_candidates.keys()))
        if intent.needs_global and global_candidates:
            active_sets.append(set(global_candidates.keys()))

        if active_sets:
            intersection_ids = set.intersection(*active_sets)
        else:
            intersection_ids = set()

        # 合并交集中的候选（保留所有路由的得分信息）
        merged: Dict[int, _Candidate] = {}
        for mid in intersection_ids:
            cand = _Candidate(memory_id=mid)
            if mid in st_candidates:
                cand.spatial_distance = st_candidates[mid].spatial_distance
            if mid in assoc_candidates:
                cand.semantic_score = assoc_candidates[mid].semantic_score
            if mid in global_candidates:
                cand.structural_score = global_candidates[mid].structural_score
            merged[mid] = cand

        # 如果交集为空但有多路激活，退化为并集（避免无结果）
        if not merged and len(active_sets) > 1:
            for cand_dict in (st_candidates, assoc_candidates, global_candidates):
                for mid, cand in cand_dict.items():
                    if mid not in merged:
                        merged[mid] = cand
                    else:
                        existing = merged[mid]
                        existing.semantic_score = max(existing.semantic_score, cand.semantic_score)
                        existing.structural_score = max(existing.structural_score, cand.structural_score)
                        if cand.spatial_distance is not None:
                            existing.spatial_distance = cand.spatial_distance

        # 综合重排
        ranked = self._rerank(merged, spatial_center, temporal_interval)

        # 加载 atom 并返回
        results: List[MemoryAtom] = []
        for cand in ranked[:limit]:
            if cand.atom is None:
                cand.atom = self.em.get_atom(cand.memory_id)
            if cand.atom is not None:
                results.append(cand.atom)
        return results

    # ========================================================================
    # System-1: Associative Route（联想/相似度路由）
    # ========================================================================

    def _assoc_cache_key(
        self, query: str, filters: Optional[Dict[str, Any]], limit: int
    ) -> str:
        """生成 associative cache key。"""
        filters_key = fast_dumps(filters, sort_keys=True) if filters else ""
        return f"{query}::{filters_key}::{limit}"

    def _get_cached_associative(
        self, query: str, filters: Optional[Dict[str, Any]], limit: int
    ) -> Optional[Dict[int, _Candidate]]:
        key = self._assoc_cache_key(query, filters, limit)
        entry = self._associative_cache.get(key)
        if entry is None:
            return None
        candidates, ts = entry
        if time.monotonic() - ts > self._assoc_cache_ttl:
            self._associative_cache.pop(key, None)
            return None
        return candidates

    def _set_cached_associative(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        limit: int,
        candidates: Dict[int, _Candidate],
    ) -> None:
        key = self._assoc_cache_key(query, filters, limit)
        self._associative_cache[key] = (candidates, time.monotonic())
        if len(self._associative_cache) > self._MAX_ASSOC_CACHE:
            # evict oldest 25%
            sorted_items = sorted(
                self._associative_cache.items(), key=lambda x: x[1][1]
            )
            for k, _ in sorted_items[: self._MAX_ASSOC_CACHE // 4]:
                del self._associative_cache[k]

    def clear_cache(self) -> None:
        """清空 associative cache（在图结构变更时调用）。"""
        self._associative_cache.clear()

    def _route_associative(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        limit: int,
    ) -> Dict[int, _Candidate]:
        """System-1: 基于 PowerMem 语义相似度检索（带 TTL cache）。"""
        cached = self._get_cached_associative(query, filters, limit)
        if cached is not None:
            return cached

        candidates: Dict[int, _Candidate] = {}
        try:
            semantic_result = self.em.memory.search(query, filters=filters, limit=limit * 3)
            for item in semantic_result.get("results", []):
                mid = item.get("id")
                if mid is not None:
                    candidates[int(mid)] = _Candidate(
                        memory_id=int(mid),
                        semantic_score=item.get("score", 0.0),
                    )
        except Exception as e:
            logger.warning("Associative route failed: %s", e)
        self._set_cached_associative(query, filters, limit, candidates)
        return candidates

    # ========================================================================
    # System-3: SpatioTemporal Route（时空路由）
    # ========================================================================

    def _route_spatiotemporal(
        self,
        spatial_center: Optional[Vec3],
        spatial_radius: Optional[float],
        temporal_interval: Optional[TemporalInterval],
        temporal_relation: Optional[IntervalRelation],
        limit: int,
    ) -> Dict[int, _Candidate]:
        """System-3: 基于空间/时间索引的物理定位检索"""
        candidates: Dict[int, _Candidate] = {}

        # 空间过滤
        spatial_ids: Optional[Set[int]] = None
        if spatial_center is not None and spatial_radius is not None:
            try:
                hits = self.em.spatial_index.query_radius(
                    spatial_center, spatial_radius, limit=limit * 3
                )
                spatial_ids = set()
                for mid, dist in hits:
                    spatial_ids.add(mid)
                    candidates[mid] = _Candidate(
                        memory_id=mid,
                        spatial_distance=dist,
                    )
            except Exception as e:
                logger.warning("Spatial route failed: %s", e)

        # 时间过滤
        temporal_ids: Optional[Set[int]] = None
        if temporal_interval is not None:
            try:
                if temporal_relation is None:
                    hits = self.em.temporal_index.query_overlapping(
                        temporal_interval, limit=limit * 3
                    )
                else:
                    hits = self.em.temporal_index.query(
                        temporal_interval, temporal_relation, limit=limit * 3
                    )
                temporal_ids = {mid for mid, _ in hits}
            except Exception as e:
                logger.warning("Temporal route failed: %s", e)

        # 计算交集：时空同时提供时必须同时满足
        final_ids: Set[int] = set()
        if spatial_ids is not None and temporal_ids is not None:
            final_ids = spatial_ids & temporal_ids
        elif spatial_ids is not None:
            final_ids = spatial_ids
        elif temporal_ids is not None:
            final_ids = temporal_ids

        # 重建候选集（仅保留交集中的 ID，保留空间距离信息）
        result: Dict[int, _Candidate] = {}
        for mid in final_ids:
            result[mid] = _Candidate(
                memory_id=mid,
                spatial_distance=candidates.get(mid, _Candidate(memory_id=mid)).spatial_distance,
            )
        return result

    # ========================================================================
    # System-2: Global Selection Route（全局模式路由）
    # ========================================================================

    def _route_global_selection(
        self,
        intent: QueryIntent,
        limit: int,
    ) -> Dict[int, _Candidate]:
        """System-2: 基于概念索引和经验图的全局模式检索（轻量骨架）

        当前实现：
        1. 在 embodied_concept_index 中搜索与查询相关的 concept
        2. 获取这些 concept 关联的 memory_ids
        3. 通过 embodied_experience_graph 做一阶扩展
        4. 给扩展节点结构分加成
        """
        candidates: Dict[int, _Candidate] = {}
        if not intent.suggested_concepts:
            # 即使没有解析出具体概念，也尝试用查询关键词匹配 concept_id
            intent.suggested_concepts = self._extract_concepts_from_query(intent.query)

        try:
            cursor = self.em.db_conn.cursor()

            # 1. 概念匹配：用 concept_id LIKE 匹配查询关键词
            matched_concepts: Set[str] = set()
            for concept in intent.suggested_concepts:
                cursor.execute(
                    "SELECT DISTINCT concept_id FROM embodied_concept_index WHERE concept_id LIKE ?",
                    (f"%{concept}%",),
                )
                for (cid,) in cursor.fetchall():
                    matched_concepts.add(cid)

            if not matched_concepts:
                # 尝试更宽松的匹配：dimension + layer 组合
                for dim in intent.suggested_dimensions:
                    cursor.execute(
                        "SELECT DISTINCT concept_id FROM embodied_concept_index WHERE dimension = ?",
                        (dim,),
                    )
                    for (cid,) in cursor.fetchall():
                        matched_concepts.add(cid)

            if not matched_concepts:
                return candidates

            # 2. 获取与这些 concept 关联的 memory_ids
            concept_memories: Dict[int, float] = {}  # memory_id -> max confidence
            placeholders = ",".join("?" * len(matched_concepts))
            cursor.execute(
                f"SELECT memory_id, confidence FROM embodied_concept_index WHERE concept_id IN ({placeholders})",
                tuple(matched_concepts),
            )
            for mid, conf in cursor.fetchall():
                mid = int(mid)
                conf = float(conf or 1.0)
                concept_memories[mid] = max(concept_memories.get(mid, 0.0), conf)

            # 3. 经验图一阶扩展
            expanded: Dict[int, Tuple[float, int]] = {}  # memory_id -> (score, hop_distance)
            for mid, conf in concept_memories.items():
                expanded[mid] = (conf, 0)
                # 一阶邻居（沿 experience_graph 出边和入边）
                cursor.execute(
                    "SELECT target_memory_id, strength FROM embodied_experience_graph WHERE source_memory_id = ?",
                    (mid,),
                )
                for nid, strength in cursor.fetchall():
                    nid = int(nid)
                    s = float(strength or 1.0) * conf * 0.5  # 一阶衰减
                    if nid not in expanded or expanded[nid][0] < s:
                        expanded[nid] = (s, 1)

                cursor.execute(
                    "SELECT source_memory_id, strength FROM embodied_experience_graph WHERE target_memory_id = ?",
                    (mid,),
                )
                for nid, strength in cursor.fetchall():
                    nid = int(nid)
                    s = float(strength or 1.0) * conf * 0.5
                    if nid not in expanded or expanded[nid][0] < s:
                        expanded[nid] = (s, 1)

            # 4. 构建候选
            for mid, (score, hops) in expanded.items():
                candidates[mid] = _Candidate(
                    memory_id=mid,
                    structural_score=score,
                )

        except Exception as e:
            logger.warning("Global selection route failed: %s", e)

        return candidates

    # ========================================================================
    # 意图解析
    # ========================================================================

    def _parse_intent(
        self,
        query: str,
        spatial_center: Optional[Vec3],
        temporal_interval: Optional[TemporalInterval],
    ) -> QueryIntent:
        """解析查询意图，决定激活哪些路由"""
        intent = QueryIntent(query=query)
        q_lower = query.lower()

        # System-3: 如果显式提供了时空参数，强制激活
        if spatial_center is not None or temporal_interval is not None:
            intent.needs_spatiotemporal = True

        # System-2: 检测抽象/模式关键词
        for kw in _GLOBAL_KEYWORDS:
            if kw in q_lower:
                intent.needs_global = True
                break

        # System-2: 检测疑问模式
        if not intent.needs_global:
            for pattern in _GLOBAL_QUESTION_PATTERNS:
                if pattern.search(q_lower):
                    intent.needs_global = True
                    break

        # 提取建议的概念和维度
        intent.suggested_concepts = self._extract_concepts_from_query(query)
        intent.suggested_dimensions = self._extract_dimensions_from_query(query)

        # 如果 System-2 未激活且没有时空参数，退化为纯 System-1
        if not intent.needs_global and not intent.needs_spatiotemporal:
            intent.needs_associative = True

        return intent

    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """从查询中提取可能的概念关键词（简单实现）"""
        # 提取引号内内容、驼峰词、下划线词
        concepts: List[str] = []
        # 引号内容
        for match in re.finditer(r'["\']([^"\']+)["\']', query):
            concepts.append(match.group(1).lower())
        # 下划线词（如 red_mug, kitchen_table）
        for match in re.finditer(r'\b[a-zA-Z_]+_[a-zA-Z0-9_]+\b', query):
            concepts.append(match.group(0).lower())
        # 驼峰词（如 WorldObject）
        for match in re.finditer(r'\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b', query):
            concepts.append(match.group(0).lower())
        return concepts

    def _extract_dimensions_from_query(self, query: str) -> List[str]:
        """从查询中提取可能的维度提示"""
        q_lower = query.lower()
        dims: List[str] = []
        dim_keywords = {
            "task": ["task", "mission", "job", "工作", "任务"],
            "physics": ["physics", "force", "friction", "collision", "物理", "力", "碰撞"],
            "spatial_region": ["region", "area", "zone", "room", "区域", "房间", "区域"],
            "social": ["human", "person", "people", "user", "人", "用户"],
        }
        for dim, kws in dim_keywords.items():
            for kw in kws:
                if kw in q_lower:
                    dims.append(dim)
                    break
        return dims

    # ========================================================================
    # 综合重排
    # ========================================================================

    def _rerank(
        self,
        candidates: Dict[int, _Candidate],
        spatial_center: Optional[Vec3],
        temporal_interval: Optional[TemporalInterval],
    ) -> List[_Candidate]:
        """综合重排：融合语义分、结构分、时空衰减

        Score = alpha * semantic + beta * structural * decay_spatial * decay_temporal + gamma * causal_boost
        """
        t_center: Optional[float] = None
        if temporal_interval is not None:
            t_center = (temporal_interval.start_sec + temporal_interval.end_sec) / 2.0

        scored: List[Tuple[_Candidate, float]] = []
        for cand in candidates.values():
            # 空间衰减
            spatial_decay = 1.0
            if spatial_center is not None and cand.spatial_distance is not None:
                spatial_decay = math.exp(-cand.spatial_distance / self.sigma_spatial)

            # 时间衰减
            temporal_decay = 1.0
            if t_center is not None and cand.temporal_delta is not None:
                temporal_decay = math.exp(-abs(cand.temporal_delta - t_center) / self.sigma_temporal)
            elif t_center is not None and cand.atom is not None and cand.atom.temporal is not None:
                t_atom = (cand.atom.temporal.start_sec + cand.atom.temporal.end_sec) / 2.0
                temporal_decay = math.exp(-abs(t_atom - t_center) / self.sigma_temporal)

            # 因果/经验加成（如果 atom 在因果链中）
            causal_boost = 0.0
            if cand.atom is not None:
                if cand.atom.causal_parents:
                    causal_boost = min(len(cand.atom.causal_parents) * 0.1, 0.5)

            final_score = (
                self.alpha * cand.semantic_score
                + self.beta * cand.structural_score * spatial_decay * temporal_decay
                + self.gamma * causal_boost
            )
            scored.append((cand, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [cand for cand, _ in scored]

    # ========================================================================
    # 工具方法：为外部模块提供概念索引写入接口
    # ========================================================================

    def index_concept(
        self,
        memory_id: int,
        dimension: str,
        layer: int,
        concept_id: str,
        confidence: float = 1.0,
    ) -> None:
        """为指定记忆添加概念索引条目"""
        cursor = self.em.db_conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO embodied_concept_index (memory_id, dimension, layer, concept_id, confidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, dimension, concept_id) DO UPDATE SET
                    layer = excluded.layer,
                    confidence = excluded.confidence
                """,
                (memory_id, dimension, layer, concept_id, confidence),
            )
            self.em.db_conn.commit()
            self.clear_cache()
        except Exception as e:
            logger.warning("Failed to index concept %s for memory %s: %s", concept_id, memory_id, e)

    def add_experience_edge(
        self,
        source_memory_id: int,
        target_memory_id: int,
        edge_type: str,
        strength: float = 1.0,
        spatial_context: Optional[Dict[str, Any]] = None,
        temporal_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """在经验图中添加一条物理关系边"""
        cursor = self.em.db_conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO embodied_experience_graph
                (source_memory_id, target_memory_id, edge_type, strength, spatial_context_json, temporal_context_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    source_memory_id,
                    target_memory_id,
                    edge_type,
                    strength,
                    fast_dumps(spatial_context) if spatial_context else None,
                    fast_dumps(temporal_context) if temporal_context else None,
                ),
            )
            self.em.db_conn.commit()
            self.clear_cache()
        except Exception as e:
            logger.warning(
                "Failed to add experience edge %s -> %s: %s", source_memory_id, target_memory_id, e
            )
