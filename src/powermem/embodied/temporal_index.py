"""
时间索引 — Allen Interval Algebra 查询引擎

基于 SeekDB B-Tree 索引实现 13 种 Allen 关系的查询。
所有时间戳以秒为单位（浮点数），支持 wall_clock / sim_time / episode 等 frame。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .types import IntervalRelation, TemporalInterval

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Allen 关系查询 SQL 生成器
# ---------------------------------------------------------------------------

_ALLEN_SQL: Dict[IntervalRelation, str] = {
    IntervalRelation.BEFORE:  "{table}.temporal_end < {q_start}",
    IntervalRelation.AFTER:   "{table}.temporal_start > {q_end}",
    IntervalRelation.MEETS:   "ABS({table}.temporal_end - {q_start}) < {eps}",
    IntervalRelation.MET_BY:  "ABS({table}.temporal_start - {q_end}) < {eps}",
    IntervalRelation.OVERLAPS: "({table}.temporal_start < {q_start}) AND ({table}.temporal_end > {q_start}) AND ({table}.temporal_end < {q_end})",
    IntervalRelation.OVERLAPPED_BY: "({table}.temporal_start > {q_start}) AND ({table}.temporal_start < {q_end}) AND ({table}.temporal_end > {q_end})",
    IntervalRelation.DURING:  "({table}.temporal_start > {q_start}) AND ({table}.temporal_end < {q_end})",
    IntervalRelation.CONTAINS: "({table}.temporal_start < {q_start}) AND ({table}.temporal_end > {q_end})",
    IntervalRelation.STARTS:  "(ABS({table}.temporal_start - {q_start}) < {eps}) AND ({table}.temporal_end < {q_end})",
    IntervalRelation.STARTED_BY: "(ABS({table}.temporal_start - {q_start}) < {eps}) AND ({table}.temporal_end > {q_end})",
    IntervalRelation.FINISHES: "(ABS({table}.temporal_end - {q_end}) < {eps}) AND ({table}.temporal_start > {q_start})",
    IntervalRelation.FINISHED_BY: "(ABS({table}.temporal_end - {q_end}) < {eps}) AND ({table}.temporal_start < {q_start})",
    IntervalRelation.EQUALS:  "(ABS({table}.temporal_start - {q_start}) < {eps}) AND (ABS({table}.temporal_end - {q_end}) < {eps})",
}


def _format_condition(relation: IntervalRelation, table: str, query: TemporalInterval, eps: float = 1e-6) -> str:
    template = _ALLEN_SQL[relation]
    return template.format(table=table, q_start=query.start_sec, q_end=query.end_sec, eps=eps)


# ---------------------------------------------------------------------------
# 时间索引管理器
# ---------------------------------------------------------------------------

class TemporalIndex:
    """时间索引 — 基于 SeekDB 的区间查询"""

    def __init__(
        self,
        db_conn: Any,
        table_name: str = "embodied_memories",
        eps_sec: float = 1e-6,
    ):
        self.db_conn = db_conn
        self.table_name = table_name
        self.eps_sec = eps_sec

    def query(
        self,
        interval: TemporalInterval,
        relation: IntervalRelation,
        frame_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Tuple[int, TemporalInterval]]:
        """执行 Allen 关系查询

        Returns:
            [(memory_id, TemporalInterval), ...]
        """
        cursor = self.db_conn.cursor()
        condition = _format_condition(relation, self.table_name, interval, self.eps_sec)

        frame_filter = ""
        if frame_id is not None:
            frame_filter = f" AND {self.table_name}.temporal_frame_id = ?"

        sql = f"""
            SELECT memory_id, temporal_start, temporal_end, temporal_frame_id
            FROM {self.table_name}
            WHERE {condition}{frame_filter}
            AND temporal_start IS NOT NULL
            ORDER BY temporal_start
            LIMIT ?
        """
        params: Tuple[Any, ...] = (limit,)
        if frame_id is not None:
            params = (frame_id, limit)

        try:
            cursor.execute(sql, params)
        except Exception as e:
            logger.warning("Temporal query failed: %s | SQL: %s", e, sql[:200])
            return []

        results: List[Tuple[int, TemporalInterval]] = []
        for row in cursor.fetchall():
            mid, t_start, t_end, t_frame = row
            results.append((int(mid), TemporalInterval(float(t_start), float(t_end), str(t_frame))))
        return results

    def query_overlapping(
        self,
        interval: TemporalInterval,
        frame_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Tuple[int, TemporalInterval]]:
        """查询所有与给定区间重叠的记忆（最常用查询）"""
        cursor = self.db_conn.cursor()
        frame_filter = ""
        params: Tuple[Any, ...] = (interval.end_sec, interval.start_sec, limit)

        if frame_id is not None:
            frame_filter = " AND temporal_frame_id = ?"
            params = (interval.end_sec, interval.start_sec, frame_id, limit)

        # 重叠条件：A.start <= B.end AND A.end >= B.start（支持点区间）
        sql = f"""
            SELECT memory_id, temporal_start, temporal_end, temporal_frame_id
            FROM {self.table_name}
            WHERE temporal_start <= ?
              AND temporal_end >= ?
              AND temporal_start IS NOT NULL
              {frame_filter}
            ORDER BY temporal_start
            LIMIT ?
        """
        cursor.execute(sql, params)

        results: List[Tuple[int, TemporalInterval]] = []
        for row in cursor.fetchall():
            mid, t_start, t_end, t_frame = row
            results.append((int(mid), TemporalInterval(float(t_start), float(t_end), str(t_frame))))
        return results

    def query_before(
        self,
        timestamp_sec: float,
        frame_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Tuple[int, TemporalInterval]]:
        """查询在 timestamp 之前结束的所有记忆"""
        cursor = self.db_conn.cursor()
        frame_filter = ""
        params: Tuple[Any, ...] = (timestamp_sec, limit)
        if frame_id is not None:
            frame_filter = " AND temporal_frame_id = ?"
            params = (timestamp_sec, frame_id, limit)

        sql = f"""
            SELECT memory_id, temporal_start, temporal_end, temporal_frame_id
            FROM {self.table_name}
            WHERE temporal_end < ?
              AND temporal_start IS NOT NULL
              {frame_filter}
            ORDER BY temporal_end DESC
            LIMIT ?
        """
        cursor.execute(sql, params)
        results: List[Tuple[int, TemporalInterval]] = []
        for row in cursor.fetchall():
            mid, t_start, t_end, t_frame = row
            results.append((int(mid), TemporalInterval(float(t_start), float(t_end), str(t_frame))))
        return results

    def query_after(
        self,
        timestamp_sec: float,
        frame_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Tuple[int, TemporalInterval]]:
        """查询在 timestamp 之后开始的所有记忆"""
        cursor = self.db_conn.cursor()
        frame_filter = ""
        params: Tuple[Any, ...] = (timestamp_sec, limit)
        if frame_id is not None:
            frame_filter = " AND temporal_frame_id = ?"
            params = (timestamp_sec, frame_id, limit)

        sql = f"""
            SELECT memory_id, temporal_start, temporal_end, temporal_frame_id
            FROM {self.table_name}
            WHERE temporal_start > ?
              AND temporal_start IS NOT NULL
              {frame_filter}
            ORDER BY temporal_start
            LIMIT ?
        """
        cursor.execute(sql, params)
        results: List[Tuple[int, TemporalInterval]] = []
        for row in cursor.fetchall():
            mid, t_start, t_end, t_frame = row
            results.append((int(mid), TemporalInterval(float(t_start), float(t_end), str(t_frame))))
        return results

    def query_contains_point(
        self,
        timestamp_sec: float,
        frame_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Tuple[int, TemporalInterval]]:
        """查询包含给定时间点的记忆"""
        cursor = self.db_conn.cursor()
        frame_filter = ""
        params: Tuple[Any, ...] = (timestamp_sec, timestamp_sec, limit)
        if frame_id is not None:
            frame_filter = " AND temporal_frame_id = ?"
            params = (timestamp_sec, timestamp_sec, frame_id, limit)

        sql = f"""
            SELECT memory_id, temporal_start, temporal_end, temporal_frame_id
            FROM {self.table_name}
            WHERE temporal_start <= ?
              AND temporal_end >= ?
              AND temporal_start IS NOT NULL
              {frame_filter}
            LIMIT ?
        """
        cursor.execute(sql, params)
        results: List[Tuple[int, TemporalInterval]] = []
        for row in cursor.fetchall():
            mid, t_start, t_end, t_frame = row
            results.append((int(mid), TemporalInterval(float(t_start), float(t_end), str(t_frame))))
        return results
