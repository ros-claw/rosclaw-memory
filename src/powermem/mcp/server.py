"""
RosClaw-Memory MCP Server — Claude 具身记忆查询接口

通过 MCP (Model Context Protocol) 协议，让 Claude 直接读取机器人的
空间记忆、轨迹、场景图、世界对象和因果关系。

Usage:
    python -m powermem.mcp.server --db-path ./embodied.db

然后在 Claude Desktop / Claude Code 的 mcp 配置中添加:
    {
        "mcpServers": {
            "rosclaw-memory": {
                "command": "python",
                "args": ["-m", "powermem.mcp.server", "--db-path", "./embodied.db"]
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import TemporalInterval, Vec3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EmbodiedMemory 生命周期管理（每个工具调用复用同一个实例）
# ---------------------------------------------------------------------------

_db_conn: Optional[sqlite3.Connection] = None
_embodied_memory: Optional[EmbodiedMemory] = None


class _MockStorageAdapter:
    """PowerMem 存储的最小 mock —— 只提供 CRUD，无 LLM/向量功能"""

    def __init__(self):
        self._store: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def add_memory(self, payload: Dict[str, Any]) -> int:
        mid = self._next_id
        self._next_id += 1
        self._store[mid] = {
            "id": mid,
            "content": payload.get("content", ""),
            "metadata": payload.get("metadata", {}),
            "user_id": payload.get("user_id", ""),
            "agent_id": payload.get("agent_id", ""),
        }
        return mid

    def get_memory(self, memory_id: int) -> Any:
        return self._store.get(memory_id)

    def delete_memory(self, memory_id: int, **kwargs: Any) -> bool:
        return self._store.pop(memory_id, None) is not None

    def search_memories(self, **kwargs: Any) -> list:
        limit = kwargs.get("limit", 30)
        results = []
        for mid, item in list(self._store.items())[:limit]:
            results.append({
                "id": mid,
                "memory": item["content"],
                "score": 0.9,
                "metadata": item.get("metadata", {}),
            })
        return results

    def update_memory(self, memory_id: int, content: str, **kwargs: Any) -> Dict[str, Any]:
        item = self._store.get(memory_id)
        if item is None:
            raise KeyError(memory_id)
        item["content"] = content
        return item


class _MockMemory:
    def __init__(self):
        self.storage = _MockStorageAdapter()
        self.agent_id = "rosclaw_mcp_agent"

    def add(self, content, **kwargs):
        return self.storage.add_memory({"content": content, "metadata": kwargs.get("metadata", {})})

    def search(self, query, **kwargs):
        results = self.storage.search_memories(limit=kwargs.get("limit", 30))
        return {"results": results, "relations": []}

    def get(self, memory_id, **kwargs):
        return self.storage.get_memory(memory_id)

    def delete(self, memory_id):
        return self.storage.delete_memory(memory_id)

    def update(self, memory_id, content, **kwargs):
        return self.storage.update_memory(memory_id, content)


def _get_embodied_memory(db_path: str) -> EmbodiedMemory:
    """获取或创建 EmbodiedMemory 实例（单例）"""
    global _db_conn, _embodied_memory
    if _embodied_memory is None:
        _db_conn = sqlite3.connect(db_path, check_same_thread=False)
        initialize_embodied_schema(_db_conn)
        mock_mem = _MockMemory()
        _embodied_memory = EmbodiedMemory(memory=mock_mem, db_conn=_db_conn, enable_plugin=False)
    return _embodied_memory


# ---------------------------------------------------------------------------
# 序列化 Helpers
# ---------------------------------------------------------------------------

def _atom_to_dict(atom) -> Dict[str, Any]:
    """将 MemoryAtom 转为可 JSON 序列化的 dict"""
    return {
        "memory_id": atom.memory_id,
        "content": atom.content,
        "spatial": (
            {"x": atom.spatial.x, "y": atom.spatial.y, "z": atom.spatial.z}
            if atom.spatial else None
        ),
        "spatial_frame_id": atom.spatial_frame_id,
        "temporal": (
            {"start_sec": atom.temporal.start_sec, "end_sec": atom.temporal.end_sec}
            if atom.temporal else None
        ),
        "action": atom.action.value,
        "prediction_error": atom.prediction_error,
        "embodied_meta": atom.embodied_meta,
    }


def _world_object_to_dict(obj) -> Dict[str, Any]:
    """将 WorldObject 转为可 JSON 序列化的 dict"""
    return {
        "obj_id": obj.obj_id,
        "obj_type": obj.obj_type,
        "name": obj.name,
        "pose": {
            "position": {
                "x": obj.pose.position.x,
                "y": obj.pose.position.y,
                "z": obj.pose.position.z,
            },
            "orientation": {
                "w": obj.pose.orientation.w,
                "x": obj.pose.orientation.x,
                "y": obj.pose.orientation.y,
                "z": obj.pose.orientation.z,
            },
        },
        "size": obj.size,
        "color": obj.color,
        "scene_id": obj.scene_id,
        "parent_obj_id": obj.parent_obj_id,
        "state": obj.state,
        "semantic_tags": obj.semantic_tags,
        "memory_id": obj.memory_id,
    }


def _spatial_relation_to_dict(rel) -> Dict[str, Any]:
    return {
        "subject_id": rel.subject_id,
        "object_id": rel.object_id,
        "relation": rel.relation,
        "confidence": rel.confidence,
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

def create_mcp_server(db_path: str):
    """创建并配置 FastMCP 服务器实例"""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "MCP SDK not installed. Run: pip install mcp"
        ) from e

    mcp = FastMCP("rosclaw-memory")

    # -----------------------------------------------------------------------
    # Tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def get_stats() -> str:
        """获取记忆系统统计信息（对象数、原子数、索引状态）"""
        em = _get_embodied_memory(db_path)
        stats = {
            "spatial_index": em.spatial_index.stats(),
            "total_memories": len(em.memory.storage._store),
        }
        return json.dumps(stats, indent=2, ensure_ascii=False)

    @mcp.tool()
    def search_near(
        center_x: float,
        center_y: float,
        center_z: float,
        radius: float = 2.0,
        limit: int = 30,
    ) -> str:
        """空间范围搜索 — 查找指定球形区域内的记忆原子"""
        em = _get_embodied_memory(db_path)
        center = Vec3(center_x, center_y, center_z)
        atoms = em.search_near(center, radius=radius, limit=limit)
        return json.dumps(
            [_atom_to_dict(a) for a in atoms],
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def search_temporal(
        start_sec: float,
        end_sec: float,
        limit: int = 30,
    ) -> str:
        """时间区间搜索 — 查找与指定时间区间重叠的记忆原子"""
        em = _get_embodied_memory(db_path)
        interval = TemporalInterval(start_sec=start_sec, end_sec=end_sec)
        atoms = em.search_temporal(interval, limit=limit)
        return json.dumps(
            [_atom_to_dict(a) for a in atoms],
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def search_similar_trajectories(
        query_waypoints_json: str,
        center_x: float,
        center_y: float,
        center_z: float,
        spatial_radius: float = 3.0,
        top_k: int = 5,
    ) -> str:
        """轨迹相似度搜索 — 查找与查询轨迹形状最相似的历史轨迹

        Args:
            query_waypoints_json: JSON 数组，格式 [{"x":0,"y":0,"z":0,"t":0}, ...]
            center_x/y/z: 空间查询中心（通常取查询轨迹中点）
            spatial_radius: 空间搜索半径（米）
            top_k: 返回最大数量
        """
        em = _get_embodied_memory(db_path)
        raw_wps = json.loads(query_waypoints_json)
        query_waypoints = []
        for wp in raw_wps:
            pos = Vec3(float(wp["x"]), float(wp["y"]), float(wp["z"]))
            ts = float(wp.get("t", wp.get("timestamp_sec", 0.0)))
            query_waypoints.append((pos, ts))

        center = Vec3(center_x, center_y, center_z)
        results = em.search_similar_trajectories(
            query_waypoints,
            spatial_center=center,
            spatial_radius=spatial_radius,
            top_k=top_k,
        )
        output = []
        for atom, dtw_dist in results:
            d = _atom_to_dict(atom)
            d["_dtw_distance"] = dtw_dist
            output.append(d)
        return json.dumps(output, indent=2, ensure_ascii=False)

    @mcp.tool()
    def get_scene_graph(scene_id: str) -> str:
        """获取场景图 — 返回指定场景中的所有对象及其空间关系"""
        em = _get_embodied_memory(db_path)
        sg = em.get_scene_graph(scene_id)
        objects = sg.get_objects()
        relations = sg.compute_relations(spatial_tolerance=0.05)
        return json.dumps(
            {
                "scene_id": scene_id,
                "object_count": len(objects),
                "relation_count": len(relations),
                "objects": [_world_object_to_dict(o) for o in objects],
                "relations": [_spatial_relation_to_dict(r) for r in relations],
            },
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def search_world_objects(
        center_x: float,
        center_y: float,
        center_z: float,
        radius: float = 2.0,
        scene_id: Optional[str] = None,
        obj_type: Optional[str] = None,
        limit: int = 30,
    ) -> str:
        """搜索世界对象 — 按空间位置查找物理世界中的物体"""
        em = _get_embodied_memory(db_path)
        center = Vec3(center_x, center_y, center_z)
        objects = em.search_world_objects(
            center,
            radius=radius,
            scene_id=scene_id,
            obj_type=obj_type,
            limit=limit,
        )
        return json.dumps(
            [_world_object_to_dict(o) for o in objects],
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def get_world_object(obj_id: str) -> str:
        """获取指定世界对象的详细信息"""
        em = _get_embodied_memory(db_path)
        obj = em.get_world_object(obj_id)
        if obj is None:
            return json.dumps({"found": False, "obj_id": obj_id})
        return json.dumps(
            {"found": True, "object": _world_object_to_dict(obj)},
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def get_causal_chain(
        memory_id: int,
        direction: str = "effects",
        limit: int = 10,
    ) -> str:
        """因果链查询 — 获取指定记忆的原因或结果

        Args:
            memory_id: 记忆 ID
            direction: "causes" 或 "effects"
            limit: 最大返回数
        """
        em = _get_embodied_memory(db_path)
        if direction == "causes":
            atoms = em.get_causes(memory_id, limit=limit)
        else:
            atoms = em.get_effects(memory_id, limit=limit)
        return json.dumps(
            {
                "memory_id": memory_id,
                "direction": direction,
                "count": len(atoms),
                "atoms": [_atom_to_dict(a) for a in atoms],
            },
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def cognitive_search(
        query: str,
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        radius: float = 0.0,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        limit: int = 30,
    ) -> str:
        """Tri-Route 认知检索 — 融合语义相似度 + 全局模式 + 时空约束

        根据查询内容自动选择路由：
        - 普通查询走 System-1（语义相似度）
        - 含抽象关键词（如"为什么/规律/模式"）自动激活 System-2（全局模式遍历）
        - 提供时空参数时激活 System-3（物理定位）

        Args:
            query: 查询文本
            center_x/y/z: 空间中心（可选）
            radius: 空间半径（米，大于0时启用空间过滤）
            start_sec/end_sec: 时间区间（可选，end_sec > start_sec 时启用时间过滤）
            limit: 最大返回数
        """
        em = _get_embodied_memory(db_path)
        spatial = Vec3(center_x, center_y, center_z) if radius > 0 else None
        temporal = TemporalInterval(start_sec, end_sec) if end_sec > start_sec else None
        atoms = em.search(
            query=query,
            spatial_center=spatial,
            spatial_radius=radius if radius > 0 else None,
            temporal_interval=temporal,
            limit=limit,
        )
        return json.dumps(
            [_atom_to_dict(a) for a in atoms],
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def index_concept(
        memory_id: int,
        dimension: str,
        layer: int,
        concept_id: str,
        confidence: float = 1.0,
    ) -> str:
        """为指定记忆添加概念索引条目（支撑 System-2 Global Selection）

        Args:
            memory_id: 记忆 ID
            dimension: 概念维度，如 "task" / "physics" / "spatial_region" / "social" / "entity"
            layer: 层级深度（1=最具体，2=抽象概念）
            concept_id: 概念标识符
            confidence: 置信度（0.0-1.0，默认 1.0）
        """
        em = _get_embodied_memory(db_path)
        em.index_concept(memory_id, dimension, layer, concept_id, confidence)
        return json.dumps(
            {"success": True, "memory_id": memory_id, "concept_id": concept_id},
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def add_experience_edge(
        source_memory_id: int,
        target_memory_id: int,
        edge_type: str,
        strength: float = 1.0,
    ) -> str:
        """在物理经验图中添加一条关系边

        Args:
            source_memory_id: 源记忆 ID
            target_memory_id: 目标记忆 ID
            edge_type: 关系类型，可选：causes, precedes, supports, contains,
                       instantiates, part_of, adjacent_to, overlaps_temporally
            strength: 边强度（0.0-1.0，默认 1.0）
        """
        em = _get_embodied_memory(db_path)
        em.add_experience_edge(
            source_memory_id, target_memory_id, edge_type, strength
        )
        return json.dumps(
            {
                "success": True,
                "source": source_memory_id,
                "target": target_memory_id,
                "edge_type": edge_type,
            },
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def run_meditation(
        phases: str = "consolidate,crystallize,extract",
    ) -> str:
        """触发离线冥想管道，自动抽象历史记忆

        三阶段熵减：
        1. consolidate — 同一实体短时观测聚合为事件
        2. crystallize — 增量式空间关系固化
        3. extract — 高频动作-结果模式提取为抽象概念

        Args:
            phases: 逗号分隔的阶段列表，默认全部三阶段
        """
        em = _get_embodied_memory(db_path)
        report = em.run_meditation(phases=phases.split(","))
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

    return mcp
