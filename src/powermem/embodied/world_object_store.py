"""
世界对象存储 —— 物体身份追踪、场景图与空间关系管理

在 PowerMem 之上提供结构化的世界对象持久化：
- 对象 CRUD（身份、位姿、状态、语义标签）
- 场景图层级（parent_obj_id）
- 空间关系边（on / in / next_to / above / below / touching）
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from ._json import fast_dumps, fast_loads
from .types import Pose, Quaternion, SpatialRelation, Vec3, WorldObject

logger = logging.getLogger(__name__)


class WorldObjectStore:
    """世界对象存储管理器

    操作 embodied_world_objects 与 embodied_spatial_relations 表。
    """

    def __init__(self, db_conn: Any, telemetry: Optional[Any] = None):
        self.db_conn = db_conn
        self._telemetry = telemetry
        # 事务深度（>0 时抑制中间 commit，与 EmbodiedMemory.transaction() 协同）
        self._transaction_depth = 0
        # 读取缓存：避免重复反序列化高频访问的世界对象（OrderedDict 实现真 LRU）
        self._object_cache: OrderedDict[str, WorldObject] = OrderedDict()
        self._MAX_OBJECT_CACHE = 512

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _world_object_to_row(obj: WorldObject) -> Tuple:
        """将 WorldObject 分解为 SQL 参数元组"""
        lcp = obj.last_confirmed_position
        return (
            obj.obj_id,
            obj.obj_type,
            obj.name,
            obj.pose.position.x,
            obj.pose.position.y,
            obj.pose.position.z,
            obj.pose.orientation.w,
            obj.pose.orientation.x,
            obj.pose.orientation.y,
            obj.pose.orientation.z,
            fast_dumps(obj.size) if obj.size else None,
            fast_dumps(obj.color) if obj.color else None,
            obj.mesh_path,
            fast_dumps(obj.physics_props) if obj.physics_props else None,
            fast_dumps(obj.semantic_tags) if obj.semantic_tags else None,
            obj.scene_id,
            obj.parent_obj_id,
            obj.state,
            obj.memory_id,
            obj.occlusion_status,
            lcp.x if lcp else None,
            lcp.y if lcp else None,
            lcp.z if lcp else None,
            obj.confidence,
            obj.last_seen_sec,
        )

    @staticmethod
    def _row_to_world_object(row: Tuple) -> WorldObject:
        """从 DB 行还原 WorldObject"""
        # 兼容旧 schema：如果没有新列，使用默认值
        if len(row) == 19:
            (
                obj_id, obj_type, name,
                pos_x, pos_y, pos_z,
                orient_w, orient_x, orient_y, orient_z,
                size_json, color_json, mesh_path,
                physics_props_json, semantic_tags_json,
                scene_id, parent_obj_id, state, memory_id,
            ) = row
            occlusion_status = "visible"
            last_confirmed_pos_x = last_confirmed_pos_y = last_confirmed_pos_z = None
            confidence = 1.0
            last_seen_sec = 0.0
        else:
            (
                obj_id, obj_type, name,
                pos_x, pos_y, pos_z,
                orient_w, orient_x, orient_y, orient_z,
                size_json, color_json, mesh_path,
                physics_props_json, semantic_tags_json,
                scene_id, parent_obj_id, state, memory_id,
                occlusion_status,
                last_confirmed_pos_x, last_confirmed_pos_y, last_confirmed_pos_z,
                confidence, last_seen_sec,
            ) = row

        size = tuple(fast_loads(size_json)) if size_json else None
        color = tuple(fast_loads(color_json)) if color_json else None
        physics_props = fast_loads(physics_props_json) if physics_props_json else {}
        semantic_tags = fast_loads(semantic_tags_json) if semantic_tags_json else []
        lcp = None
        if last_confirmed_pos_x is not None:
            lcp = Vec3(
                float(last_confirmed_pos_x),
                float(last_confirmed_pos_y),
                float(last_confirmed_pos_z),
            )

        return WorldObject(
            obj_id=obj_id,
            obj_type=obj_type or "box",
            name=name or "",
            pose=Pose(
                position=Vec3(float(pos_x or 0), float(pos_y or 0), float(pos_z or 0)),
                orientation=Quaternion(
                    float(orient_w or 1), float(orient_x or 0),
                    float(orient_y or 0), float(orient_z or 0),
                ),
            ),
            size=size,
            color=color,
            mesh_path=mesh_path,
            physics_props=physics_props,
            semantic_tags=semantic_tags,
            scene_id=scene_id,
            parent_obj_id=parent_obj_id,
            state=state or "present",
            memory_id=int(memory_id) if memory_id is not None else None,
            occlusion_status=occlusion_status or "visible",
            last_confirmed_position=lcp,
            confidence=float(confidence) if confidence is not None else 1.0,
            last_seen_sec=float(last_seen_sec) if last_seen_sec is not None else 0.0,
        )

    # -----------------------------------------------------------------------
    # WorldObject CRUD
    # -----------------------------------------------------------------------

    def save(self, obj: WorldObject) -> str:
        """保存 WorldObject（INSERT OR REPLACE），返回 obj_id"""
        cursor = self.db_conn.cursor()
        sql = """
            INSERT INTO embodied_world_objects (
                obj_id, obj_type, name,
                pos_x, pos_y, pos_z,
                orient_w, orient_x, orient_y, orient_z,
                size_json, color_json, mesh_path,
                physics_props_json, semantic_tags_json,
                scene_id, parent_obj_id, state, memory_id,
                occlusion_status,
                last_confirmed_pos_x, last_confirmed_pos_y, last_confirmed_pos_z,
                confidence, last_seen_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(obj_id) DO UPDATE SET
                obj_type = excluded.obj_type,
                name = excluded.name,
                pos_x = excluded.pos_x, pos_y = excluded.pos_y, pos_z = excluded.pos_z,
                orient_w = excluded.orient_w, orient_x = excluded.orient_x,
                orient_y = excluded.orient_y, orient_z = excluded.orient_z,
                size_json = excluded.size_json,
                color_json = excluded.color_json,
                mesh_path = excluded.mesh_path,
                physics_props_json = excluded.physics_props_json,
                semantic_tags_json = excluded.semantic_tags_json,
                scene_id = excluded.scene_id,
                parent_obj_id = excluded.parent_obj_id,
                state = excluded.state,
                memory_id = excluded.memory_id,
                occlusion_status = excluded.occlusion_status,
                last_confirmed_pos_x = excluded.last_confirmed_pos_x,
                last_confirmed_pos_y = excluded.last_confirmed_pos_y,
                last_confirmed_pos_z = excluded.last_confirmed_pos_z,
                confidence = excluded.confidence,
                last_seen_sec = excluded.last_seen_sec
        """
        cursor.execute(sql, self._world_object_to_row(obj))
        if self._transaction_depth == 0:
            self.db_conn.commit()
        self._cache_object(obj)
        logger.debug("Saved world object %s (%s)", obj.obj_id, obj.name)
        return obj.obj_id

    def _cache_object(self, obj: WorldObject) -> None:
        self._object_cache[obj.obj_id] = obj
        self._object_cache.move_to_end(obj.obj_id)
        while len(self._object_cache) > self._MAX_OBJECT_CACHE:
            self._object_cache.popitem(last=False)

    def _invalidate_object_cache(self, obj_id: str) -> None:
        self._object_cache.pop(obj_id, None)

    def load(self, obj_id: str) -> Optional[WorldObject]:
        """按 obj_id 读取 WorldObject（带 LRU 缓存）"""
        cached = self._object_cache.get(obj_id)
        if cached is not None:
            self._object_cache.move_to_end(obj_id)
            if self._telemetry is not None:
                self._telemetry.record_cache_hit("world_object")
            return cached
        if self._telemetry is not None:
            self._telemetry.record_cache_miss("world_object")
        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT obj_id, obj_type, name, pos_x, pos_y, pos_z, "
            "orient_w, orient_x, orient_y, orient_z, "
            "size_json, color_json, mesh_path, "
            "physics_props_json, semantic_tags_json, "
            "scene_id, parent_obj_id, state, memory_id, "
            "occlusion_status, last_confirmed_pos_x, last_confirmed_pos_y, last_confirmed_pos_z, "
            "confidence, last_seen_sec "
            "FROM embodied_world_objects WHERE obj_id = ?",
            (obj_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        obj = self._row_to_world_object(row)
        self._cache_object(obj)
        return obj

    def load_many(self, obj_ids: List[str]) -> List[WorldObject]:
        """批量读取 WorldObject"""
        if not obj_ids:
            return []
        cursor = self.db_conn.cursor()
        placeholders = ", ".join("?" for _ in obj_ids)
        cursor.execute(
            "SELECT obj_id, obj_type, name, pos_x, pos_y, pos_z, "
            "orient_w, orient_x, orient_y, orient_z, "
            "size_json, color_json, mesh_path, "
            "physics_props_json, semantic_tags_json, "
            "scene_id, parent_obj_id, state, memory_id, "
            "occlusion_status, last_confirmed_pos_x, last_confirmed_pos_y, last_confirmed_pos_z, "
            "confidence, last_seen_sec "
            "FROM embodied_world_objects WHERE obj_id IN (" + placeholders + ")",
            tuple(obj_ids),
        )
        return [self._row_to_world_object(row) for row in cursor.fetchall()]

    def list_by_scene(
        self,
        scene_id: str,
        obj_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[WorldObject]:
        """按场景列举对象"""
        cursor = self.db_conn.cursor()
        if obj_type:
            cursor.execute(
                "SELECT obj_id, obj_type, name, pos_x, pos_y, pos_z, "
                "orient_w, orient_x, orient_y, orient_z, "
                "size_json, color_json, mesh_path, "
                "physics_props_json, semantic_tags_json, "
                "scene_id, parent_obj_id, state, memory_id, "
                "occlusion_status, last_confirmed_pos_x, last_confirmed_pos_y, last_confirmed_pos_z, "
                "confidence, last_seen_sec "
                "FROM embodied_world_objects WHERE scene_id = ? AND obj_type = ? LIMIT ?",
                (scene_id, obj_type, limit),
            )
        else:
            cursor.execute(
                "SELECT obj_id, obj_type, name, pos_x, pos_y, pos_z, "
                "orient_w, orient_x, orient_y, orient_z, "
                "size_json, color_json, mesh_path, "
                "physics_props_json, semantic_tags_json, "
                "scene_id, parent_obj_id, state, memory_id, "
                "occlusion_status, last_confirmed_pos_x, last_confirmed_pos_y, last_confirmed_pos_z, "
                "confidence, last_seen_sec "
                "FROM embodied_world_objects WHERE scene_id = ? LIMIT ?",
                (scene_id, limit),
            )
        return [self._row_to_world_object(row) for row in cursor.fetchall()]

    def list_by_type(
        self,
        obj_type: str,
        scene_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[WorldObject]:
        """按类型列举对象"""
        cursor = self.db_conn.cursor()
        if scene_id:
            cursor.execute(
                "SELECT obj_id, obj_type, name, pos_x, pos_y, pos_z, "
                "orient_w, orient_x, orient_y, orient_z, "
                "size_json, color_json, mesh_path, "
                "physics_props_json, semantic_tags_json, "
                "scene_id, parent_obj_id, state, memory_id, "
                "occlusion_status, last_confirmed_pos_x, last_confirmed_pos_y, last_confirmed_pos_z, "
                "confidence, last_seen_sec "
                "FROM embodied_world_objects WHERE obj_type = ? AND scene_id = ? LIMIT ?",
                (obj_type, scene_id, limit),
            )
        else:
            cursor.execute(
                "SELECT obj_id, obj_type, name, pos_x, pos_y, pos_z, "
                "orient_w, orient_x, orient_y, orient_z, "
                "size_json, color_json, mesh_path, "
                "physics_props_json, semantic_tags_json, "
                "scene_id, parent_obj_id, state, memory_id "
                "FROM embodied_world_objects WHERE obj_type = ? LIMIT ?",
                (obj_type, limit),
            )
        return [self._row_to_world_object(row) for row in cursor.fetchall()]

    def update_pose(self, obj_id: str, pose: Pose, state: Optional[str] = None) -> bool:
        """更新对象位姿和可选状态，同时重置遮挡置信度"""
        cursor = self.db_conn.cursor()
        if state is not None:
            cursor.execute(
                "UPDATE embodied_world_objects SET "
                "pos_x = ?, pos_y = ?, pos_z = ?, "
                "orient_w = ?, orient_x = ?, orient_y = ?, orient_z = ?, "
                "state = ?, occlusion_status = 'visible', confidence = 1.0, "
                "last_confirmed_pos_x = ?, last_confirmed_pos_y = ?, last_confirmed_pos_z = ? "
                "WHERE obj_id = ?",
                (
                    pose.position.x, pose.position.y, pose.position.z,
                    pose.orientation.w, pose.orientation.x,
                    pose.orientation.y, pose.orientation.z,
                    state,
                    pose.position.x, pose.position.y, pose.position.z,
                    obj_id,
                ),
            )
        else:
            cursor.execute(
                "UPDATE embodied_world_objects SET "
                "pos_x = ?, pos_y = ?, pos_z = ?, "
                "orient_w = ?, orient_x = ?, orient_y = ?, orient_z = ?, "
                "occlusion_status = 'visible', confidence = 1.0, "
                "last_confirmed_pos_x = ?, last_confirmed_pos_y = ?, last_confirmed_pos_z = ? "
                "WHERE obj_id = ?",
                (
                    pose.position.x, pose.position.y, pose.position.z,
                    pose.orientation.w, pose.orientation.x,
                    pose.orientation.y, pose.orientation.z,
                    pose.position.x, pose.position.y, pose.position.z,
                    obj_id,
                ),
            )
        if self._transaction_depth == 0:
            self.db_conn.commit()
        self._invalidate_object_cache(obj_id)
        return cursor.rowcount > 0

    def update_state(self, obj_id: str, state: str) -> bool:
        """更新对象状态"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            "UPDATE embodied_world_objects SET state = ? WHERE obj_id = ?",
            (state, obj_id),
        )
        if self._transaction_depth == 0:
            self.db_conn.commit()
        self._invalidate_object_cache(obj_id)
        return cursor.rowcount > 0

    def update_occlusion(
        self,
        obj_id: str,
        occlusion_status: str,
        confidence: float,
        last_seen_sec: float,
    ) -> bool:
        """更新对象遮挡状态和存在置信度"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            "UPDATE embodied_world_objects SET "
            "occlusion_status = ?, confidence = ?, last_seen_sec = ? "
            "WHERE obj_id = ?",
            (occlusion_status, confidence, last_seen_sec, obj_id),
        )
        if self._transaction_depth == 0:
            self.db_conn.commit()
        self._invalidate_object_cache(obj_id)
        return cursor.rowcount > 0

    def delete(self, obj_id: str) -> bool:
        """删除对象（级联删除其空间关系）"""
        cursor = self.db_conn.cursor()
        cursor.execute("DELETE FROM embodied_world_objects WHERE obj_id = ?", (obj_id,))
        if self._transaction_depth == 0:
            self.db_conn.commit()
        self._invalidate_object_cache(obj_id)
        return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # 空间关系管理
    # -----------------------------------------------------------------------

    def add_relation(self, relation: SpatialRelation) -> int:
        """添加空间关系，返回关系 ID"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            "INSERT INTO embodied_spatial_relations (subject_id, object_id, relation, confidence) "
            "VALUES (?, ?, ?, ?)",
            (relation.subject_id, relation.object_id, relation.relation, relation.confidence),
        )
        if self._transaction_depth == 0:
            self.db_conn.commit()
        return cursor.lastrowid

    def get_relations(
        self,
        obj_id: str,
        direction: str = "both",
    ) -> List[SpatialRelation]:
        """获取对象的空间关系

        Args:
            obj_id: 对象 ID
            direction: "outgoing" | "incoming" | "both"
        """
        cursor = self.db_conn.cursor()
        relations: List[SpatialRelation] = []

        if direction in ("outgoing", "both"):
            cursor.execute(
                "SELECT subject_id, object_id, relation, confidence "
                "FROM embodied_spatial_relations WHERE subject_id = ?",
                (obj_id,),
            )
            for row in cursor.fetchall():
                relations.append(SpatialRelation(*row))

        if direction in ("incoming", "both"):
            cursor.execute(
                "SELECT subject_id, object_id, relation, confidence "
                "FROM embodied_spatial_relations WHERE object_id = ?",
                (obj_id,),
            )
            for row in cursor.fetchall():
                relations.append(SpatialRelation(*row))

        return relations

    def get_scene_graph(self, scene_id: str) -> Dict[str, List[SpatialRelation]]:
        """获取整个场景的关系图

        Returns:
            {obj_id: [SpatialRelation, ...]} —— 包含该对象的所有出边和入边
        """
        objects = self.list_by_scene(scene_id)
        obj_ids = [o.obj_id for o in objects]
        if not obj_ids:
            return {}

        cursor = self.db_conn.cursor()
        placeholders = ", ".join("?" for _ in obj_ids)
        cursor.execute(
            "SELECT subject_id, object_id, relation, confidence "
            "FROM embodied_spatial_relations "
            "WHERE subject_id IN (" + placeholders + ") OR object_id IN (" + placeholders + ")",
            tuple(obj_ids + obj_ids),
        )

        graph: Dict[str, List[SpatialRelation]] = {oid: [] for oid in obj_ids}
        for row in cursor.fetchall():
            rel = SpatialRelation(*row)
            if rel.subject_id in graph:
                graph[rel.subject_id].append(rel)
            if rel.object_id in graph:
                graph[rel.object_id].append(rel)
        return graph

    def delete_relations_for_object(self, obj_id: str) -> int:
        """删除对象的所有空间关系（用于清理）"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            "DELETE FROM embodied_spatial_relations WHERE subject_id = ? OR object_id = ?",
            (obj_id, obj_id),
        )
        if self._transaction_depth == 0:
            self.db_conn.commit()
        return cursor.rowcount
