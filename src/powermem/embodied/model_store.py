"""
物理模型存储 — ParseResult 的持久化与版本管理

将多格式解析结果（URDF/MJCF/SDF/Xacro/USD）存入 embodied_physical_models 表，
支持版本控制、碰撞体构建、模型检索。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._json import fast_dumps, fast_loads
from .collision import CollisionBody, CollisionChecker, build_collision_bodies
from .parsers.base import ParseResult
from .physical_model import RobotDynamics

logger = logging.getLogger(__name__)


@dataclass
class StoredModel:
    """从数据库还原的物理模型"""
    model_id: str
    model_type: str
    dynamics: RobotDynamics
    collision_bodies: List[CollisionBody] = field(default_factory=list)
    source_hash: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class ModelStore:
    """物理模型存储管理器

    操作 embodied_physical_models 表，提供：
    - save(): 将 ParseResult 持久化
    - load(): 按 model_id 读取
    - list_models(): 按类型枚举
    - build_checker(): 从存储模型构建 CollisionChecker
    """

    def __init__(self, db_conn: Any):
        self.db_conn = db_conn

    # -----------------------------------------------------------------------
    # 写入
    # -----------------------------------------------------------------------

    def save(
        self,
        result: ParseResult,
        model_id: Optional[str] = None,
        model_type: str = "robot",
    ) -> str:
        """保存 ParseResult 到数据库

        Args:
            result: 解析结果
            model_id: 显式指定模型 ID，None 时自动生成
            model_type: 'robot' | 'environment' | 'object'

        Returns:
            实际使用的 model_id
        """
        if model_id is None:
            model_id = self._auto_model_id(result)

        dyn = result.dynamics
        cursor = self.db_conn.cursor()
        sql = """
            INSERT INTO embodied_physical_models (
                model_id, model_type,
                joint_names, dh_params, mass_matrix, coriolis_matrix, gravity_vector,
                joint_limits, link_masses, link_inertias, collision_geoms, source_urdf_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                model_type = excluded.model_type,
                joint_names = excluded.joint_names,
                dh_params = excluded.dh_params,
                mass_matrix = excluded.mass_matrix,
                coriolis_matrix = excluded.coriolis_matrix,
                gravity_vector = excluded.gravity_vector,
                joint_limits = excluded.joint_limits,
                link_masses = excluded.link_masses,
                link_inertias = excluded.link_inertias,
                collision_geoms = excluded.collision_geoms,
                source_urdf_hash = excluded.source_urdf_hash,
                updated_at = CURRENT_TIMESTAMP
        """
        params = (
            model_id,
            model_type,
            fast_dumps(dyn.joint_names),
            fast_dumps([dh.to_dict() for dh in dyn.dh_params]),
            dyn.mass_matrix_expr,
            dyn.coriolis_matrix_expr,
            dyn.gravity_vector_expr,
            fast_dumps([jl.to_dict() for jl in dyn.joint_limits]),
            fast_dumps(dyn.link_masses),
            fast_dumps(dyn.link_inertias),
            fast_dumps(dyn.collision_geoms),
            result.source_hash or self._hash_source(result),
        )
        try:
            cursor.execute(sql, params)
            self.db_conn.commit()
            logger.debug("Saved physical model %s (%s)", model_id, model_type)
        except Exception as e:
            logger.warning("Failed to save model %s: %s", model_id, e)
            raise
        return model_id

    def _auto_model_id(self, result: ParseResult) -> str:
        """自动生成 model_id：格式_关节数_哈希前8位"""
        fmt = result.format or "unknown"
        n_joints = len(result.dynamics.joint_names)
        h = result.source_hash or self._hash_source(result)
        return f"{fmt}_{n_joints}dof_{h[:8]}"

    @staticmethod
    def _hash_source(result: ParseResult) -> str:
        """对解析结果做稳定哈希"""
        payload = fast_dumps(result.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # -----------------------------------------------------------------------
    # 读取
    # -----------------------------------------------------------------------

    def load(self, model_id: str) -> Optional[StoredModel]:
        """按 model_id 读取模型"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT model_type, joint_names, dh_params, mass_matrix, coriolis_matrix, "
            "gravity_vector, joint_limits, link_masses, link_inertias, "
            "collision_geoms, source_urdf_hash "
            "FROM embodied_physical_models WHERE model_id = ?",
            (model_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        (
            model_type, joint_names_json, dh_params_json, mass_matrix, coriolis_matrix,
            gravity_vector, joint_limits_json, link_masses_json, link_inertias_json,
            collision_geoms_json, source_hash,
        ) = row

        from .physical_model import DHParameter, JointLimit
        joint_names = fast_loads(joint_names_json or "[]")
        dh_params = [DHParameter.from_dict(d) for d in fast_loads(dh_params_json or "[]")]
        joint_limits = [JointLimit.from_dict(d) for d in fast_loads(joint_limits_json or "[]")]
        link_masses = fast_loads(link_masses_json or "[]")
        link_inertias = fast_loads(link_inertias_json or "[]")
        collision_geoms = fast_loads(collision_geoms_json or "[]")

        dynamics = RobotDynamics(
            joint_names=joint_names,
            dh_params=dh_params,
            mass_matrix_expr=mass_matrix or "{}",
            coriolis_matrix_expr=coriolis_matrix or "{}",
            gravity_vector_expr=gravity_vector or "{}",
            joint_limits=joint_limits,
            link_masses=link_masses,
            link_inertias=link_inertias,
            collision_geoms=collision_geoms,
        )

        # 从 links 构建碰撞体（如果 collision_geoms 为空则回退到 build_collision_bodies）
        bodies: List[CollisionBody] = []
        if collision_geoms:
            bodies = self._geoms_to_bodies(collision_geoms, model_id)

        return StoredModel(
            model_id=model_id,
            model_type=model_type or "robot",
            dynamics=dynamics,
            collision_bodies=bodies,
            source_hash=source_hash or "",
        )

    def _geoms_to_bodies(
        self,
        geoms: List[Dict[str, Any]],
        model_id: str,
    ) -> List[CollisionBody]:
        """将 JSON 碰撞体描述还原为 CollisionBody"""
        from .collision import AABB, Capsule, Sphere
        from .types import Vec3

        bodies: List[CollisionBody] = []
        for g in geoms:
            gtype = g.get("type", "sphere")
            link = g.get("link", "")
            frame = g.get("frame", "world")
            if gtype == "sphere":
                c = Vec3(*g.get("center", [0.0, 0.0, 0.0]))
                r = float(g.get("radius", 0.05))
                bodies.append(CollisionBody(
                    entity_id=f"{model_id}:{link}",
                    geom_type="sphere",
                    geometry=Sphere(center=c, radius=r),
                    link_name=link,
                    frame_id=frame,
                ))
            elif gtype == "capsule":
                a = Vec3(*g.get("a", [0.0, 0.0, 0.0]))
                b = Vec3(*g.get("b", [0.0, 0.0, 0.0]))
                r = float(g.get("radius", 0.05))
                bodies.append(CollisionBody(
                    entity_id=f"{model_id}:{link}",
                    geom_type="capsule",
                    geometry=Capsule(a=a, b=b, radius=r),
                    link_name=link,
                    frame_id=frame,
                ))
            elif gtype == "aabb":
                mn = Vec3(*g.get("min", [0.0, 0.0, 0.0]))
                mx = Vec3(*g.get("max", [0.0, 0.0, 0.0]))
                bodies.append(CollisionBody(
                    entity_id=f"{model_id}:{link}",
                    geom_type="aabb",
                    geometry=AABB(min=mn, max=mx),
                    link_name=link,
                    frame_id=frame,
                ))
        return bodies

    def list_models(
        self,
        model_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """枚举存储的模型摘要"""
        cursor = self.db_conn.cursor()
        if model_type:
            sql = (
                "SELECT model_id, model_type, source_urdf_hash, created_at "
                "FROM embodied_physical_models WHERE model_type = ? LIMIT ?"
            )
            cursor.execute(sql, (model_type, limit))
        else:
            sql = (
                "SELECT model_id, model_type, source_urdf_hash, created_at "
                "FROM embodied_physical_models LIMIT ?"
            )
            cursor.execute(sql, (limit,))

        cols = [desc[0] for desc in cursor.description]
        rows = []
        for row in cursor.fetchall():
            rows.append(dict(zip(cols, row)))
        return rows

    def delete(self, model_id: str) -> bool:
        """删除模型"""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM embodied_physical_models WHERE model_id = ?",
                (model_id,),
            )
            self.db_conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.warning("Failed to delete model %s: %s", model_id, e)
            return False

    # -----------------------------------------------------------------------
    # 碰撞检测快捷方法
    # -----------------------------------------------------------------------

    def build_checker(self, model_id: str) -> Optional[CollisionChecker]:
        """从存储模型构建 CollisionChecker"""
        model = self.load(model_id)
        if model is None:
            return None
        checker = CollisionChecker()
        if model.collision_bodies:
            checker.add_bodies(model.collision_bodies)
        return checker

    def check_self_collision(self, model_id: str) -> List[tuple]:
        """检测模型自碰撞，返回相交的碰撞体对"""
        checker = self.build_checker(model_id)
        if checker is None:
            return []
        return checker.check_intersections()

    def check_collision_at_config(
        self,
        model_id: str,
        joint_angles: List[float],
    ) -> List[tuple]:
        """在指定关节角配置下检测碰撞（世界坐标系）

        Args:
            model_id: 模型 ID
            joint_angles: 关节角列表（弧度）

        Returns:
            相交的碰撞体对列表
        """
        from .kinematics import forward_kinematics, transform_collision_bodies

        model = self.load(model_id)
        if model is None or not model.collision_bodies:
            return []

        # 构建 link_name -> joint_index 映射（按 joint_names 顺序）
        # 简化：假设 collision body 的 link_name 对应 dynamics 中的 joint index
        link_index_map = {name: i for i, name in enumerate(model.dynamics.joint_names)}

        transforms = forward_kinematics(model.dynamics.dh_params, joint_angles)
        world_bodies = transform_collision_bodies(
            model.collision_bodies, transforms, link_index_map
        )

        checker = CollisionChecker()
        checker.add_bodies(world_bodies)
        return checker.check_intersections()
