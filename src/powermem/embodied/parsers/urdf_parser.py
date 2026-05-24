"""
URDF 解析器 — 纯 Python 零 ROS 依赖

将 URDF XML 解析为标准化的 RobotDynamics 和 ParseResult。
支持：link, joint, transmission, material, mesh, collision, visual

URDF 规范参考：http://wiki.ros.org/urdf/XML
"""

from __future__ import annotations

import hashlib
import logging
import math
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from ..physical_model import DHParameter, JointLimit, RobotDynamics
from .base import ModelParser, ParseResult

logger = logging.getLogger(__name__)


class URDFParser(ModelParser):
    """URDF 解析器"""

    @property
    def supported_formats(self) -> List[str]:
        return [".urdf"]

    def parse(self, source: str, **kwargs) -> ParseResult:
        try:
            root = ET.fromstring(source)
        except ET.ParseError as e:
            return ParseResult(
                format="urdf",
                warnings=[f"XML parse error: {e}"],
            )

        result = ParseResult(format="urdf")
        result.source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

        # --- links ---
        links_data: List[Dict[str, Any]] = []
        link_masses: List[float] = []
        link_inertias: List[List[float]] = []

        for link in root.findall("link"):
            link_name = link.get("name", "")
            inertial = link.find("inertial")
            mass = 1.0
            inertia_flat = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]
            com = {"x": 0.0, "y": 0.0, "z": 0.0}

            if inertial is not None:
                mass_elem = inertial.find("mass")
                if mass_elem is not None:
                    mass = float(mass_elem.get("value", 1.0))

                origin = inertial.find("origin")
                if origin is not None:
                    xyz = _parse_xyz(origin.get("xyz", "0 0 0"))
                    com = {"x": xyz[0], "y": xyz[1], "z": xyz[2]}

                inertia = inertial.find("inertia")
                if inertia is not None:
                    inertia_flat = _parse_inertia(inertia)

            link_masses.append(mass)
            link_inertias.append(inertia_flat)

            # visual / collision meshes + collision geometry
            mesh_paths = []
            collision_geoms: List[Dict[str, Any]] = []
            for geom_type in ("visual", "collision"):
                for geom in link.findall(geom_type):
                    geo = geom.find("geometry")
                    if geo is None:
                        continue

                    mesh = geo.find("mesh")
                    if mesh is not None:
                        path = mesh.get("filename", "")
                        if path:
                            mesh_paths.append(path)

                    # --- 提取真实碰撞几何 ---
                    if geom_type == "collision":
                        origin = geom.find("origin")
                        oxyz = _parse_xyz(origin.get("xyz", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0)
                        orpy = _parse_rpy(origin.get("rpy", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0)

                        sphere = geo.find("sphere")
                        if sphere is not None:
                            r = float(sphere.get("radius", 0.05))
                            collision_geoms.append({
                                "type": "sphere",
                                "link": link_name,
                                "center": [oxyz[0], oxyz[1], oxyz[2]],
                                "radius": r,
                            })

                        cyl = geo.find("cylinder")
                        if cyl is not None:
                            r = float(cyl.get("radius", 0.05))
                            length = float(cyl.get("length", 0.1))
                            half_axis = _apply_rpy((0.0, 0.0, length / 2.0), orpy)
                            center = (oxyz[0], oxyz[1], oxyz[2])
                            collision_geoms.append({
                                "type": "capsule",
                                "link": link_name,
                                "a": [center[0] - half_axis[0], center[1] - half_axis[1], center[2] - half_axis[2]],
                                "b": [center[0] + half_axis[0], center[1] + half_axis[1], center[2] + half_axis[2]],
                                "radius": r,
                            })

                        box = geo.find("box")
                        if box is not None:
                            size = _parse_xyz(box.get("size", "0.1 0.1 0.1"))
                            collision_geoms.append({
                                "type": "aabb",
                                "link": link_name,
                                "min": [oxyz[0] - size[0] / 2, oxyz[1] - size[1] / 2, oxyz[2] - size[2] / 2],
                                "max": [oxyz[0] + size[0] / 2, oxyz[1] + size[1] / 2, oxyz[2] + size[2] / 2],
                            })

            links_data.append({
                "name": link_name,
                "mass": mass,
                "inertia": inertia_flat,
                "com": com,
                "mesh_paths": mesh_paths,
                "collision_geoms": collision_geoms,
            })
            result.mesh_paths.extend(mesh_paths)

        # --- joints ---
        joints_data: List[Dict[str, Any]] = []
        joint_names: List[str] = []
        joint_limits: List[JointLimit] = []
        dh_params: List[DHParameter] = []

        for joint in root.findall("joint"):
            jname = joint.get("name", "")
            jtype = joint.get("type", "revolute")
            joint_names.append(jname)

            parent = joint.find("parent")
            child = joint.find("child")
            origin = joint.find("origin")
            axis = joint.find("axis")
            limit = joint.find("limit")

            # DH params (approximate from origin)
            if origin is not None:
                xyz = _parse_xyz(origin.get("xyz", "0 0 0"))
                rpy = _parse_rpy(origin.get("rpy", "0 0 0"))
                dh = DHParameter(
                    d=xyz[2],      # z-offset as d
                    theta=rpy[2],  # z-rotation as theta
                    a=xyz[0],      # x-offset as a
                    alpha=rpy[0],  # x-rotation as alpha
                )
            else:
                dh = DHParameter()
            dh_params.append(dh)

            # joint limits
            if limit is not None:
                jl = JointLimit(
                    min_rad=float(limit.get("lower", -3.14159)),
                    max_rad=float(limit.get("upper", 3.14159)),
                    max_vel=float(limit.get("velocity", 1.0)),
                    max_torque=float(limit.get("effort", 10.0)),
                )
            elif jtype in ("fixed", "floating"):
                jl = JointLimit(min_rad=0.0, max_rad=0.0, max_vel=0.0, max_torque=0.0)
            else:
                jl = JointLimit()
            joint_limits.append(jl)

            joints_data.append({
                "name": jname,
                "type": jtype,
                "parent": parent.get("link", "") if parent is not None else "",
                "child": child.get("link", "") if child is not None else "",
                "axis": _parse_xyz(axis.get("xyz", "1 0 0")) if axis is not None else (1.0, 0.0, 0.0),
                "limit": jl.to_dict(),
            })

        # --- materials ---
        for material in root.findall("material"):
            color = material.find("color")
            mat_info = {"name": material.get("name", "")}
            if color is not None:
                rgba = color.get("rgba", "0.5 0.5 0.5 1")
                mat_info["rgba"] = [float(x) for x in rgba.split()]
            result.materials.append(mat_info)

        # 汇总所有碰撞几何到 dynamics
        all_collision_geoms: List[Dict[str, Any]] = []
        for link in links_data:
            all_collision_geoms.extend(link.get("collision_geoms", []))

        result.links = links_data
        result.joints = joints_data
        result.dynamics = RobotDynamics(
            joint_names=joint_names,
            dh_params=dh_params,
            joint_limits=joint_limits,
            link_masses=link_masses,
            link_inertias=link_inertias,
            collision_geoms=all_collision_geoms,
        )

        return result


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _parse_xyz(text: str) -> Tuple[float, float, float]:
    parts = text.strip().split()
    if len(parts) < 3:
        return (0.0, 0.0, 0.0)
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_rpy(text: str) -> Tuple[float, float, float]:
    return _parse_xyz(text)


def _parse_inertia(elem) -> List[float]:
    """解析 3x3 惯性矩阵为扁平列表 [ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz]"""
    ixx = float(elem.get("ixx", 0.001))
    ixy = float(elem.get("ixy", 0.0))
    ixz = float(elem.get("ixz", 0.0))
    iyy = float(elem.get("iyy", 0.001))
    iyz = float(elem.get("iyz", 0.0))
    izz = float(elem.get("izz", 0.001))
    return [ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz]


def _apply_rpy(v: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """对向量 v 应用 extrinsic XYZ 旋转 (roll-pitch-yaw)"""
    x, y, z = v
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Rx
    y1 = cr * y - sr * z
    z1 = sr * y + cr * z
    # Ry
    x2 = cp * x + sp * z1
    z2 = -sp * x + cp * z1
    # Rz
    x3 = cy * x2 - sy * y1
    y3 = sy * x2 + cy * y1

    return (x3, y3, z2)
