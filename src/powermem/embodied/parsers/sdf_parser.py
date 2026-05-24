"""
SDF 解析器 — Gazebo 模拟格式

SDF 与 URDF 类似但更丰富，支持世界定义、插件、传感器等。
这里只提取机器人相关的动力学信息。

参考：http://sdformat.org/
"""

from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

from ..physical_model import DHParameter, JointLimit, RobotDynamics
from .base import ModelParser, ParseResult

logger = logging.getLogger(__name__)


class SDFParser(ModelParser):
    """SDF (Gazebo) 解析器"""

    @property
    def supported_formats(self) -> List[str]:
        return [".sdf"]

    def parse(self, source: str, **kwargs) -> ParseResult:
        try:
            root = ET.fromstring(source)
        except ET.ParseError as e:
            return ParseResult(format="sdf", warnings=[f"XML parse error: {e}"])

        result = ParseResult(format="sdf")
        result.source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

        # SDF 根是 <sdf>，里面是 <world> 或 <model>
        model = root.find("model")
        if model is None:
            model = root  # 有时直接是 <model> 作为根

        if model is None or model.tag not in ("model", "world"):
            result.warnings.append("No <model> found in SDF")
            return result

        if model.tag == "world":
            model = model.find("model")
            if model is None:
                result.warnings.append("No <model> found inside <world>")
                return result

        links_data: List[Dict[str, Any]] = []
        joints_data: List[Dict[str, Any]] = []
        joint_names: List[str] = []
        joint_limits: List[JointLimit] = []
        dh_params: List[DHParameter] = []
        link_masses: List[float] = []
        link_inertias: List[List[float]] = []

        # --- links ---
        for link in model.findall("link"):
            lname = link.get("name", "")
            inertial = link.find("inertial")
            mass = 1.0
            com = {"x": 0.0, "y": 0.0, "z": 0.0}
            inertia_flat = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]

            if inertial is not None:
                mass_elem = inertial.find("mass")
                if mass_elem is not None:
                    mass = float(mass_elem.text or 1.0)

                pose = inertial.find("pose")
                if pose is not None:
                    vals = (pose.text or "0 0 0 0 0 0").split()
                    if len(vals) >= 3:
                        com = {"x": float(vals[0]), "y": float(vals[1]), "z": float(vals[2])}

                inertia = inertial.find("inertia")
                if inertia is not None:
                    inertia_flat = [
                        float(inertia.findtext("ixx", "0.001")),
                        float(inertia.findtext("ixy", "0.0")),
                        float(inertia.findtext("ixz", "0.0")),
                        float(inertia.findtext("ixy", "0.0")),
                        float(inertia.findtext("iyy", "0.001")),
                        float(inertia.findtext("iyz", "0.0")),
                        float(inertia.findtext("ixz", "0.0")),
                        float(inertia.findtext("iyz", "0.0")),
                        float(inertia.findtext("izz", "0.001")),
                    ]

            link_masses.append(mass)
            link_inertias.append(inertia_flat)

            mesh_paths = []
            collision_geoms: List[Dict[str, Any]] = []
            for geom_type in ("visual", "collision"):
                for geom in link.findall(geom_type):
                    geo = geom.find("geometry")
                    if geo is None:
                        continue

                    mesh = geo.find("mesh")
                    if mesh is not None:
                        uri = mesh.findtext("uri", "")
                        if uri:
                            mesh_paths.append(uri)

                    if geom_type == "collision":
                        pose = geom.find("pose")
                        pxyz = _parse_sdf_pose(pose)

                        sphere = geo.find("sphere")
                        if sphere is not None:
                            r = float(sphere.findtext("radius", "0.05"))
                            collision_geoms.append({
                                "type": "sphere",
                                "link": lname,
                                "center": [pxyz[0], pxyz[1], pxyz[2]],
                                "radius": r,
                            })

                        cyl = geo.find("cylinder")
                        if cyl is not None:
                            r = float(cyl.findtext("radius", "0.05"))
                            length = float(cyl.findtext("length", "0.1"))
                            collision_geoms.append({
                                "type": "capsule",
                                "link": lname,
                                "a": [pxyz[0], pxyz[1], pxyz[2] - length / 2],
                                "b": [pxyz[0], pxyz[1], pxyz[2] + length / 2],
                                "radius": r,
                            })

                        box = geo.find("box")
                        if box is not None:
                            size = _parse_xyz(box.findtext("size", "0.1 0.1 0.1"))
                            collision_geoms.append({
                                "type": "aabb",
                                "link": lname,
                                "min": [pxyz[0] - size[0] / 2, pxyz[1] - size[1] / 2, pxyz[2] - size[2] / 2],
                                "max": [pxyz[0] + size[0] / 2, pxyz[1] + size[1] / 2, pxyz[2] + size[2] / 2],
                            })

            links_data.append({
                "name": lname,
                "mass": mass,
                "inertia": inertia_flat,
                "com": com,
                "mesh_paths": mesh_paths,
                "collision_geoms": collision_geoms,
            })
            result.mesh_paths.extend(mesh_paths)

        # --- joints ---
        for joint in model.findall("joint"):
            jname = joint.get("name", "")
            jtype = joint.get("type", "revolute")
            joint_names.append(jname)

            parent = joint.find("parent")
            child = joint.find("child")
            pose = joint.find("pose")
            axis = joint.find("axis")
            limit = joint.find("axis/limit") or joint.find("limit")

            if pose is not None:
                vals = (pose.text or "0 0 0 0 0 0").split()
                if len(vals) >= 6:
                    dh = DHParameter(
                        d=float(vals[2]),
                        theta=float(vals[5]),
                        a=float(vals[0]),
                        alpha=float(vals[3]),
                    )
                else:
                    dh = DHParameter()
            else:
                dh = DHParameter()
            dh_params.append(dh)

            if limit is not None:
                lower = float(limit.findtext("lower", "-3.14159"))
                upper = float(limit.findtext("upper", "3.14159"))
                velocity = float(limit.findtext("velocity", "1.0"))
                effort = float(limit.findtext("effort", "10.0"))
                jl = JointLimit(min_rad=lower, max_rad=upper, max_vel=velocity, max_torque=effort)
            elif jtype in ("fixed", "ball"):
                jl = JointLimit(min_rad=0.0, max_rad=0.0, max_vel=0.0, max_torque=0.0)
            else:
                jl = JointLimit()
            joint_limits.append(jl)

            axis_xyz = (1.0, 0.0, 0.0)
            if axis is not None:
                xyz_elem = axis.find("xyz")
                if xyz_elem is not None:
                    parts = (xyz_elem.text or "1 0 0").split()
                    if len(parts) >= 3:
                        axis_xyz = (float(parts[0]), float(parts[1]), float(parts[2]))

            joints_data.append({
                "name": jname,
                "type": jtype,
                "parent": parent.text if parent is not None else "",
                "child": child.text if child is not None else "",
                "axis": axis_xyz,
                "limit": jl.to_dict(),
            })

        # --- world objects ---
        for obj in root.findall("world/model"):
            if obj is not model:
                result.world_objects.append({
                    "name": obj.get("name", ""),
                    "type": "model",
                })

        # 汇总碰撞几何
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


def _parse_sdf_pose(pose) -> Tuple[float, float, float]:
    """从 SDF <pose> 元素提取 xyz"""
    if pose is None or pose.text is None:
        return (0.0, 0.0, 0.0)
    vals = pose.text.strip().split()
    if len(vals) >= 3:
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    return (0.0, 0.0, 0.0)


def _parse_xyz(text: str) -> Tuple[float, float, float]:
    parts = text.strip().split()
    if len(parts) < 3:
        return (0.0, 0.0, 0.0)
    return (float(parts[0]), float(parts[1]), float(parts[2]))
