"""
MJCF 解析器 — MuJoCo XML 格式

MuJoCo 的 MJCF 格式与 URDF 有很大不同：
- 世界体是一个树，根节点是 <worldbody>
- 每个 <body> 可以包含 <joint>、<geom>、<body>
- 惯性在 <inertial> 中，或自动从 geom 计算
- 使用 <compiler> 设置角度单位、坐标系等

参考：https://mujoco.readthedocs.io/en/latest/XMLreference.html
"""

from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from ..physical_model import DHParameter, JointLimit, RobotDynamics
from .base import ModelParser, ParseResult

logger = logging.getLogger(__name__)


class MJCFParser(ModelParser):
    """MJCF (MuJoCo) 解析器"""

    @property
    def supported_formats(self) -> List[str]:
        return [".xml", ".mjcf"]

    def parse(self, source: str, **kwargs) -> ParseResult:
        try:
            root = ET.fromstring(source)
        except ET.ParseError as e:
            return ParseResult(format="mjcf", warnings=[f"XML parse error: {e}"])

        result = ParseResult(format="mjcf")
        result.source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

        # --- 编译器选项 ---
        compiler = root.find("compiler")
        angle_unit = "rad"
        if compiler is not None:
            angle = compiler.get("angle", "rad")
            angle_unit = angle

        # --- 关节默认参数 ---
        defaults = self._parse_defaults(root)

        # --- 遍历 body 树 ---
        worldbody = root.find("worldbody")
        if worldbody is None:
            result.warnings.append("No <worldbody> found")
            return result

        links_data: List[Dict[str, Any]] = []
        joints_data: List[Dict[str, Any]] = []
        joint_names: List[str] = []
        joint_limits: List[JointLimit] = []
        dh_params: List[DHParameter] = []
        link_masses: List[float] = []
        link_inertias: List[List[float]] = []

        def traverse_body(body_elem, parent_name="world"):
            bname = body_elem.get("name", "")
            if not bname:
                bname = f"body_{len(links_data)}"

            # inertial
            inertial = body_elem.find("inertial")
            mass = 0.0
            com = {"x": 0.0, "y": 0.0, "z": 0.0}
            inertia_flat = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]

            if inertial is not None:
                mass = float(inertial.get("mass", 0.0))
                pos = inertial.get("pos", "0 0 0").split()
                if len(pos) >= 3:
                    com = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
                diag = inertial.get("diaginertia", "0.001 0.001 0.001").split()
                if len(diag) >= 3:
                    inertia_flat = [
                        float(diag[0]), 0.0, 0.0,
                        0.0, float(diag[1]), 0.0,
                        0.0, 0.0, float(diag[2]),
                    ]
            else:
                # 从 geom 累加质量
                for geom in body_elem.findall("geom"):
                    gm = float(geom.get("mass", 0.0))
                    if gm > 0:
                        mass += gm

            link_masses.append(mass if mass > 0 else 0.001)
            link_inertias.append(inertia_flat)

            mesh_paths = []
            collision_geoms: List[Dict[str, Any]] = []
            for geom in body_elem.findall("geom"):
                mesh_name = geom.get("mesh", "")
                if mesh_name:
                    mesh_paths.append(mesh_name)

                gtype = geom.get("type", "sphere")
                pos = _parse_xyz(geom.get("pos", "0 0 0"))
                size_str = geom.get("size", "0.05")
                size_parts = [float(x) for x in size_str.split()]

                if gtype == "sphere":
                    r = size_parts[0] if size_parts else 0.05
                    collision_geoms.append({
                        "type": "sphere",
                        "link": bname,
                        "center": [pos[0], pos[1], pos[2]],
                        "radius": r,
                    })
                elif gtype == "capsule":
                    r = size_parts[0] if size_parts else 0.05
                    fromto = geom.get("fromto", "")
                    if fromto:
                        ft = [float(x) for x in fromto.split()]
                        if len(ft) >= 6:
                            collision_geoms.append({
                                "type": "capsule",
                                "link": bname,
                                "a": [ft[0], ft[1], ft[2]],
                                "b": [ft[3], ft[4], ft[5]],
                                "radius": r,
                            })
                    else:
                        # fallback to sphere if no fromto
                        collision_geoms.append({
                            "type": "sphere",
                            "link": bname,
                            "center": [pos[0], pos[1], pos[2]],
                            "radius": r,
                        })
                elif gtype == "cylinder":
                    r = size_parts[0] if size_parts else 0.05
                    h = float(geom.get("height", "0.1"))
                    # approximate as capsule along Z
                    half_h = h / 2.0
                    collision_geoms.append({
                        "type": "capsule",
                        "link": bname,
                        "a": [pos[0], pos[1], pos[2] - half_h],
                        "b": [pos[0], pos[1], pos[2] + half_h],
                        "radius": r,
                    })
                elif gtype == "box":
                    # size in MuJoCo is half-extents
                    hx = size_parts[0] if len(size_parts) > 0 else 0.05
                    hy = size_parts[1] if len(size_parts) > 1 else hx
                    hz = size_parts[2] if len(size_parts) > 2 else hx
                    collision_geoms.append({
                        "type": "aabb",
                        "link": bname,
                        "min": [pos[0] - hx, pos[1] - hy, pos[2] - hz],
                        "max": [pos[0] + hx, pos[1] + hy, pos[2] + hz],
                    })

            links_data.append({
                "name": bname,
                "mass": mass,
                "inertia": inertia_flat,
                "com": com,
                "mesh_paths": mesh_paths,
                "collision_geoms": collision_geoms,
                "parent": parent_name,
            })
            result.mesh_paths.extend(mesh_paths)

            # joints
            for joint in body_elem.findall("joint"):
                jname = joint.get("name", "")
                if not jname:
                    jname = f"joint_{len(joint_names)}"
                jtype = joint.get("type", "hinge")
                joint_names.append(jname)

                axis = joint.get("axis", "0 0 1").split()
                pos = joint.get("pos", "0 0 0").split()
                range_str = joint.get("range", "-3.14159 3.14159").split()

                if angle_unit == "degree":
                    range_vals = [float(x) * 3.14159 / 180.0 for x in range_str]
                else:
                    range_vals = [float(x) for x in range_str]

                jl = JointLimit(
                    min_rad=range_vals[0] if len(range_vals) > 0 else -3.14159,
                    max_rad=range_vals[1] if len(range_vals) > 1 else 3.14159,
                    max_vel=float(joint.get("damping", 1.0)),  # 近似
                    max_torque=float(defaults.get("joint", {}).get("armature", 10.0)),
                )
                joint_limits.append(jl)

                # 简化 DH 参数
                if len(pos) >= 3:
                    dh = DHParameter(
                        d=float(pos[2]),
                        theta=0.0,
                        a=float(pos[0]),
                        alpha=0.0,
                    )
                else:
                    dh = DHParameter()
                dh_params.append(dh)

                joints_data.append({
                    "name": jname,
                    "type": jtype,
                    "parent": bname,
                    "child": "",  # MJCF 中 joint 在 body 内，child 是 body 自身
                    "axis": tuple(float(x) for x in axis[:3]),
                    "limit": jl.to_dict(),
                })

            # 递归子 body
            for child in body_elem.findall("body"):
                traverse_body(child, bname)

        for body in worldbody.findall("body"):
            traverse_body(body, "world")

        # --- assets (meshes) ---
        asset = root.find("asset")
        if asset is not None:
            for mesh in asset.findall("mesh"):
                name = mesh.get("name", "")
                file_path = mesh.get("file", "")
                if file_path:
                    result.mesh_paths.append(file_path)
            for material in asset.findall("material"):
                mat_info = {"name": material.get("name", "")}
                rgba = material.get("rgba", "")
                if rgba:
                    mat_info["rgba"] = [float(x) for x in rgba.split()]
                result.materials.append(mat_info)

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

    def _parse_defaults(self, root: ET.Element) -> Dict[str, Dict[str, Any]]:
        """解析 <default> 元素"""
        defaults: Dict[str, Dict[str, Any]] = {}
        for default in root.findall("default"):
            for child in default:
                cls_name = child.tag
                defaults.setdefault(cls_name, {}).update(child.attrib)
        return defaults


def _parse_xyz(text: str) -> Tuple[float, float, float]:
    parts = text.strip().split()
    if len(parts) < 3:
        return (0.0, 0.0, 0.0)
    return (float(parts[0]), float(parts[1]), float(parts[2]))
