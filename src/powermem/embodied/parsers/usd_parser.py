"""
OpenUSD 解析器 — NVIDIA Omniverse / Pixar USD

USD 是一种场景描述格式，支持层、引用、变体等复杂特性。
本解析器提取机器人相关的关节、连杆、质量和几何信息。

注意：需要安装 usd-core（pxr）库：
    pip install usd-core

如果没有安装，解析器会返回警告并提供手动解析路径。
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from ..physical_model import DHParameter, JointLimit, RobotDynamics
from .base import ModelParser, ParseResult

logger = logging.getLogger(__name__)

# 尝试导入 USD 库
try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
    _HAS_USD = True
except ImportError:
    _HAS_USD = False
    logger.debug("usd-core (pxr) not installed; USDParser will use fallback")


class USDParser(ModelParser):
    """OpenUSD 解析器"""

    @property
    def supported_formats(self) -> List[str]:
        return [".usd", ".usda", ".usdc", ".usdz"]

    def parse(self, source: str, **kwargs) -> ParseResult:
        """解析 USD 内容

        Args:
            source: 文件路径（.usd/.usda/.usdc）或 ASCII 内容字符串
            **kwargs: stage_options 等 USD 特定选项
        """
        if not _HAS_USD:
            return self._fallback_parse(source)

        # 判断是路径还是内容
        is_path = source.strip().startswith("#usda") or source.strip().startswith("#usdc")
        if not is_path and len(source) < 500 and not source.strip().startswith("<"):
            # 很可能是文件路径
            return self._parse_file(source, **kwargs)

        # 内容字符串 — 写入临时文件再解析
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".usda", delete=False, mode="w", encoding="utf-8") as f:
            f.write(source)
            temp_path = f.name
        return self._parse_file(temp_path, **kwargs)

    def _parse_file(self, path: str, **kwargs) -> ParseResult:
        """从 USD 文件解析"""
        stage = Usd.Stage.Open(path)
        if not stage:
            return ParseResult(format="usd", warnings=[f"Failed to open USD stage: {path}"])

        result = ParseResult(format="usd")
        result.source_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]

        links_data: List[Dict[str, Any]] = []
        joints_data: List[Dict[str, Any]] = []
        joint_names: List[str] = []
        joint_limits: List[JointLimit] = []
        dh_params: List[DHParameter] = []
        link_masses: List[float] = []
        link_inertias: List[List[float]] = []

        # 遍历 Prim 树
        for prim in stage.Traverse():
            # --- 关节 (UsdPhysics.RevoluteJoint / PrismaticJoint / FixedJoint) ---
            joint_api = UsdPhysics.RevoluteJoint(prim) if prim.IsA(UsdPhysics.RevoluteJoint) else None
            if joint_api and joint_api.GetPrim().IsValid():
                jname = prim.GetName()
                joint_names.append(jname)

                # 轴
                axis = joint_api.GetAxisAttr().Get()
                axis_vec = (1.0, 0.0, 0.0)
                if axis == UsdGeom.Tokens.x:
                    axis_vec = (1.0, 0.0, 0.0)
                elif axis == UsdGeom.Tokens.y:
                    axis_vec = (0.0, 1.0, 0.0)
                elif axis == UsdGeom.Tokens.z:
                    axis_vec = (0.0, 0.0, 1.0)

                # 限制
                lower = -3.14159
                upper = 3.14159
                try:
                    limits = joint_api.GetLowerLimitAttr().Get()
                    if limits is not None:
                        lower = float(limits)
                    limits = joint_api.GetUpperLimitAttr().Get()
                    if limits is not None:
                        upper = float(limits)
                except Exception:
                    pass

                jl = JointLimit(min_rad=lower, max_rad=upper)
                joint_limits.append(jl)
                dh_params.append(DHParameter())

                # 父子关系
                parent_prim = joint_api.GetBody0Rel().GetTargets()
                child_prim = joint_api.GetBody1Rel().GetTargets()
                parent_name = str(parent_prim[0].name) if parent_prim else ""
                child_name = str(child_prim[0].name) if child_prim else ""

                joints_data.append({
                    "name": jname,
                    "type": "revolute",
                    "parent": parent_name,
                    "child": child_name,
                    "axis": axis_vec,
                    "limit": jl.to_dict(),
                })

            # --- 刚体 (UsdPhysics.RigidBody) ---
            rb = UsdPhysics.RigidBody(prim) if prim.IsA(UsdPhysics.RigidBody) else None
            if rb and rb.GetPrim().IsValid():
                lname = prim.GetName()
                mass = 1.0
                com = {"x": 0.0, "y": 0.0, "z": 0.0}
                inertia_flat = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]

                # 质量 API
                mass_api = UsdPhysics.MassAPI(prim)
                if mass_api and mass_api.GetPrim().IsValid():
                    m = mass_api.GetMassAttr().Get()
                    if m is not None:
                        mass = float(m)
                    diag = mass_api.GetDiagonalInertiaAttr().Get()
                    if diag is not None:
                        inertia_flat = [
                            float(diag[0]), 0.0, 0.0,
                            0.0, float(diag[1]), 0.0,
                            0.0, 0.0, float(diag[2]),
                        ]

                link_masses.append(mass)
                link_inertias.append(inertia_flat)

                # 变换（获取质心位置）
                xform = UsdGeom.Xformable(prim)
                if xform:
                    tf = xform.GetLocalTransformation()
                    trans = tf.ExtractTranslation()
                    com = {"x": trans[0], "y": trans[1], "z": trans[2]}

                # 几何体 / 网格
                mesh_paths = []
                for child in prim.GetChildren():
                    if child.IsA(UsdGeom.Mesh):
                        mesh = UsdGeom.Mesh(child)
                        # USD mesh 通常内嵌，没有外部文件路径
                        # 但可能有引用
                        refs = child.GetPrimStack()
                        for ref in refs:
                            if hasattr(ref, 'payload') and ref.payload:
                                mesh_paths.append(str(ref.payload.path))

                links_data.append({
                    "name": lname,
                    "mass": mass,
                    "inertia": inertia_flat,
                    "com": com,
                    "mesh_paths": mesh_paths,
                })
                result.mesh_paths.extend(mesh_paths)

            # --- 材质 (UsdShade.Material) ---
            if prim.IsA(UsdShade.Material):
                mat_info = {"name": prim.GetName()}
                # 尝试获取漫反射颜色
                shader = UsdShade.Shader(prim)
                if shader:
                    diffuse = shader.GetInput("diffuseColor")
                    if diffuse:
                        val = diffuse.Get()
                        if val:
                            mat_info["rgba"] = [float(val[i]) for i in range(4)] if len(val) >= 4 else [float(val[0]), float(val[1]), float(val[2]), 1.0]
                result.materials.append(mat_info)

        result.links = links_data
        result.joints = joints_data
        result.dynamics = RobotDynamics(
            joint_names=joint_names,
            dh_params=dh_params,
            joint_limits=joint_limits,
            link_masses=link_masses,
            link_inertias=link_inertias,
        )
        return result

    def _fallback_parse(self, source: str) -> ParseResult:
        """USD 库不可用时，尝试从 ASCII 内容做轻量级解析"""
        result = ParseResult(format="usd")
        result.warnings.append(
            "usd-core (pxr) not installed. "
            "Install with: pip install usd-core. "
            "Returning partial parse from ASCII content."
        )

        # 轻量级正则提取 joint/link 名称
        import re
        joint_names = re.findall(r'def PhysicsRevoluteJoint "([^"]+)"', source)
        body_names = re.findall(r'def PhysicsRigidBody "([^"]+)"', source)
        mesh_refs = re.findall(r'prepend references = @([^@]+)@', source)

        result.mesh_paths = mesh_refs
        result.dynamics = RobotDynamics(
            joint_names=joint_names,
            dh_params=[DHParameter() for _ in joint_names],
            joint_limits=[JointLimit() for _ in joint_names],
            link_masses=[1.0] * len(body_names),
            link_inertias=[[0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001] for _ in body_names],
        )
        return result
