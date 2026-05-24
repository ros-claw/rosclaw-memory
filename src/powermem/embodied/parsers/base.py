"""
物理模型解析器基类 — 统一接口

所有解析器（URDF/MJCF/SDF/Xacro/USD）实现此接口，
输出标准化的 ParseResult。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..physical_model import RobotDynamics


@dataclass
class ParseResult:
    """解析结果 — 所有格式解析器的统一输出"""

    # 核心动力学模型
    dynamics: RobotDynamics = field(default_factory=RobotDynamics)

    # 原始格式元数据
    format: str = "unknown"
    source_path: Optional[str] = None
    source_hash: str = ""

    # 完整链路列表（含视觉/碰撞几何体）
    links: List[Dict[str, Any]] = field(default_factory=list)
    joints: List[Dict[str, Any]] = field(default_factory=list)

    # 网格文件引用（用于多模态感知对齐）
    mesh_paths: List[str] = field(default_factory=list)

    # 材质/颜色信息
    materials: List[Dict[str, Any]] = field(default_factory=list)

    # 世界/环境对象（非机器人）
    world_objects: List[Dict[str, Any]] = field(default_factory=list)

    # 解析警告
    warnings: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return len(self.dynamics.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dynamics": self.dynamics.to_dict(),
            "format": self.format,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "links": self.links,
            "joints": self.joints,
            "mesh_paths": self.mesh_paths,
            "materials": self.materials,
            "world_objects": self.world_objects,
            "warnings": self.warnings,
        }


class ModelParser(ABC):
    """物理模型解析器抽象基类"""

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """返回支持的文件扩展名列表，如 ['.urdf', '.xacro']"""
        ...

    @abstractmethod
    def parse(self, source: str, **kwargs) -> ParseResult:
        """解析物理模型

        Args:
            source: 文件路径或 XML 字符串内容
            **kwargs: 格式特定选项

        Returns:
            ParseResult
        """
        ...

    def parse_file(self, path: str, **kwargs) -> ParseResult:
        """从文件路径解析"""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        result = self.parse(content, **kwargs)
        result.source_path = path
        return result
