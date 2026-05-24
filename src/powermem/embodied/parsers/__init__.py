"""
ROSClaw-Memory 物理模型解析器包

支持格式：URDF, MJCF (MuJoCo), SDF (Gazebo), Xacro, OpenUSD

使用示例：
    from powermem.embodied.parsers import parse_model

    result = parse_model("robot.urdf")
    print(result.dynamics.joint_names)
    print(result.mesh_paths)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .base import ModelParser, ParseResult

logger = logging.getLogger(__name__)

# 延迟导入解析器（避免加载未使用的依赖）

_REGISTRY: Dict[str, type] = {}


def register_parser(fmt: str, parser_cls: type) -> None:
    """注册解析器"""
    _REGISTRY[fmt.lower()] = parser_cls
    logger.debug("Registered parser for format: %s", fmt)


def _ensure_loaded() -> None:
    """延迟加载所有解析器模块"""
    if _REGISTRY:
        return
    try:
        from .urdf_parser import URDFParser
        register_parser("urdf", URDFParser)
    except Exception as e:
        logger.warning("URDF parser not available: %s", e)

    try:
        from .mjcf_parser import MJCFParser
        register_parser("mjcf", MJCFParser)
    except Exception as e:
        logger.warning("MJCF parser not available: %s", e)

    try:
        from .sdf_parser import SDFParser
        register_parser("sdf", SDFParser)
    except Exception as e:
        logger.warning("SDF parser not available: %s", e)

    try:
        from .xacro_parser import XacroParser
        register_parser("xacro", XacroParser)
    except Exception as e:
        logger.warning("Xacro parser not available: %s", e)

    try:
        from .usd_parser import USDParser
        register_parser("usd", USDParser)
    except Exception as e:
        logger.warning("USD parser not available: %s", e)


def get_parser(fmt: str) -> Optional[ModelParser]:
    """获取指定格式的解析器实例"""
    _ensure_loaded()
    cls = _REGISTRY.get(fmt.lower())
    if cls is None:
        return None
    return cls()


def parse_model(source: str, fmt: Optional[str] = None, **kwargs) -> ParseResult:
    """自动检测格式并解析物理模型

    Args:
        source: 文件路径或内容字符串
        fmt: 显式指定格式（如 'urdf', 'mjcf'），None 时自动检测
        **kwargs: 传递给解析器的额外参数

    Returns:
        ParseResult

    Raises:
        ValueError: 格式不支持或自动检测失败
    """
    _ensure_loaded()

    # 判断 source 是路径还是内容
    is_path = os.path.isfile(source)
    content = source
    if is_path:
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
        if fmt is None:
            _, ext = os.path.splitext(source)
            fmt = ext.lstrip(".").lower()

    # 自动检测格式
    if fmt is None:
        fmt = _detect_format(content)

    parser = get_parser(fmt)
    if parser is None:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {list(_REGISTRY.keys())}")

    result = parser.parse(content, **kwargs)
    if is_path:
        result.source_path = source
    return result


def _detect_format(content: str) -> str:
    """基于 XML 根标签自动检测格式"""
    stripped = content.strip()

    # 非 XML 格式：USD
    if stripped.startswith("#usda") or stripped.startswith("#usdc"):
        return "usd"

    if not stripped.startswith("<"):
        raise ValueError("Cannot auto-detect format: content does not appear to be XML")

    lowered = stripped.lower()
    if "<mujoco" in lowered[:200]:
        return "mjcf"
    if "<sdf" in lowered[:200]:
        return "sdf"
    if "<robot" in lowered[:200]:
        return "urdf"
    if "<xacro:" in lowered[:500]:
        return "xacro"
    if any(tag in lowered[:200] for tag in ("<usda", "<usdc", "<usd")):
        return "usd"

    # 默认尝试 URDF（最常见）
    return "urdf"


__all__ = [
    "ModelParser",
    "ParseResult",
    "parse_model",
    "get_parser",
    "register_parser",
]
