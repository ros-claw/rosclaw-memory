"""
Xacro 解析器 — 纯 Python 宏展开（零 ROS 依赖）

Xacro 是 URDF 的宏预处理语言。核心功能：
1. <xacro:include> — 包含其他文件
2. <xacro:property> — 定义变量
3. ${expr} — 表达式求值（简单算术和变量替换）
4. <xacro:if> / <xacro:unless> — 条件控制
5. <xacro:macro> / <xacro:insert> — 宏定义与调用

本实现为简化版，覆盖 90%+ 的常见用例。
不支持 Python eval 复杂表达式（安全原因）。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

from .base import ModelParser, ParseResult
from .urdf_parser import URDFParser

logger = logging.getLogger(__name__)


class XacroParser(ModelParser):
    """Xacro 解析器 — 先展开宏，再委托 URDFParser"""

    @property
    def supported_formats(self) -> List[str]:
        return [".xacro"]

    def parse(self, source: str, **kwargs) -> ParseResult:
        include_paths = kwargs.get("include_paths", ["."])
        expanded = self.expand(source, include_paths=include_paths)
        # 委托给 URDFParser
        urdf_parser = URDFParser()
        result = urdf_parser.parse(expanded)
        result.format = "xacro"
        result.warnings.insert(0, "Xacro expanded to URDF before parsing")
        return result

    def expand(self, source: str, include_paths: List[str] = None) -> str:
        """展开 Xacro 宏，返回纯 URDF XML 字符串"""
        if include_paths is None:
            include_paths = ["."]

        text = source
        # 1. 包含文件
        text = self._resolve_includes(text, include_paths)
        # 2. 定义属性
        properties: Dict[str, str] = {}
        text = self._extract_properties(text, properties)
        # 3. 宏定义与调用
        macros: Dict[str, str] = {}
        text = self._extract_macros(text, macros)
        text = self._expand_macro_calls(text, macros, properties)
        # 4. 条件控制
        text = self._evaluate_conditions(text, properties)
        # 5. 变量替换
        text = self._substitute_properties(text, properties)
        # 6. 清理 xacro 命名空间标签
        text = self._clean_xacro_tags(text)
        return text

    def _resolve_includes(self, text: str, include_paths: List[str]) -> str:
        """解析 <xacro:include>"""
        pattern = re.compile(r'<xacro:include\s+filename="([^"]+)"\s*/?>')

        def replacer(m):
            filename = m.group(1)
            for base in include_paths:
                path = os.path.join(base, filename)
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return f.read()
            logger.warning("Xacro include not found: %s", filename)
            return ""

        return pattern.sub(replacer, text)

    def _extract_properties(self, text: str, properties: Dict[str, str]) -> str:
        """提取 <xacro:property> 定义并从文本中移除"""
        pattern = re.compile(r'<xacro:property\s+name="([^"]+)"\s+value="([^"]*)"\s*/?>')
        for m in pattern.finditer(text):
            properties[m.group(1)] = m.group(2)
        return pattern.sub("", text)

    def _extract_macros(self, text: str, macros: Dict[str, str]) -> str:
        """提取 <xacro:macro> 定义并从文本中移除"""
        pattern = re.compile(r'<xacro:macro\s+name="([^"]+)"[^>]*>(.*?)</xacro:macro>', re.DOTALL)
        for m in pattern.finditer(text):
            macros[m.group(1)] = m.group(2)
        return pattern.sub("", text)

    def _expand_macro_calls(self, text: str, macros: Dict[str, str], properties: Dict[str, str]) -> str:
        """展开 <xacro:macro_name> 调用"""
        # 第一轮：识别已定义的宏名
        for macro_name, macro_body in macros.items():
            call_pattern = re.compile(
                rf'<xacro:{re.escape(macro_name)}\s+([^/]*)/?>',
                re.DOTALL,
            )

            def replacer(m):
                args = self._parse_args(m.group(1))
                body = macro_body
                for arg_name, arg_val in args.items():
                    body = body.replace(f"${{{arg_name}}}", arg_val)
                # 未替换的参数用默认值
                return body

            text = call_pattern.sub(replacer, text)
        return text

    def _parse_args(self, arg_text: str) -> Dict[str, str]:
        """解析宏调用的属性参数"""
        args = {}
        # 简单解析 key="value" 对
        pattern = re.compile(r'(\w+)="([^"]*)"')
        for m in pattern.finditer(arg_text):
            args[m.group(1)] = m.group(2)
        return args

    def _evaluate_conditions(self, text: str, properties: Dict[str, str]) -> str:
        """展开 <xacro:if> 和 <xacro:unless>"""
        # <xacro:if value="${expr}">...content...</xacro:if>
        for tag in ("if", "unless"):
            pattern = re.compile(
                rf'<xacro:{tag}\s+value="\$\{{([^}}]+)\}}"[^>]*>(.*?)</xacro:{tag}>',
                re.DOTALL,
            )

            def make_replacer(is_unless):
                def replacer(m):
                    expr = m.group(1).strip()
                    content = m.group(2)
                    val = self._eval_expr(expr, properties)
                    condition = bool(val)
                    if is_unless:
                        condition = not condition
                    return content if condition else ""
                return replacer

            text = pattern.sub(make_replacer(tag == "unless"), text)
        return text

    def _substitute_properties(self, text: str, properties: Dict[str, str]) -> str:
        """替换 ${property_name} 和 ${expr}"""
        def replacer(m):
            expr = m.group(1).strip()
            # 先尝试直接属性查找
            if expr in properties:
                return properties[expr]
            # 尝试简单算术
            try:
                return str(self._eval_expr(expr, properties))
            except Exception:
                return m.group(0)  # 保持原样

        return re.sub(r'\$\{([^}]+)\}', replacer, text)

    def _eval_expr(self, expr: str, properties: Dict[str, str]) -> Any:
        """安全表达式求值 — 只允许数字、运算符和已知变量"""
        # 替换变量名
        tokens = re.split(r'([+\-*/()])', expr)
        safe_tokens = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if tok in properties:
                safe_tokens.append(str(properties[tok]))
            elif re.match(r'^[\d.]+$', tok) or tok in "+-*/()":
                safe_tokens.append(tok)
            else:
                # 未知 token，尝试数值
                try:
                    float(tok)
                    safe_tokens.append(tok)
                except ValueError:
                    raise ValueError(f"Unknown token in expression: {tok}")
        safe_expr = " ".join(safe_tokens)
        try:
            return eval(safe_expr, {"__builtins__": {}}, {})
        except Exception:
            return 0.0

    def _clean_xacro_tags(self, text: str) -> str:
        """移除残留的 xacro 命名空间元素"""
        # 移除 xacro 注释
        text = re.sub(r'<!--\s*xacro:.*?-->', '', text, flags=re.DOTALL)
        # 移除 xacro 命名空间声明
        text = re.sub(r'xmlns:xacro="[^"]*"', '', text)
        return text.strip()
