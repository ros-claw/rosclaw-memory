"""
不确定性模型 — 融合、传播与决策

核心设计：
1. 区分 Aleatoric（不可减少）与 Epistemic（可通过探索减少）
2. 多传感器不确定性融合（协方差交集 / 加权平均）
3. 不确定性驱动的主动探索建议
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .types import UncertaintyEstimate, UncertaintyType, Vec3


# ---------------------------------------------------------------------------
# 单传感器不确定性融合
# ---------------------------------------------------------------------------

def fuse_uncertainties_ci(
    estimates: List[UncertaintyEstimate],
) -> UncertaintyEstimate:
    """协方差交集（Covariance Intersection）— 保守但一致的融合

    适用于任意相关性的多源估计融合，保证结果不比最差源差。
    """
    if not estimates:
        return UncertaintyEstimate()
    if len(estimates) == 1:
        return estimates[0]

    # 简化为标量版本的 CI：加权调和平均
    weights = [e.confidence for e in estimates]
    total_w = sum(weights) or 1.0

    fused_std = 0.0
    fused_confidence = 0.0
    for e, w in zip(estimates, weights):
        fused_std += w * e.std
        fused_confidence += w * e.confidence

    fused_std /= total_w
    fused_confidence /= total_w

    # 熵的保守估计：取最大（最不确定）
    max_entropy = max((1.0 - e.confidence) for e in estimates)
    fused_confidence = min(fused_confidence, 1.0 - max_entropy * 0.5)

    # 类型提升规则：只要有 epistemic，结果标记为 epistemic
    has_epistemic = any(e.type == UncertaintyType.EPISTEMIC for e in estimates)
    fused_type = UncertaintyType.EPISTEMIC if has_epistemic else estimates[0].type

    return UncertaintyEstimate(
        type=fused_type,
        std=fused_std,
        confidence=fused_confidence,
        sample_count=sum(e.sample_count for e in estimates),
    )


def fuse_uncertainties_kalman(
    estimates: List[UncertaintyEstimate],
) -> UncertaintyEstimate:
    """卡尔曼融合 — 假设各源独立，结果更乐观

    仅用于确认各传感器噪声独立的情况。
    """
    if not estimates:
        return UncertaintyEstimate()

    # 信息滤波器形式：方差倒数之和
    info_sum = sum(1.0 / max(e.std ** 2, 1e-12) for e in estimates)
    fused_var = 1.0 / max(info_sum, 1e-12)
    fused_std = math.sqrt(fused_var)

    # 置信度采用几何平均（独立事件联合概率）
    log_conf = sum(math.log(max(e.confidence, 1e-12)) for e in estimates)
    fused_confidence = math.exp(log_conf / len(estimates))

    return UncertaintyEstimate(
        type=UncertaintyType.ALEATORIC,
        std=fused_std,
        confidence=min(fused_confidence, 1.0),
        sample_count=sum(e.sample_count for e in estimates),
    )


# ---------------------------------------------------------------------------
# 空间不确定性传播
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SpatialUncertainty:
    """3D 空间位置的不确定性椭圆"""
    position: Vec3
    # 协方差矩阵（6 元素：xx, xy, xz, yy, yz, zz）
    covariance: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def std_xyz(self) -> Vec3:
        c = self.covariance
        return Vec3(
            x=math.sqrt(max(c[0], 0.0)),
            y=math.sqrt(max(c[3], 0.0)),
            z=math.sqrt(max(c[5], 0.0)),
        )

    def volume_3sigma(self) -> float:
        """3-sigma 置信椭球体积（近似）"""
        s = self.std_xyz()
        return (4.0 / 3.0) * math.pi * (3 * s.x) * (3 * s.y) * (3 * s.z)


def propagate_spatial_uncertainty(
    base: SpatialUncertainty,
    motion: Vec3,
    motion_std: Vec3,
) -> SpatialUncertainty:
    """简单平移不确定性传播：新位置 = 原位置 + 运动

    假设运动误差与原位置误差独立。
    """
    new_pos = Vec3(
        base.position.x + motion.x,
        base.position.y + motion.y,
        base.position.z + motion.z,
    )
    c = list(base.covariance)
    # 对角线方差相加
    c[0] += motion_std.x ** 2
    c[3] += motion_std.y ** 2
    c[5] += motion_std.z ** 2
    return SpatialUncertainty(position=new_pos, covariance=tuple(c))


# ---------------------------------------------------------------------------
# 探索决策
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExplorationSuggestion:
    """主动探索建议"""
    target_position: Optional[Vec3]
    target_modality: str
    reason: str
    expected_information_gain: float  # 预期信息增益（比特或归一化分数）
    priority: float  # [0, 1]


def suggest_exploration(
    observations: List[Tuple[Vec3, UncertaintyEstimate, str]],
    current_position: Vec3,
    max_range: float = 5.0,
) -> List[ExplorationSuggestion]:
    """基于观测不确定性生成探索建议

    Args:
        observations: [(position, uncertainty, modality), ...]
        current_position: 当前机器人位置
        max_range: 只考虑在此距离内的观测

    Returns:
        按 priority 降序排列的探索建议列表
    """
    suggestions: List[ExplorationSuggestion] = []

    for pos, unc, modality in observations:
        dist = pos.distance_to(current_position)
        if dist > max_range:
            continue

        # Epistemic 不确定性越高，探索价值越大
        if unc.type != UncertaintyType.EPISTEMIC:
            continue

        # 信息增益 ∝ uncertainty * (1 / distance) — 近处的未知更重要
        ig = (1.0 - unc.confidence) * (1.0 / (1.0 + dist))

        if ig > 0.1:  # 阈值过滤
            suggestions.append(
                ExplorationSuggestion(
                    target_position=pos,
                    target_modality=modality,
                    reason=f"high epistemic uncertainty ({1.0 - unc.confidence:.2f}) at {dist:.1f}m",
                    expected_information_gain=ig,
                    priority=min(1.0, ig),
                )
            )

    suggestions.sort(key=lambda s: s.priority, reverse=True)
    return suggestions
