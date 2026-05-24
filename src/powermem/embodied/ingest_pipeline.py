"""
IngestPipeline — 传感器数据接入管线

职责：
1. 接收原始传感器数据（图像、点云、关节角、力矩等）
2. 特征提取（将原始数据转为 feature_vec，避免存储像素）
3. 通过 Surprisal Gate 过滤冗余
4. 包装为 MemoryAtom 并送入 PowerMem

设计原则：
- 零 ROS 依赖：输入是原始 numpy array 或 Python 原生类型
- 可插拔：特征提取器通过注册表动态加载
- 不阻塞：批量缓冲 + 异步 flush
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .memory_atom import MemoryAtom
from .surprisal_gate import Predictor, SurprisalGate, ZeroOrderHoldPredictor
from .types import (
    AffectiveTag,
    MemoryAction,
    Modality,
    PerceptualSnapshot,
    Pose,
    UncertaintyEstimate,
    Vec3,
    to_jsonable,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 特征提取器注册表
# ---------------------------------------------------------------------------

FeatureExtractor = Callable[[Any, Modality], Tuple[Tuple[float, ...], Dict[str, Any]]]

_FEATURE_EXTRACTORS: Dict[str, FeatureExtractor] = {}


def register_feature_extractor(name: str, fn: FeatureExtractor) -> None:
    """注册一个特征提取器"""
    _FEATURE_EXTRACTORS[name] = fn
    logger.info("Registered feature extractor: %s", name)


def get_feature_extractor(name: str) -> Optional[FeatureExtractor]:
    return _FEATURE_EXTRACTORS.get(name)


# 默认特征提取器：简单统计特征（均值、标准差、分位数）
def _default_feature_extractor(data: Any, modality: Modality) -> Tuple[Tuple[float, ...], Dict[str, Any]]:
    """默认特征提取 — 将任意数组转为统计特征向量"""
    import numpy as np

    try:
        arr = np.asarray(data, dtype=np.float32).flatten()
        if arr.size == 0:
            return ((), {"error": "empty array"})

        features = [
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.median(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
        ]
        # 如果数据量大，加一些频域/形状特征
        if arr.size >= 16:
            features.append(float(np.percentile(arr, 10)))
            features.append(float(np.percentile(arr, 90)))
            features.append(float(np.sqrt(np.mean(arr ** 2))))  # RMS

        meta = {"feature_count": len(features), "source_shape": getattr(data, "shape", None)}
        return (tuple(features), meta)
    except Exception as e:
        logger.warning("Default feature extraction failed: %s", e)
        return ((), {"error": str(e)})


register_feature_extractor("default", _default_feature_extractor)


# ---------------------------------------------------------------------------
# 传感器帧定义
# ---------------------------------------------------------------------------

@dataclass
class SensorFrame:
    """传感器原始帧"""
    modality: Modality
    timestamp_sec: float
    data: Any  # 原始数据（numpy array, list, dict 等）
    sensor_pose: Pose = field(default_factory=Pose)
    uncertainty: UncertaintyEstimate = field(default_factory=UncertaintyEstimate)
    frame_id: str = "world"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IngestPipeline
# ---------------------------------------------------------------------------

@dataclass
class IngestPipeline:
    """传感器数据接入管线

    Args:
        memory_store: 存储回调，接收 MemoryAtom 并返回 memory_id
        surprisal_gate: 惊奇门控实例（可选）
        feature_extractor_name: 特征提取器名称（默认 "default"）
        buffer_size: 批量缓冲大小（默认 100）
        flush_interval_sec: 自动 flush 间隔（默认 1.0）
    """

    memory_store: Callable[[MemoryAtom], int]
    surprisal_gate: Optional[SurprisalGate] = None
    feature_extractor_name: str = "default"
    buffer_size: int = 100
    flush_interval_sec: float = 1.0

    _buffer: Deque[MemoryAtom] = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    _last_flush_time: float = field(default_factory=time.time, repr=False)

    def __post_init__(self):
        if self.surprisal_gate is None:
            self.surprisal_gate = SurprisalGate(predictor=ZeroOrderHoldPredictor())

    def ingest(self, frame: SensorFrame, content: Optional[str] = None) -> Optional[int]:
        """单帧摄入 — 特征提取 → 惊奇门控 → 缓冲/存储

        Returns:
            memory_id if stored, None if filtered by surprisal gate
        """
        # 1. 特征提取
        extractor = get_feature_extractor(self.feature_extractor_name) or _default_feature_extractor
        feature_vec, feat_meta = extractor(frame.data, frame.modality)

        # 2. 构建 PerceptualSnapshot
        perceptual = PerceptualSnapshot(
            modality=frame.modality,
            feature_vec=feature_vec,
            sensor_pose=frame.sensor_pose,
            uncertainty=frame.uncertainty,
            sensor_meta={
                "frame_id": frame.frame_id,
                "feature_meta": feat_meta,
                **frame.metadata,
            },
        )

        # 3. 构建 MemoryAtom
        atom = MemoryAtom(
            content=content or f"{frame.modality.value}_observation_at_{frame.timestamp_sec:.3f}",
            spatial=frame.sensor_pose.position if frame.sensor_pose else None,
            temporal=None,  # 单帧瞬时记忆，时间区间在 flush 时扩展
            perceptual=perceptual,
            uncertainty=frame.uncertainty,
            action=MemoryAction.OBSERVE,
        )

        # 4. 惊奇门控
        if self.surprisal_gate is not None:
            predictor_id = f"{frame.modality.value}_{frame.frame_id}"
            # 使用 feature_vec 的第一维作为观测值（简化）
            observation_value = feature_vec[0] if feature_vec else 0.0
            passed_atom = self.surprisal_gate.filter_atom(atom, observation_value, predictor_id)
            if passed_atom is None:
                logger.debug("SurprisalGate blocked %s frame at t=%.3f", frame.modality.value, frame.timestamp_sec)
                return None
            atom = passed_atom

        # 5. 缓冲或立即存储
        self._buffer.append(atom)

        # 检查是否需要 flush
        now = time.time()
        should_flush = (
            len(self._buffer) >= self.buffer_size
            or (now - self._last_flush_time) >= self.flush_interval_sec
        )

        if should_flush:
            return self.flush()

        # 单条模式下返回缓冲中的 atom（未分配 memory_id）
        return None

    def flush(self) -> Optional[int]:
        """将缓冲区的记忆批量写入存储

        策略：
        - 对连续同模态、同位置的观测做时间区间合并
        - 返回最后一条的 memory_id
        """
        if not self._buffer:
            return None

        # 简单合并：按 (modality, frame_id) 分组，合并时间区间
        merged_groups: Dict[str, List[MemoryAtom]] = {}
        for atom in self._buffer:
            key = "default"
            if atom.perceptual:
                frame_id = atom.perceptual.sensor_meta.get("frame_id", "world")
                key = f"{atom.perceptual.modality.value}_{frame_id}"
            merged_groups.setdefault(key, []).append(atom)

        last_id: Optional[int] = None
        for group in merged_groups.values():
            if not group:
                continue

            # 合并为一条记忆（时间区间扩展）
            merged = self._merge_group(group)
            try:
                memory_id = self.memory_store(merged)
                last_id = memory_id
                logger.debug("Stored merged memory id=%s (%d atoms)", memory_id, len(group))
            except Exception as e:
                logger.warning("Failed to store merged memory: %s", e)

        self._buffer.clear()
        self._last_flush_time = time.time()
        return last_id

    def _merge_group(self, atoms: List[MemoryAtom]) -> MemoryAtom:
        """合并一组原子为单条记忆"""
        if len(atoms) == 1:
            return atoms[0]

        # 以第一个为基底
        base = atoms[0]

        # 扩展时间区间
        from .types import TemporalInterval
        timestamps = []
        for a in atoms:
            if a.temporal:
                timestamps.extend([a.temporal.start_sec, a.temporal.end_sec])
            elif a.perceptual and a.perceptual.sensor_meta.get("timestamp_sec"):
                timestamps.append(a.perceptual.sensor_meta["timestamp_sec"])

        if timestamps:
            base.temporal = TemporalInterval(min(timestamps), max(timestamps))

        # 融合不确定性
        from .uncertainty import fuse_uncertainties_ci
        uncertainties = [a.uncertainty for a in atoms if a.uncertainty]
        if uncertainties:
            base.uncertainty = fuse_uncertainties_ci(uncertainties)

        # 更新内容描述
        base.content = f"{base.content} [merged {len(atoms)} frames]"

        return base

    def get_stats(self) -> Dict[str, Any]:
        """获取管线统计信息"""
        return {
            "buffer_size": len(self._buffer),
            "buffer_capacity": self.buffer_size,
            "last_flush_sec": self._last_flush_time,
            "surprisal_state": (
                {k: self.surprisal_gate.get_state(k) for k in self.surprisal_gate._stats}
                if self.surprisal_gate else None
            ),
        }

    def reset(self) -> None:
        """清空缓冲区和门控状态"""
        self._buffer.clear()
        if self.surprisal_gate:
            self.surprisal_gate.reset()
