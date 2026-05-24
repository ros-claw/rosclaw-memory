"""
Surprisal Gate — 预测编码惊奇门控

核心思想：传感器流中 90%+ 的数据是冗余的。
只有当实际观测与内部预测的误差超过动态阈值时，才产生记忆。

算法：
1. 维护每个传感器的滑动窗口统计（均值 μ、标准差 σ）
2. 动态阈值 = μ + k * σ（默认 k=3，即 3-sigma）
3. 预测误差 |observation - prediction| > 阈值 → 通过门控
4. 通过后更新滑动窗口（但仅当误差 < 5σ 时，排除极端 outlier 污染统计）
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .memory_atom import MemoryAtom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 滑动窗口统计
# ---------------------------------------------------------------------------

@dataclass
class _RunningStats:
    """Welford 在线算法 — O(1) 内存计算均值和方差"""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of differences from the mean

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / self.count)

    def to_dict(self) -> Dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> _RunningStats:
        return cls(count=int(d.get("count", 0)), mean=float(d.get("mean", 0.0)), m2=float(d.get("m2", 0.0)))


# ---------------------------------------------------------------------------
# 预测器接口
# ---------------------------------------------------------------------------

class Predictor:
    """预测器抽象 — 任何可预测下一个观测的模型"""

    def predict(self, predictor_id: str, history: List[Any]) -> Any:
        """基于历史预测下一个值"""
        raise NotImplementedError

    def prediction_error(self, prediction: Any, observation: Any) -> float:
        """计算预测值与观测值之间的标量误差"""
        raise NotImplementedError


class ZeroOrderHoldPredictor(Predictor):
    """零阶保持：预测 = 上一次观测（最简单的基线）"""

    def predict(self, predictor_id: str, history: List[Any]) -> Any:
        if not history:
            return 0.0
        return history[-1]

    def prediction_error(self, prediction: Any, observation: Any) -> float:
        try:
            return abs(float(prediction) - float(observation))
        except (TypeError, ValueError):
            # 向量情况：L2 范数
            if isinstance(prediction, (list, tuple)) and isinstance(observation, (list, tuple)):
                return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(prediction, observation)))
            return 1.0  # 无法比较时默认高误差


class LinearPredictor(Predictor):
    """线性预测：v_next = v_last + (v_last - v_{last-1})"""

    def predict(self, predictor_id: str, history: List[Any]) -> Any:
        if len(history) < 2:
            return history[-1] if history else 0.0
        try:
            return 2 * float(history[-1]) - float(history[-2])
        except (TypeError, ValueError):
            return history[-1]

    def prediction_error(self, prediction: Any, observation: Any) -> float:
        try:
            return abs(float(prediction) - float(observation))
        except (TypeError, ValueError):
            if isinstance(prediction, (list, tuple)) and isinstance(observation, (list, tuple)):
                return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(prediction, observation)))
            return 1.0


# ---------------------------------------------------------------------------
# Surprisal Gate
# ---------------------------------------------------------------------------

@dataclass
class SurprisalGate:
    """惊奇门控 — 物理 AI 的数据过滤器

    Args:
        k_sigma: 动态阈值系数（默认 3.0 = 3-sigma）
        max_sigma_multiplier: 超过此倍数的 outlier 不更新统计（默认 5.0）
        min_samples: 初始化滑动窗口所需的最小样本数（默认 10）
        predictor: 预测器实例（默认 ZeroOrderHoldPredictor）
        state_store: 可选的状态持久化回调，用于跨会话恢复
    """

    k_sigma: float = 3.0
    max_sigma_multiplier: float = 5.0
    min_samples: int = 10
    predictor: Predictor = field(default_factory=ZeroOrderHoldPredictor)
    state_store: Optional[Callable[[str, Dict[str, Any]], None]] = None
    state_load: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None

    # 内部状态：每个 predictor_id 的滑动窗口
    _stats: Dict[str, _RunningStats] = field(default_factory=dict)
    _history: Dict[str, List[Any]] = field(default_factory=dict)
    _last_predictions: Dict[str, Any] = field(default_factory=dict)

    def _get_stats(self, predictor_id: str) -> _RunningStats:
        if predictor_id not in self._stats:
            # 尝试从持久化存储恢复
            if self.state_load is not None:
                saved = self.state_load(predictor_id)
                if saved:
                    self._stats[predictor_id] = _RunningStats.from_dict(saved)
                    return self._stats[predictor_id]
            self._stats[predictor_id] = _RunningStats()
        return self._stats[predictor_id]

    def _save_stats(self, predictor_id: str) -> None:
        if self.state_store is not None and predictor_id in self._stats:
            self.state_store(predictor_id, self._stats[predictor_id].to_dict())

    def check(
        self,
        predictor_id: str,
        observation: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, float]:
        """检查观测是否通过惊奇门控

        Returns:
            (passed, prediction_error, threshold)
            passed: True = 应该产生记忆
            prediction_error: 实际预测误差
            threshold: 当前动态阈值
        """
        stats = self._get_stats(predictor_id)
        history = self._history.setdefault(predictor_id, [])

        # 1. 预测
        prediction = self.predictor.predict(predictor_id, history)
        self._last_predictions[predictor_id] = prediction

        # 2. 计算误差
        error = self.predictor.prediction_error(prediction, observation)

        # 3. 计算动态阈值
        threshold = stats.mean + self.k_sigma * stats.std() if stats.count >= self.min_samples else float("inf")

        # 4. 决策
        passed = error > threshold or stats.count < self.min_samples

        # 5. 更新统计（但排除极端 outlier 污染）
        outlier_limit = stats.mean + self.max_sigma_multiplier * stats.std() if stats.count >= 2 else float("inf")
        if error < outlier_limit or stats.count < self.min_samples:
            stats.update(error)
            self._save_stats(predictor_id)

        # 6. 更新历史
        history.append(observation)
        # 限制历史长度（防止内存无限增长）
        if len(history) > 1000:
            history[:] = history[-500:]

        if passed:
            logger.debug(
                "SurprisalGate[%s] PASSED: error=%.4f > threshold=%.4f (μ=%.4f, σ=%.4f)",
                predictor_id, error, threshold, stats.mean, stats.std(),
            )
        else:
            logger.debug(
                "SurprisalGate[%s] blocked: error=%.4f <= threshold=%.4f",
                predictor_id, error, threshold,
            )

        return passed, error, threshold

    def filter_atom(
        self,
        atom: MemoryAtom,
        observation_value: Any,
        predictor_id: Optional[str] = None,
    ) -> Optional[MemoryAtom]:
        """对 MemoryAtom 应用惊奇门控，通过则返回（带 prediction_error），否则返回 None"""
        pid = predictor_id or (atom.perceptual.modality.value if atom.perceptual else "default")
        passed, error, threshold = self.check(pid, observation_value)

        if passed:
            atom.prediction_error = error
            return atom
        return None

    def get_state(self, predictor_id: str) -> Dict[str, Any]:
        """获取指定预测器的状态（用于监控/调试）"""
        stats = self._get_stats(predictor_id)
        return {
            "predictor_id": predictor_id,
            "mean": stats.mean,
            "std": stats.std(),
            "count": stats.count,
            "k_sigma": self.k_sigma,
            "threshold": stats.mean + self.k_sigma * stats.std() if stats.count >= self.min_samples else None,
            "last_prediction": self._last_predictions.get(predictor_id),
            "history_length": len(self._history.get(predictor_id, [])),
        }

    def reset(self, predictor_id: Optional[str] = None) -> None:
        """重置指定或全部预测器状态"""
        if predictor_id is None:
            self._stats.clear()
            self._history.clear()
            self._last_predictions.clear()
        else:
            self._stats.pop(predictor_id, None)
            self._history.pop(predictor_id, None)
            self._last_predictions.pop(predictor_id, None)
