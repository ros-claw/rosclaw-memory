"""
EmbodiedIntelligencePlugin — 具身智能记忆生命周期插件

继承 PowerMem 的 IntelligentMemoryPlugin，在 on_add / on_get / on_search 钩子中
注入具身智能特有的处理逻辑：
1. on_add: 惊奇门控过滤、显著性评估、不确定性校验、空间体素键生成
2. on_get: 预测编码更新、情感标记衰减、物理一致性检查
3. on_search: 时空相关性重排序、不确定性加权
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from powermem.intelligence.plugin import IntelligentMemoryPlugin
from powermem.utils.utils import get_current_datetime

from .memory_atom import MemoryAtom
from .surprisal_gate import SurprisalGate
from .types import AffectiveTag, MemoryAction, Modality, UncertaintyEstimate

logger = logging.getLogger(__name__)


class EmbodiedIntelligencePlugin(IntelligentMemoryPlugin):
    """具身智能插件 — ROSClaw-Memory 的核心生命周期控制器

    配置项（通过 config dict 传入）：
        enabled: bool — 是否启用
        surprisal_k_sigma: float — 惊奇门控阈值系数（默认 3.0）
        surprisal_min_samples: int — 门控初始化样本数（默认 10）
        salience_threshold_high: float — 高显著性阈值（默认 0.8）
        salience_decay_rate: float — 显著性衰减率（默认 0.01/访问）
        uncertainty_check: bool — 是否校验不确定性（默认 True）
        auto_voxel_key: bool — 是否自动生成体素键（默认 True）
        voxel_size: float — 体素大小米（默认 0.1）
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._surprisal: Optional[SurprisalGate] = None
        self._init_surprisal()

    def _init_surprisal(self) -> None:
        if not self.enabled:
            return
        try:
            self._surprisal = SurprisalGate(
                k_sigma=float(self.config.get("surprisal_k_sigma", 3.0)),
                min_samples=int(self.config.get("surprisal_min_samples", 10)),
            )
        except Exception as e:
            logger.warning("Failed to init SurprisalGate: %s", e)

    # -----------------------------------------------------------------------
    # on_add: 记忆写入前处理
    # -----------------------------------------------------------------------

    def on_add(self, *, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        metadata = metadata or {}

        # 如果不是具身记忆，交给父类处理（不干预纯文本记忆）
        if "rosclaw_version" not in metadata:
            return {}

        try:
            # 1. 从 metadata 还原 MemoryAtom（轻量级，不验证完整字段）
            atom = MemoryAtom.from_metadata(content, metadata)

            # 2. 不确定性校验：Aleatoric 必须存在，Epistemic 建议存在
            if self.config.get("uncertainty_check", True):
                if atom.uncertainty is None and atom.perceptual is not None:
                    # 为感知记忆补默认不确定性
                    atom.uncertainty = UncertaintyEstimate(
                        type=UncertaintyEstimate.ALEATORIC,
                        std=0.1,
                        confidence=0.9,
                    )
                    logger.debug("Auto-assigned default uncertainty for perceptual memory")

            # 3. 显著性快速评估（如果没有人工标记）
            if atom.affective is None and atom.perceptual is not None:
                salience = self._estimate_salience(atom)
                atom.affective = AffectiveTag(salience=salience, trigger="auto_eval")

            # 4. 自动生成体素键
            if self.config.get("auto_voxel_key", True) and atom.spatial is not None:
                voxel_size = float(self.config.get("voxel_size", 0.1))
                atom.compute_voxel_key(voxel_size)

            # 5. 将处理后的具身字段写回 metadata
            enhanced_meta = atom.to_metadata()

            # 6. 附加系统级 embodied 标记
            enhanced_meta["embodied_processing"] = {
                "plugin_version": "0.1.0",
                "processed_at": get_current_datetime(),
                "surprisal_enabled": self._surprisal is not None,
            }

            return {"metadata": enhanced_meta}

        except Exception as e:
            logger.warning("EmbodiedIntelligencePlugin.on_add failed: %s", e)
            return {}

    def _estimate_salience(self, atom: MemoryAtom) -> float:
        """快速显著性估计 — 无需 LLM，基于启发式规则"""
        score = 0.5

        # 高不确定性 = 显著（需要关注）
        if atom.uncertainty:
            if atom.uncertainty.confidence < 0.3:
                score += 0.2
            if atom.uncertainty.type == UncertaintyEstimate.EPISTEMIC:
                score += 0.1

        # 大预测误差 = 显著（惊奇事件）
        score += min(0.3, abs(atom.prediction_error))

        # 动作类型权重
        action_weights = {
            MemoryAction.ACT: 0.1,
            MemoryAction.CORRECT: 0.15,
            MemoryAction.REFLECT: 0.1,
        }
        score += action_weights.get(atom.action, 0.0)

        # 碰撞/力矩传感器 = 高显著
        if atom.perceptual and atom.perceptual.modality in (
            Modality.FORCE_TORQUE,
            Modality.TACTILE,
        ):
            score += 0.15

        return min(1.0, score)

    # -----------------------------------------------------------------------
    # on_get: 单条记忆访问处理
    # -----------------------------------------------------------------------

    def on_get(self, memory: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool]:
        if not self.enabled:
            return None, False

        meta = memory.get("metadata") or {}
        if "rosclaw_version" not in meta:
            # 非具身记忆，只更新基础访问计数
            return None, False

        try:
            updates: Dict[str, Any] = {}
            meta_updates: Dict[str, Any] = {}

            # 1. 显著性衰减（类似情绪消退）
            affective = meta.get("affective")
            if affective:
                decay_rate = float(self.config.get("salience_decay_rate", 0.01))
                old_salience = float(affective.get("salience", 0.5))
                new_salience = max(0.0, old_salience - decay_rate)
                if new_salience != old_salience:
                    affective = dict(affective)
                    affective["salience"] = round(new_salience, 4)
                    meta_updates["affective"] = affective

            # 2. 预测编码状态更新（如果记忆有 prediction_error）
            pe = float(meta.get("prediction_error", 0.0))
            if pe > 0 and self._surprisal is not None:
                modality = meta.get("perceptual", {}).get("modality", "rgb")
                self._surprisal.check(modality, pe)

            # 3. 物理一致性检查（标记可能过期的物理约束）
            physical = meta.get("physical")
            if physical and physical.get("entity_id"):
                # 如果记忆超过一定年龄且是物理不变量，标记为"需验证"
                created_at = memory.get("created_at", "")
                if created_at and self._is_stale(created_at, hours=24):
                    meta_updates["physical_stale"] = True

            if meta_updates:
                updates["metadata"] = {**meta, **meta_updates}
                updates["updated_at"] = get_current_datetime()

            return updates, False

        except Exception as e:
            logger.warning("EmbodiedIntelligencePlugin.on_get failed: %s", e)
            return None, False

    def _is_stale(self, created_at: str, hours: int = 24) -> bool:
        """简单判断记忆是否过期（可扩展为更复杂策略）"""
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - dt
            return age.total_seconds() > hours * 3600
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # on_search: 搜索结果增强
    # -----------------------------------------------------------------------

    def on_search(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[str]]:
        if not self.enabled:
            return [], []

        updates: List[Tuple[str, Dict[str, Any]]] = []
        deletes: List[str] = []

        for item in results:
            try:
                mem_id = item.get("id") or item.get("memory_id")
                if not mem_id:
                    continue

                meta = item.get("metadata") or {}
                if "rosclaw_version" not in meta:
                    continue

                # 1. 调用 on_get 处理生命周期
                upd, delete_flag = self.on_get(item)
                if delete_flag:
                    deletes.append(str(mem_id))
                    continue

                # 2. 时空相关性加权：根据查询上下文调整分数
                # （此处预留接口，实际加权在 EmbodiedMemory.search 中做 RRF 融合）
                if upd is None:
                    upd = {}

                # 标记最后搜索时间
                meta_upd = upd.get("metadata", {})
                meta_upd["last_searched_at"] = get_current_datetime()
                upd["metadata"] = meta_upd

                updates.append((str(mem_id), upd))

            except Exception as e:
                logger.warning("EmbodiedIntelligencePlugin.on_search item failed: %s", e)
                continue

        return updates, deletes

    # -----------------------------------------------------------------------
    # 状态监控
    # -----------------------------------------------------------------------

    def get_surprisal_state(self, predictor_id: Optional[str] = None) -> Dict[str, Any]:
        """获取惊奇门控状态（用于调试和监控）"""
        if self._surprisal is None:
            return {"enabled": False}
        if predictor_id:
            return self._surprisal.get_state(predictor_id)
        return {"enabled": True, "predictor_count": len(self._surprisal._stats)}
