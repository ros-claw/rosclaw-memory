"""
Object Permanence — 对象恒存引擎

Physical AI 的核心能力：被遮挡的对象不会消失。
当传感器未检测到一个已知对象时，系统应：
1. 标记为 occluded（遮挡）
2. 按时间衰减其存在置信度
3. 置信度低于阈值后标记为 missing（消失）
4. 重新检测到时恢复为 visible

与 Graphiti 的 Temporal Knowledge Graph 不同，
Object Permanence 处理的是物理世界中的"可见性状态机"，
而非语义事实的时间有效性。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .types import Vec3, WorldObject

logger = logging.getLogger(__name__)


@dataclass
class PermanenceReport:
    """对象恒存同步报告"""

    transitions: List[str] = field(default_factory=list)
    visible: List[str] = field(default_factory=list)
    occluded: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transitions": self.transitions,
            "visible": self.visible,
            "occluded": self.occluded,
            "missing": self.missing,
            "added": self.added,
        }


class ObjectPermanenceTracker:
    """对象恒存追踪器

    将当前帧的感知检测结果与持久化世界对象同步，
    自动管理遮挡状态转移和置信度衰减。

    运动预测：维护每个对象的速度历史（滑动窗口），
    在遮挡期间使用线性外推预测位置，辅助重检测匹配。
    """

    def __init__(
        self,
        world_object_store: Any,
        decay_rate: float = 0.05,
        missing_threshold: float = 0.2,
        velocity_window: int = 5,
        enable_prediction: bool = True,
    ):
        """
        Args:
            world_object_store: WorldObjectStore 实例
            decay_rate: 每秒置信度衰减率（默认 5%/秒）
            missing_threshold: 置信度低于此值时标记为 missing
            velocity_window: 速度滑动窗口大小
            enable_prediction: 是否启用遮挡期间运动预测
        """
        self.store = world_object_store
        self.decay_rate = decay_rate
        self.missing_threshold = missing_threshold
        self.velocity_window = velocity_window
        self.enable_prediction = enable_prediction

        # 运动状态：obj_id -> {velocity: Vec3, history: List[Vec3], last_update_sec: float}
        self._motion_state: Dict[str, Dict[str, Any]] = {}
        # 预测位置：obj_id -> Vec3（仅遮挡期间有效）
        self._predicted_positions: Dict[str, Vec3] = {}

    def sync_detections(
        self,
        scene_id: str,
        detections: List[WorldObject],
        timestamp_sec: float,
        occlusion_radius: float = 0.5,
    ) -> PermanenceReport:
        """将当前帧检测结果与已有世界对象同步

        策略：
        1. 加载该场景所有已有对象
        2. 按 obj_id 匹配检测结果 → visible，更新 pose
        3. 未匹配到的检测结果 → 按空间+类型匹配最近对象
        4. 仍未匹配的检测结果 → 作为新对象添加
        5. 已有但未被匹配的对象 → 衰减 confidence
        6. confidence < threshold → missing

        Args:
            scene_id: 场景 ID
            detections: 当前帧检测到的对象列表
            timestamp_sec: 当前时间戳（秒）
            occlusion_radius: 空间重检测匹配半径（米）

        Returns:
            PermanenceReport
        """
        report = PermanenceReport()
        existing = self.store.list_by_scene(scene_id, limit=1000)
        existing_by_id = {obj.obj_id: obj for obj in existing}

        matched_existing_ids: set = set()
        matched_detection_indices: set = set()

        # Step 1: 按 obj_id 精确匹配
        for i, det in enumerate(detections):
            if det.obj_id in existing_by_id:
                existing_obj = existing_by_id[det.obj_id]
                self._confirm_visible(existing_obj, det, timestamp_sec)
                report.visible.append(det.obj_id)
                matched_existing_ids.add(det.obj_id)
                matched_detection_indices.add(i)
                if existing_obj.occlusion_status != "visible":
                    report.transitions.append(
                        f"{det.obj_id}: {existing_obj.occlusion_status} -> visible"
                    )

        # Step 2: 未精确匹配的检测结果 → 按空间+类型匹配
        unmatched_detections = [
            (i, det) for i, det in enumerate(detections)
            if i not in matched_detection_indices
        ]

        for i, det in unmatched_detections:
            match_id = self._find_spatial_match(
                det, existing, matched_existing_ids, occlusion_radius
            )
            if match_id:
                existing_obj = existing_by_id[match_id]
                self._confirm_visible(existing_obj, det, timestamp_sec)
                report.visible.append(match_id)
                matched_existing_ids.add(match_id)
                matched_detection_indices.add(i)
                if existing_obj.occlusion_status != "visible":
                    report.transitions.append(
                        f"{match_id}: {existing_obj.occlusion_status} -> visible"
                    )
            else:
                # Step 3: 新对象
                new_obj = WorldObject(
                    obj_id=det.obj_id or self._generate_obj_id(det, scene_id),
                    obj_type=det.obj_type,
                    name=det.name,
                    pose=det.pose,
                    size=det.size,
                    color=det.color,
                    mesh_path=det.mesh_path,
                    physics_props=det.physics_props,
                    semantic_tags=det.semantic_tags,
                    scene_id=scene_id,
                    parent_obj_id=det.parent_obj_id,
                    state="present",
                    occlusion_status="visible",
                    last_confirmed_position=det.pose.position,
                    confidence=1.0,
                    last_seen_sec=timestamp_sec,
                )
                self.store.save(new_obj)
                report.added.append(new_obj.obj_id)
                matched_detection_indices.add(i)

        # Step 4: 已有但未被匹配的对象 → 衰减 confidence，更新预测位置
        for obj_id, obj in existing_by_id.items():
            if obj_id in matched_existing_ids:
                continue
            new_confidence = self._decay_confidence(obj, timestamp_sec)
            new_status = obj.occlusion_status

            if new_confidence < self.missing_threshold:
                new_status = "missing"
                report.missing.append(obj_id)
                report.transitions.append(
                    f"{obj_id}: {obj.occlusion_status} ({new_confidence:.2f}) -> missing"
                )
                # 清除运动状态
                self._motion_state.pop(obj_id, None)
                self._predicted_positions.pop(obj_id, None)
            elif new_status == "visible":
                new_status = "occluded"
                report.occluded.append(obj_id)
                report.transitions.append(
                    f"{obj_id}: visible -> occluded ({new_confidence:.2f})"
                )
                # 初始化遮挡期间的预测
                if self.enable_prediction:
                    self._predict_position(obj_id, timestamp_sec)
            else:
                # 保持 occluded 状态，confidence 继续衰减，更新预测位置
                report.occluded.append(obj_id)
                if self.enable_prediction:
                    self._predict_position(obj_id, timestamp_sec)

            self.store.update_occlusion(
                obj_id,
                occlusion_status=new_status,
                confidence=new_confidence,
                last_seen_sec=obj.last_seen_sec,  # 不更新时间，保持最后看见的时间
            )

        logger.info(
            "ObjectPermanence: scene=%s visible=%d occluded=%d missing=%d added=%d",
            scene_id,
            len(report.visible),
            len(report.occluded),
            len(report.missing),
            len(report.added),
        )
        return report

    def _confirm_visible(
        self,
        existing: WorldObject,
        detection: WorldObject,
        timestamp_sec: float,
    ) -> None:
        """确认对象可见，更新位姿、遮挡状态和速度历史"""
        if self.enable_prediction:
            self._update_velocity(existing, detection, timestamp_sec)
            # 清除预测位置（对象已重新可见）
            self._predicted_positions.pop(existing.obj_id, None)

        self.store.update_pose(
            existing.obj_id,
            detection.pose,
            state=detection.state if detection.state != existing.state else None,
        )
        # update_pose 已经重置了 occlusion_status='visible' 和 confidence=1.0
        # 但需要更新 last_seen_sec
        self.store.update_occlusion(
            existing.obj_id,
            occlusion_status="visible",
            confidence=1.0,
            last_seen_sec=timestamp_sec,
        )

    def _update_velocity(
        self,
        existing: WorldObject,
        detection: WorldObject,
        timestamp_sec: float,
    ) -> None:
        """更新对象速度历史（滑动窗口平均）"""
        obj_id = existing.obj_id
        old_state = self._motion_state.get(obj_id)
        if old_state is None:
            self._motion_state[obj_id] = {
                "velocity": Vec3(0, 0, 0),
                "history": [],
                "last_update_sec": timestamp_sec,
            }
            return

        dt = timestamp_sec - old_state["last_update_sec"]
        if dt <= 0:
            return

        old_pos = existing.pose.position
        new_pos = detection.pose.position
        vx = (new_pos.x - old_pos.x) / dt
        vy = (new_pos.y - old_pos.y) / dt
        vz = (new_pos.z - old_pos.z) / dt
        instant_vel = Vec3(vx, vy, vz)

        history: List[Vec3] = old_state["history"]
        history.append(instant_vel)
        if len(history) > self.velocity_window:
            history.pop(0)

        # 滑动窗口平均速度
        avg_vel = Vec3(
            sum(v.x for v in history) / len(history),
            sum(v.y for v in history) / len(history),
            sum(v.z for v in history) / len(history),
        )
        old_state["velocity"] = avg_vel
        old_state["last_update_sec"] = timestamp_sec

    def _predict_position(self, obj_id: str, timestamp_sec: float) -> Optional[Vec3]:
        """基于速度历史预测对象当前位置（遮挡期间）"""
        if not self.enable_prediction:
            return None
        state = self._motion_state.get(obj_id)
        if state is None:
            return None
        vel = state["velocity"]
        if vel.x == 0 and vel.y == 0 and vel.z == 0:
            return None
        last_update = state["last_update_sec"]
        dt = timestamp_sec - last_update
        if dt <= 0:
            return None
        # 获取当前存储的位置（可能是上一次预测后的位置或最后确认位置）
        obj = self.store.load(obj_id)
        if obj is None:
            return None
        base_pos = self._predicted_positions.get(obj_id, obj.pose.position)
        pred = Vec3(
            base_pos.x + vel.x * dt,
            base_pos.y + vel.y * dt,
            base_pos.z + vel.z * dt,
        )
        self._predicted_positions[obj_id] = pred
        state["last_update_sec"] = timestamp_sec
        return pred

    def _find_spatial_match(
        self,
        detection: WorldObject,
        existing_objects: List[WorldObject],
        exclude_ids: set,
        radius: float,
    ) -> Optional[str]:
        """在半径内按类型匹配最近的对象

        对遮挡对象使用预测位置而非最后确认位置，
        提高运动对象在重检测时的匹配成功率。
        """
        best_match: Optional[str] = None
        best_dist = float("inf")

        det_pos = detection.pose.position
        for obj in existing_objects:
            if obj.obj_id in exclude_ids:
                continue
            if obj.obj_type != detection.obj_type:
                continue
            # 忽略 missing 状态的对象（已经认为消失了）
            if obj.occlusion_status == "missing":
                continue
            # 对遮挡对象优先使用预测位置
            if obj.occlusion_status == "occluded" and obj.obj_id in self._predicted_positions:
                obj_pos = self._predicted_positions[obj.obj_id]
            else:
                obj_pos = obj.pose.position
            dist = det_pos.distance_to(obj_pos)
            if dist <= radius and dist < best_dist:
                best_dist = dist
                best_match = obj.obj_id

        return best_match

    def _decay_confidence(self, obj: WorldObject, timestamp_sec: float) -> float:
        """按时间衰减置信度"""
        elapsed = timestamp_sec - obj.last_seen_sec
        if elapsed <= 0:
            return obj.confidence
        new_confidence = obj.confidence * math.exp(-self.decay_rate * elapsed)
        return max(0.0, min(1.0, new_confidence))

    @staticmethod
    def _generate_obj_id(det: WorldObject, scene_id: str) -> str:
        """为没有 obj_id 的检测生成唯一 ID"""
        import hashlib

        pos = det.pose.position
        data = f"{scene_id}:{det.obj_type}:{det.name}:{pos.x:.3f}:{pos.y:.3f}:{pos.z:.3f}"
        h = hashlib.md5(data.encode()).hexdigest()[:8]
        return f"{det.obj_type}_{h}"
