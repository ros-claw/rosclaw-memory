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
    """

    def __init__(
        self,
        world_object_store: Any,
        decay_rate: float = 0.05,
        missing_threshold: float = 0.2,
    ):
        """
        Args:
            world_object_store: WorldObjectStore 实例
            decay_rate: 每秒置信度衰减率（默认 5%/秒）
            missing_threshold: 置信度低于此值时标记为 missing
        """
        self.store = world_object_store
        self.decay_rate = decay_rate
        self.missing_threshold = missing_threshold

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

        # Step 4: 已有但未被匹配的对象 → 衰减 confidence
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
            elif new_status == "visible":
                new_status = "occluded"
                report.occluded.append(obj_id)
                report.transitions.append(
                    f"{obj_id}: visible -> occluded ({new_confidence:.2f})"
                )
            else:
                # 保持 occluded 状态，但 confidence 继续衰减
                report.occluded.append(obj_id)

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
        """确认对象可见，更新位姿和遮挡状态"""
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

    def _find_spatial_match(
        self,
        detection: WorldObject,
        existing_objects: List[WorldObject],
        exclude_ids: set,
        radius: float,
    ) -> Optional[str]:
        """在半径内按类型匹配最近的对象"""
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
            dist = det_pos.distance_to(obj.pose.position)
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
