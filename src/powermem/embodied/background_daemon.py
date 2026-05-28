"""
BackgroundDaemon — 具身记忆的后台维护守护线程

周期性执行维护任务：
- vacuum_indexes(): 重建空间索引，清理死条目
- forget_old_memories(): 遗忘过期低显著性记忆
- object_permanence_decay(): 衰减被遮挡对象的置信度
- cache_cleanup(): 清理过期缓存

设计原则：
- 所有任务独立，一个失败不影响其他
- 可配置间隔和启用状态
- 线程安全，不阻塞主线程
- 优雅关闭（设置停止标志，等待当前周期完成）
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .embodied_memory import EmbodiedMemory

logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """后台守护线程配置"""

    # 总开关
    enabled: bool = True

    # vacuum_indexes()
    vacuum_enabled: bool = True
    vacuum_interval_sec: float = 3600.0  # 1 小时
    vacuum_min_operations: int = 100  # 至少 100 次写操作后才触发

    # forget_old_memories()
    forget_enabled: bool = True
    forget_interval_sec: float = 86400.0  # 24 小时
    forget_max_age_days: float = 30.0
    forget_min_salience: float = 0.0

    # object_permanence_decay()
    decay_enabled: bool = True
    decay_interval_sec: float = 60.0  # 1 分钟
    decay_rate: float = 0.05  # 每秒衰减 5%

    # cache_cleanup()
    cache_cleanup_enabled: bool = True
    cache_cleanup_interval_sec: float = 600.0  # 10 分钟

    # 守护线程
    sleep_interval_sec: float = 1.0  # 主循环检查间隔


@dataclass
class DaemonStats:
    """守护线程运行统计"""

    start_time: float = 0.0
    last_vacuum_time: float = 0.0
    last_forget_time: float = 0.0
    last_decay_time: float = 0.0
    last_cache_cleanup_time: float = 0.0

    vacuum_count: int = 0
    forget_count: int = 0
    decay_count: int = 0
    cache_cleanup_count: int = 0

    vacuum_errors: int = 0
    forget_errors: int = 0
    decay_errors: int = 0
    cache_cleanup_errors: int = 0

    memories_forgotten: int = 0
    objects_decayed: int = 0
    cache_entries_cleaned: int = 0

    is_running: bool = False
    error_messages: list = field(default_factory=list)


class BackgroundDaemon:
    """具身记忆后台维护守护线程

    用法：
        daemon = BackgroundDaemon(embodied_memory, config)
        daemon.start()
        # ... 主程序运行 ...
        daemon.stop()  # 优雅关闭

    或通过 EmbodiedMemory 集成：
        em = EmbodiedMemory(...)
        em.start_background_daemon(config)
        # ...
        em.stop_background_daemon()
    """

    def __init__(
        self,
        em: "EmbodiedMemory",
        config: Optional[DaemonConfig] = None,
    ):
        self._em = em
        self._config = config or DaemonConfig()
        self._stats = DaemonStats()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()  # 保护 stats 更新

    @property
    def is_running(self) -> bool:
        """守护线程是否正在运行"""
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> DaemonStats:
        """获取运行统计（线程安全）"""
        with self._lock:
            return DaemonStats(**{
                k: (list(v) if isinstance(v, list) else v)
                for k, v in self._stats.__dict__.items()
            })

    def start(self) -> None:
        """启动守护线程"""
        if not self._config.enabled:
            logger.info("BackgroundDaemon disabled by config")
            return

        if self.is_running:
            logger.warning("BackgroundDaemon already running")
            return

        self._stop_event.clear()
        with self._lock:
            self._stats.start_time = time.time()
            self._stats.is_running = True

        self._thread = threading.Thread(
            target=self._run_loop,
            name="EmbodiedMemory-Daemon",
            daemon=True,  # 主程序退出时自动终止
        )
        self._thread.start()
        logger.info("BackgroundDaemon started")

    def stop(self, timeout: float = 5.0) -> None:
        """优雅关闭守护线程

        Args:
            timeout: 等待线程退出的最大秒数
        """
        if not self.is_running:
            return

        logger.info("BackgroundDaemon stopping...")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("BackgroundDaemon did not stop within %.1fs", timeout)
            else:
                logger.info("BackgroundDaemon stopped")

        with self._lock:
            self._stats.is_running = False
        self._thread = None

    def _run_loop(self) -> None:
        """守护线程主循环"""
        logger.debug("BackgroundDaemon loop started")

        while not self._stop_event.is_set():
            now = time.time()

            # vacuum_indexes()
            if self._should_run_vacuum(now):
                self._do_vacuum()

            # forget_old_memories()
            if self._should_run_forget(now):
                self._do_forget()

            # object_permanence_decay()
            if self._should_run_decay(now):
                self._do_decay()

            # cache_cleanup()
            if self._should_run_cache_cleanup(now):
                self._do_cache_cleanup()

            # 等待下一个检查周期
            self._stop_event.wait(timeout=self._config.sleep_interval_sec)

        logger.debug("BackgroundDaemon loop exited")

    # ========================================================================
    # 调度判断
    # ========================================================================

    def _should_run_vacuum(self, now: float) -> bool:
        if not self._config.vacuum_enabled:
            return False
        elapsed = now - self._stats.last_vacuum_time
        return elapsed >= self._config.vacuum_interval_sec

    def _should_run_forget(self, now: float) -> bool:
        if not self._config.forget_enabled:
            return False
        elapsed = now - self._stats.last_forget_time
        return elapsed >= self._config.forget_interval_sec

    def _should_run_decay(self, now: float) -> bool:
        if not self._config.decay_enabled:
            return False
        elapsed = now - self._stats.last_decay_time
        return elapsed >= self._config.decay_interval_sec

    def _should_run_cache_cleanup(self, now: float) -> bool:
        if not self._config.cache_cleanup_enabled:
            return False
        elapsed = now - self._stats.last_cache_cleanup_time
        return elapsed >= self._config.cache_cleanup_interval_sec

    # ========================================================================
    # 任务执行
    # ========================================================================

    def _do_vacuum(self) -> None:
        """执行 vacuum_indexes()"""
        try:
            with self._lock:
                result = self._em.vacuum_indexes()
                self._stats.last_vacuum_time = time.time()
                self._stats.vacuum_count += 1
            logger.debug("vacuum completed: %s", result)
        except Exception as e:
            with self._lock:
                self._stats.vacuum_errors += 1
                self._stats.error_messages.append(f"vacuum: {e}")
            logger.error("vacuum failed: %s", e, exc_info=True)

    def _do_forget(self) -> None:
        """执行 forget_old_memories()"""
        try:
            with self._lock:
                result = self._em.forget_old_memories(
                    max_age_days=self._config.forget_max_age_days,
                    min_salience=self._config.forget_min_salience,
                    dry_run=False,
                )
                deleted = result.get("deleted", 0)
                self._stats.last_forget_time = time.time()
                self._stats.forget_count += 1
                self._stats.memories_forgotten += deleted
            if deleted > 0:
                logger.info("forget: deleted %d memories", deleted)
            else:
                logger.debug("forget: no memories to delete")
        except Exception as e:
            with self._lock:
                self._stats.forget_errors += 1
                self._stats.error_messages.append(f"forget: {e}")
            logger.error("forget failed: %s", e, exc_info=True)

    def _do_decay(self) -> None:
        """执行 object_permanence_decay()

        衰减被遮挡对象的置信度，将 confidence < 0.2 的对象标记为 missing。
        """
        try:
            with self._lock:
                decayed_count = self._decay_occluded_objects()
                self._stats.last_decay_time = time.time()
                self._stats.decay_count += 1
                self._stats.objects_decayed += decayed_count
            if decayed_count > 0:
                logger.debug("decay: %d objects decayed", decayed_count)
        except Exception as e:
            with self._lock:
                self._stats.decay_errors += 1
                self._stats.error_messages.append(f"decay: {e}")
            logger.error("decay failed: %s", e, exc_info=True)

    def _decay_occluded_objects(self) -> int:
        """衰减被遮挡对象的置信度

        通过 WorldObjectStore.update_occlusion() 写入，确保缓存自动失效。

        Returns:
            衰减的对象数量
        """
        cursor = self._em.db_conn.cursor()

        # 查找所有 occluded 状态的对象
        cursor.execute("""
            SELECT obj_id, confidence, last_seen_sec
            FROM embodied_world_objects
            WHERE occlusion_status = 'occluded'
        """)

        rows = cursor.fetchall()
        now = time.time()
        decayed = 0
        threshold = 0.2

        for obj_id, confidence, last_seen_sec in rows:
            confidence = float(confidence or 1.0)
            last_seen_sec = float(last_seen_sec or 0.0)

            # 计算时间差（秒）
            elapsed = now - last_seen_sec if last_seen_sec > 0 else 0.0

            # 衰减: confidence = confidence * (1 - decay_rate * elapsed)
            new_confidence = confidence * (1.0 - self._config.decay_rate * elapsed)
            new_confidence = max(0.0, min(1.0, new_confidence))

            if new_confidence < confidence:
                new_status = "missing" if new_confidence < threshold else "occluded"
                # 通过 store 写入 + 自动缓存失效
                self._em.world_object_store.update_occlusion(
                    obj_id, new_status, new_confidence, last_seen_sec,
                )
                decayed += 1

        return decayed

    def _do_cache_cleanup(self) -> None:
        """清理过期缓存条目"""
        try:
            with self._lock:
                cleaned = self._cleanup_caches()
                self._stats.last_cache_cleanup_time = time.time()
                self._stats.cache_cleanup_count += 1
                self._stats.cache_entries_cleaned += cleaned
            if cleaned > 0:
                logger.debug("cache_cleanup: removed %d entries", cleaned)
        except Exception as e:
            with self._lock:
                self._stats.cache_cleanup_errors += 1
                self._stats.error_messages.append(f"cache_cleanup: {e}")
            logger.error("cache_cleanup failed: %s", e, exc_info=True)

    def _cleanup_caches(self) -> int:
        """清理各类缓存

        Returns:
            清理的条目总数
        """
        cleaned = 0

        # 1. 清理原子缓存中的已删除条目
        # （注意：OrderedDict 已有 LRU，这里额外清理无效引用）
        atom_cache = self._em._atom_cache
        to_remove = []
        for mid in list(atom_cache.keys()):
            # 检查是否还在数据库中
            cursor = self._em.db_conn.cursor()
            cursor.execute(
                "SELECT 1 FROM embodied_memories WHERE memory_id = ?",
                (mid,),
            )
            if cursor.fetchone() is None:
                to_remove.append(mid)
        for mid in to_remove:
            atom_cache.pop(mid, None)
            cleaned += 1

        # 2. 清理场景图缓存中的空场景
        sg_cache = self._em._scene_graph_cache
        to_remove = []
        for scene_id in list(sg_cache.keys()):
            cursor = self._em.db_conn.cursor()
            cursor.execute(
                "SELECT 1 FROM embodied_world_objects WHERE scene_id = ? LIMIT 1",
                (scene_id,),
            )
            if cursor.fetchone() is None:
                to_remove.append(scene_id)
        for scene_id in to_remove:
            sg_cache.pop(scene_id, None)
            cleaned += 1

        # 3. 清理世界对象缓存中的已删除条目
        wo_cache = self._em.world_object_store._object_cache
        to_remove = []
        for obj_id in list(wo_cache.keys()):
            cursor = self._em.db_conn.cursor()
            cursor.execute(
                "SELECT 1 FROM embodied_world_objects WHERE obj_id = ?",
                (obj_id,),
            )
            if cursor.fetchone() is None:
                to_remove.append(obj_id)
        for obj_id in to_remove:
            wo_cache.pop(obj_id, None)
            cleaned += 1

        return cleaned
