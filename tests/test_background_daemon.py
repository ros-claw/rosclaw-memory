"""
Tests for BackgroundDaemon — 后台维护守护线程
"""

import sqlite3
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from powermem.embodied import (
    BackgroundDaemon,
    DaemonConfig,
    DaemonStats,
    EmbodiedMemory,
    MemoryAtom,
    Pose,
    Vec3,
    WorldObject,
)
from powermem.embodied.schema import initialize_embodied_schema


# ---------------------------------------------------------------------------
# Mock adapters (same shape as test_embodied_deep.py)
# ---------------------------------------------------------------------------

class MockStorageAdapter:
    def __init__(self):
        self._store: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1000

    def add_memory(self, payload: Dict[str, Any]) -> int:
        mid = self._next_id
        self._next_id += 1
        self._store[mid] = {
            "id": mid,
            "data": payload.get("content", ""),
            "content": payload.get("content", ""),
            "metadata": payload.get("metadata", {}),
            "user_id": payload.get("user_id", ""),
            "agent_id": payload.get("agent_id", ""),
            "run_id": payload.get("run_id", ""),
            "created_at": "2024-01-01T00:00:00",
        }
        return mid

    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        return self._store.get(memory_id)

    def get_many_memories(self, memory_ids):
        return [self._store.get(mid) for mid in memory_ids]

    def delete_memory(self, memory_id: int, user_id=None, agent_id=None) -> bool:
        return self._store.pop(memory_id, None) is not None

    def search_memories(self, **kwargs) -> List[Dict[str, Any]]:
        limit = kwargs.get("limit", 30)
        results = []
        for mid, item in list(self._store.items())[:limit]:
            results.append({
                "id": mid,
                "memory": item["data"],
                "score": 0.9,
                "metadata": item.get("metadata", {}),
            })
        return results


class MockMemory:
    def __init__(self):
        self.storage = MockStorageAdapter()
        self.agent_id = "test_agent"

    def search(self, query, user_id=None, agent_id=None, run_id=None, filters=None, limit=30, threshold=None):
        results = self.storage.search_memories(
            query_embedding=None, user_id=user_id, agent_id=agent_id,
            run_id=run_id, filters=filters, limit=limit, query=query, threshold=threshold,
        )
        return {"results": results, "relations": []}

    def delete(self, memory_id: int) -> bool:
        return self.storage.delete_memory(memory_id)

    def update(self, memory_id: int, content: str, user_id=None, agent_id=None, metadata=None) -> Dict[str, Any]:
        item = self.storage._store.get(memory_id)
        if item is None:
            raise KeyError(memory_id)
        item["data"] = content
        item["content"] = content
        if metadata is not None:
            item["metadata"] = metadata
        return item


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_conn():
    # check_same_thread=False: daemon runs in a background thread and needs
    # to share the same in-memory connection; the daemon's _lock serialises access.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    initialize_embodied_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def embodied_memory(sqlite_conn):
    mock_mem = MockMemory()
    em = EmbodiedMemory(
        memory=mock_mem,
        db_conn=sqlite_conn,
        voxel_size=1.0,
        enable_plugin=False,
    )
    return em


@pytest.fixture
def daemon_config():
    """快速执行配置（缩短间隔用于测试）"""
    return DaemonConfig(
        enabled=True,
        vacuum_enabled=True,
        vacuum_interval_sec=0.1,
        forget_enabled=True,
        forget_interval_sec=0.1,
        decay_enabled=True,
        decay_interval_sec=0.1,
        cache_cleanup_enabled=True,
        cache_cleanup_interval_sec=0.1,
        sleep_interval_sec=0.05,
    )


@pytest.fixture
def fast_config():
    """极快配置（用于单次执行测试）"""
    return DaemonConfig(
        enabled=True,
        vacuum_enabled=True,
        vacuum_interval_sec=0.01,
        forget_enabled=True,
        forget_interval_sec=0.01,
        decay_enabled=True,
        decay_interval_sec=0.01,
        cache_cleanup_enabled=True,
        cache_cleanup_interval_sec=0.01,
        sleep_interval_sec=0.01,
    )


class TestBackgroundDaemonBasic:
    """基本功能测试"""

    def test_daemon_init(self, embodied_memory, daemon_config):
        daemon = BackgroundDaemon(embodied_memory, daemon_config)
        assert daemon._em is embodied_memory
        assert daemon._config is daemon_config
        assert not daemon.is_running

    def test_daemon_start_stop(self, embodied_memory, daemon_config):
        daemon = BackgroundDaemon(embodied_memory, daemon_config)
        daemon.start()
        assert daemon.is_running
        time.sleep(0.1)
        daemon.stop()
        assert not daemon.is_running

    def test_daemon_disabled(self, embodied_memory):
        config = DaemonConfig(enabled=False)
        daemon = BackgroundDaemon(embodied_memory, config)
        daemon.start()
        assert not daemon.is_running  # 不应启动

    def test_daemon_double_start(self, embodied_memory, daemon_config):
        daemon = BackgroundDaemon(embodied_memory, daemon_config)
        daemon.start()
        daemon.start()  # 第二次应忽略
        assert daemon.is_running
        daemon.stop()

    def test_daemon_stats(self, embodied_memory, daemon_config):
        daemon = BackgroundDaemon(embodied_memory, daemon_config)
        stats = daemon.stats
        assert isinstance(stats, DaemonStats)
        assert not stats.is_running
        assert stats.vacuum_count == 0
        assert stats.forget_count == 0


class TestVacuumTask:
    """vacuum_indexes() 任务测试"""

    def test_vacuum_executed(self, embodied_memory, fast_config):
        # 添加一些数据
        for i in range(5):
            atom = MemoryAtom(
                content=f"test {i}",
                spatial=Vec3(i, 0, 0),
            )
            embodied_memory.add_atom(atom)

        daemon = BackgroundDaemon(embodied_memory, fast_config)
        daemon.start()
        time.sleep(0.15)  # 等待至少一次 vacuum
        daemon.stop()

        stats = daemon.stats
        assert stats.vacuum_count >= 1
        assert stats.vacuum_errors == 0

    def test_vacuum_error_handling(self, embodied_memory, fast_config):
        # Mock vacuum_indexes 抛出异常
        with patch.object(embodied_memory, "vacuum_indexes", side_effect=Exception("test error")):
            daemon = BackgroundDaemon(embodied_memory, fast_config)
            daemon.start()
            time.sleep(0.15)
            daemon.stop()

            stats = daemon.stats
            assert stats.vacuum_errors >= 1
            assert any("vacuum" in msg for msg in stats.error_messages)


class TestForgetTask:
    """forget_old_memories() 任务测试"""

    def test_forget_executed(self, embodied_memory, fast_config):
        # 添加一些记忆
        for i in range(3):
            atom = MemoryAtom(content=f"old {i}")
            embodied_memory.add_atom(atom)

        daemon = BackgroundDaemon(embodied_memory, fast_config)
        daemon.start()
        time.sleep(0.15)
        daemon.stop()

        stats = daemon.stats
        assert stats.forget_count >= 1
        assert stats.forget_errors == 0

    def test_forget_with_config(self, embodied_memory):
        config = DaemonConfig(
            enabled=True,
            forget_enabled=True,
            forget_interval_sec=0.01,
            forget_max_age_days=7.0,
            forget_min_salience=0.5,
            sleep_interval_sec=0.01,
        )
        daemon = BackgroundDaemon(embodied_memory, config)
        daemon.start()
        time.sleep(0.15)
        daemon.stop()

        stats = daemon.stats
        assert stats.forget_count >= 1


class TestDecayTask:
    """object_permanence_decay() 任务测试"""

    def test_decay_executed(self, embodied_memory, fast_config):
        # 添加被遮挡对象
        obj = WorldObject(
            obj_id="occluded_1",
            pose=Pose(position=Vec3(0, 0, 0)),
            scene_id="test",
            occlusion_status="occluded",
            confidence=0.8,
            last_seen_sec=time.time() - 100,
        )
        embodied_memory.add_world_object(obj)

        daemon = BackgroundDaemon(embodied_memory, fast_config)
        daemon.start()
        time.sleep(0.15)
        daemon.stop()

        stats = daemon.stats
        assert stats.decay_count >= 1
        assert stats.objects_decayed >= 1

        # 检查对象置信度已衰减
        updated = embodied_memory.get_world_object("occluded_1")
        assert updated.confidence < 0.8

    def test_decay_marks_missing(self, embodied_memory, fast_config):
        # 添加低置信度被遮挡对象
        obj = WorldObject(
            obj_id="low_conf",
            pose=Pose(position=Vec3(0, 0, 0)),
            scene_id="test",
            occlusion_status="occluded",
            confidence=0.25,
            last_seen_sec=time.time() - 100,
        )
        embodied_memory.add_world_object(obj)

        daemon = BackgroundDaemon(embodied_memory, fast_config)
        daemon.start()
        time.sleep(0.15)
        daemon.stop()

        # 检查对象被标记为 missing
        updated = embodied_memory.get_world_object("low_conf")
        assert updated.occlusion_status == "missing"
        assert updated.confidence < 0.2

    def test_decay_no_visible_objects(self, embodied_memory, fast_config):
        # 添加可见对象（不应衰减）
        obj = WorldObject(
            obj_id="visible_1",
            pose=Pose(position=Vec3(0, 0, 0)),
            scene_id="test",
            occlusion_status="visible",
            confidence=1.0,
        )
        embodied_memory.add_world_object(obj)

        daemon = BackgroundDaemon(embodied_memory, fast_config)
        daemon.start()
        time.sleep(0.15)
        daemon.stop()

        # 检查对象未变化
        updated = embodied_memory.get_world_object("visible_1")
        assert updated.confidence == 1.0
        assert updated.occlusion_status == "visible"


class TestCacheCleanupTask:
    """cache_cleanup() 任务测试"""

    def test_cache_cleanup_executed(self, embodied_memory, fast_config):
        # 添加一些数据到缓存
        atom = MemoryAtom(content="cached")
        mid = embodied_memory.add_atom(atom)
        embodied_memory.get_atom(mid)  # 加载到缓存

        daemon = BackgroundDaemon(embodied_memory, fast_config)
        daemon.start()
        time.sleep(0.15)
        daemon.stop()

        stats = daemon.stats
        assert stats.cache_cleanup_count >= 1

    def test_cache_cleanup_removes_deleted(self, embodied_memory, fast_config):
        # 添加一个原子，加载到缓存
        atom = MemoryAtom(content="to_delete")
        mid = embodied_memory.add_atom(atom)
        embodied_memory.get_atom(mid)
        assert mid in embodied_memory._atom_cache

        # 删除原子 + 从 embodied_memories 表中移除（模拟 mock 不级联删除的情况）
        embodied_memory.delete_atom(mid)
        embodied_memory.db_conn.execute(
            "DELETE FROM embodied_memories WHERE memory_id = ?", (mid,)
        )
        embodied_memory.db_conn.commit()

        # 手动重新插入一个过期缓存条目（模拟缓存不一致）
        embodied_memory._atom_cache[mid] = atom

        # 直接同步调用 _cleanup_caches 验证逻辑正确性
        daemon = BackgroundDaemon(embodied_memory, fast_config)
        cleaned = daemon._cleanup_caches()

        assert cleaned >= 1
        assert mid not in embodied_memory._atom_cache


class TestEmbodiedMemoryIntegration:
    """EmbodiedMemory 集成测试"""

    def test_start_stop_daemon(self, embodied_memory):
        config = {
            "vacuum_interval_sec": 0.1,
            "forget_interval_sec": 0.1,
            "decay_interval_sec": 0.1,
            "cache_cleanup_interval_sec": 0.1,
            "sleep_interval_sec": 0.05,
        }
        embodied_memory.start_background_daemon(config)
        assert embodied_memory._daemon is not None
        assert embodied_memory._daemon.is_running

        time.sleep(0.15)

        embodied_memory.stop_background_daemon()
        assert embodied_memory._daemon is None

    def test_get_daemon_stats(self, embodied_memory):
        config = {"sleep_interval_sec": 0.05}
        embodied_memory.start_background_daemon(config)
        time.sleep(0.1)

        stats = embodied_memory.get_daemon_stats()
        assert stats is not None
        assert "is_running" in stats
        assert stats["is_running"]

        embodied_memory.stop_background_daemon()

    def test_get_daemon_stats_not_running(self, embodied_memory):
        stats = embodied_memory.get_daemon_stats()
        assert stats is None

    def test_daemon_with_real_workflow(self, embodied_memory):
        """完整工作流：添加数据 → 启动守护线程 → 验证维护"""
        # 1. 添加多个原子
        atoms = []
        for i in range(10):
            atom = MemoryAtom(
                content=f"workflow {i}",
                spatial=Vec3(i, 0, 0),
            )
            mid = embodied_memory.add_atom(atom)
            atoms.append(mid)

        # 2. 添加被遮挡对象
        obj = WorldObject(
            obj_id="workflow_obj",
            pose=Pose(position=Vec3(0, 0, 0)),
            scene_id="workflow",
            occlusion_status="occluded",
            confidence=0.9,
            last_seen_sec=time.time() - 50,
        )
        embodied_memory.add_world_object(obj)

        # 3. 删除一些原子
        for mid in atoms[:3]:
            embodied_memory.delete_atom(mid)

        # 4. 启动守护线程
        config = {
            "vacuum_interval_sec": 0.05,
            "forget_interval_sec": 0.05,
            "decay_interval_sec": 0.05,
            "cache_cleanup_interval_sec": 0.05,
            "sleep_interval_sec": 0.02,
        }
        embodied_memory.start_background_daemon(config)
        time.sleep(0.2)  # 等待多次维护周期

        # 5. 验证
        stats = embodied_memory.get_daemon_stats()
        assert stats["vacuum_count"] >= 1
        assert stats["decay_count"] >= 1
        assert stats["cache_cleanup_count"] >= 1

        # 检查被遮挡对象已衰减
        updated_obj = embodied_memory.get_world_object("workflow_obj")
        assert updated_obj.confidence < 0.9

        # 检查已删除原子不在缓存中
        for mid in atoms[:3]:
            assert mid not in embodied_memory._atom_cache

        embodied_memory.stop_background_daemon()


class TestDaemonConfig:
    """配置测试"""

    def test_default_config(self):
        config = DaemonConfig()
        assert config.enabled
        assert config.vacuum_interval_sec == 3600.0
        assert config.forget_interval_sec == 86400.0
        assert config.decay_interval_sec == 60.0
        assert config.cache_cleanup_interval_sec == 600.0

    def test_custom_config(self):
        config = DaemonConfig(
            enabled=False,
            vacuum_interval_sec=100.0,
            forget_max_age_days=7.0,
        )
        assert not config.enabled
        assert config.vacuum_interval_sec == 100.0
        assert config.forget_max_age_days == 7.0

    def test_config_from_dict(self, embodied_memory):
        config_dict = {
            "vacuum_interval_sec": 500.0,
            "forget_enabled": False,
        }
        embodied_memory.start_background_daemon(config_dict)
        assert embodied_memory._daemon._config.vacuum_interval_sec == 500.0
        assert not embodied_memory._daemon._config.forget_enabled
        embodied_memory.stop_background_daemon()


class TestDaemonStats:
    """统计测试"""

    def test_stats_initial(self):
        stats = DaemonStats()
        assert stats.start_time == 0.0
        assert stats.vacuum_count == 0
        assert stats.memories_forgotten == 0
        assert stats.objects_decayed == 0
        assert not stats.is_running
        assert stats.error_messages == []

    def test_stats_tracking(self, embodied_memory, fast_config):
        daemon = BackgroundDaemon(embodied_memory, fast_config)
        daemon.start()
        time.sleep(0.15)
        daemon.stop()

        stats = daemon.stats
        assert stats.start_time > 0
        assert stats.vacuum_count >= 1
        assert stats.forget_count >= 1
        assert stats.decay_count >= 1
        assert stats.cache_cleanup_count >= 1

    def test_stats_error_tracking(self, embodied_memory, fast_config):
        with patch.object(embodied_memory, "vacuum_indexes", side_effect=Exception("test")):
            daemon = BackgroundDaemon(embodied_memory, fast_config)
            daemon.start()
            time.sleep(0.15)
            daemon.stop()

            stats = daemon.stats
            assert stats.vacuum_errors >= 1
            assert len(stats.error_messages) >= 1
