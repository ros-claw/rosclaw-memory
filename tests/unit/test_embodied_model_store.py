"""
Unit tests for ModelStore (physical model persistence).

Uses SQLite in-memory DB to avoid external dependencies.
"""

import sqlite3

import pytest

from powermem.embodied.model_store import ModelStore, StoredModel
from powermem.embodied.parsers.base import ParseResult
from powermem.embodied.physical_model import DHParameter, JointLimit, RobotDynamics
from powermem.embodied.schema import initialize_embodied_schema


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    initialize_embodied_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_result():
    result = ParseResult()
    result.format = "urdf"
    result.source_hash = "deadbeef"
    result.dynamics = RobotDynamics(
        joint_names=["shoulder", "elbow"],
        dh_params=[DHParameter(d=0.5, a=0.0), DHParameter(d=0.0, a=0.3)],
        joint_limits=[
            JointLimit(min_rad=-1.57, max_rad=1.57, max_vel=2.0, max_torque=50.0),
            JointLimit(min_rad=-1.57, max_rad=1.57, max_vel=2.0, max_torque=30.0),
        ],
        link_masses=[2.0, 1.5],
        link_inertias=[[0.01] * 9, [0.005] * 9],
        collision_geoms=[
            {"type": "sphere", "link": "base", "center": [0, 0, 0.5], "radius": 0.1},
            {"type": "capsule", "link": "arm", "a": [0, 0, 1], "b": [0, 0, 2], "radius": 0.05},
        ],
    )
    return result


class TestModelStoreSaveLoad:
    def test_save_and_load(self, db, sample_result):
        store = ModelStore(db)
        model_id = store.save(sample_result, model_id="ur5_test", model_type="robot")
        assert model_id == "ur5_test"

        loaded = store.load(model_id)
        assert loaded is not None
        assert isinstance(loaded, StoredModel)
        assert loaded.model_id == "ur5_test"
        assert loaded.model_type == "robot"
        assert loaded.source_hash == "deadbeef"

    def test_load_missing(self, db):
        store = ModelStore(db)
        assert store.load("nonexistent") is None

    def test_auto_model_id(self, db, sample_result):
        store = ModelStore(db)
        mid = store.save(sample_result)
        assert mid.startswith("urdf_2dof_")
        loaded = store.load(mid)
        assert loaded is not None

    def test_update_existing(self, db, sample_result):
        store = ModelStore(db)
        store.save(sample_result, model_id="same")
        sample_result.dynamics.link_masses = [3.0, 2.0]
        store.save(sample_result, model_id="same")
        loaded = store.load("same")
        assert loaded is not None


class TestModelStoreListDelete:
    def test_list_all(self, db, sample_result):
        store = ModelStore(db)
        store.save(sample_result, model_id="m1", model_type="robot")
        store.save(sample_result, model_id="m2", model_type="environment")
        all_models = store.list_models()
        assert len(all_models) == 2

    def test_list_by_type(self, db, sample_result):
        store = ModelStore(db)
        store.save(sample_result, model_id="m1", model_type="robot")
        store.save(sample_result, model_id="m2", model_type="environment")
        robots = store.list_models(model_type="robot")
        assert len(robots) == 1
        assert robots[0]["model_id"] == "m1"

    def test_delete(self, db, sample_result):
        store = ModelStore(db)
        store.save(sample_result, model_id="delme")
        assert store.delete("delme")
        assert store.load("delme") is None
        assert not store.delete("delme")


class TestModelStoreCollision:
    def test_build_checker(self, db, sample_result):
        store = ModelStore(db)
        store.save(sample_result, model_id="col_test")
        checker = store.build_checker("col_test")
        assert checker is not None
        assert len(checker.bodies) == 2

    def test_build_checker_missing(self, db):
        store = ModelStore(db)
        assert store.build_checker("missing") is None

    def test_check_self_collision_none(self, db, sample_result):
        store = ModelStore(db)
        store.save(sample_result, model_id="no_col")
        # spheres/capsules are far apart in sample_result
        pairs = store.check_self_collision("no_col")
        assert len(pairs) == 0

    def test_check_self_collision_detected(self, db):
        store = ModelStore(db)
        result = ParseResult()
        result.format = "urdf"
        result.dynamics = RobotDynamics(
            joint_names=["j1"],
            dh_params=[DHParameter()],
            joint_limits=[JointLimit()],
            collision_geoms=[
                {"type": "sphere", "link": "a", "center": [0, 0, 0], "radius": 1.0},
                {"type": "sphere", "link": "b", "center": [0.5, 0, 0], "radius": 1.0},
            ],
        )
        store.save(result, model_id="has_col")
        pairs = store.check_self_collision("has_col")
        assert len(pairs) == 1


class TestStoredModelDynamics:
    def test_dynamics_roundtrip(self, db, sample_result):
        store = ModelStore(db)
        store.save(sample_result, model_id="dyn")
        loaded = store.load("dyn")
        assert loaded is not None
        dyn = loaded.dynamics
        assert dyn.joint_names == ["shoulder", "elbow"]
        assert len(dyn.dh_params) == 2
        assert dyn.dh_params[0].d == pytest.approx(0.5)
        assert len(dyn.joint_limits) == 2
        assert dyn.joint_limits[0].max_torque == pytest.approx(50.0)


class TestEndToEndPipeline:
    """端到端：解析 -> 构建碰撞体 -> 存储 -> 加载 -> 自碰撞检测"""

    def test_full_pipeline_from_parsed_urdf(self, db):
        from powermem.embodied.collision import build_collision_bodies
        from powermem.embodied.parsers import parse_model

        urdf = """\
<?xml version="1.0"?>
<robot name="two_link_arm">
  <link name="base">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry><sphere radius="0.1"/></geometry>
    </collision>
  </link>
  <link name="upper_arm">
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.6"/></geometry>
    </collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" velocity="2.0" effort="50"/>
  </joint>
</robot>
"""
        # 1. 解析
        result = parse_model(urdf)
        assert result.format == "urdf"
        assert len(result.links) == 2

        # 2. 从解析结果构建碰撞体
        bodies = build_collision_bodies(result)
        assert len(bodies) == 2
        assert bodies[0].geom_type == "sphere"
        assert bodies[1].geom_type == "capsule"

        # 3. 存储到数据库
        store = ModelStore(db)
        model_id = store.save(result, model_id="two_link_arm_v1")

        # 4. 加载并重建碰撞检测器
        checker = store.build_checker(model_id)
        assert checker is not None
        assert len(checker.bodies) == 2

        # 5. 自碰撞检测（零位姿下，base 的球在 z=0.5，upper_arm 的胶囊在 z=0，不应相交）
        pairs = store.check_self_collision(model_id)
        assert len(pairs) == 0

        # 6. 修改碰撞体使其相交，重新保存并检测
        result.dynamics.collision_geoms = [
            {"type": "sphere", "link": "base", "center": [0, 0, 0], "radius": 0.5},
            {"type": "sphere", "link": "upper_arm", "center": [0.3, 0, 0], "radius": 0.5},
        ]
        store.save(result, model_id="two_link_arm_col")
        pairs = store.check_self_collision("two_link_arm_col")
        assert len(pairs) == 1

    def test_check_collision_at_config(self, db):
        """在特定关节角配置下检测碰撞（FK + 碰撞检测）"""
        import math
        from powermem.embodied.physical_model import DHParameter, JointLimit, RobotDynamics

        store = ModelStore(db)
        result = ParseResult()
        result.format = "urdf"
        result.dynamics = RobotDynamics(
            joint_names=["shoulder"],
            dh_params=[DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0)],
            joint_limits=[JointLimit()],
            collision_geoms=[
                {"type": "sphere", "link": "shoulder", "center": [0.5, 0, 0], "radius": 0.2},
                {"type": "sphere", "link": "shoulder", "center": [1.5, 0, 0], "radius": 0.2},
            ],
        )
        store.save(result, model_id="fk_test")

        # 零位姿：两个球沿 X 轴相距 1.0，半径各 0.2，不相交
        pairs = store.check_collision_at_config("fk_test", [0.0])
        assert len(pairs) == 0

        # 旋转 180 度：第二个球绕 Z 转到 (-1.5, 0, 0)，第一个在 (0.5, 0, 0)
        # 距离 = 2.0，不相交
        pairs = store.check_collision_at_config("fk_test", [math.pi])
        assert len(pairs) == 0

        # 把两个球放近，旋转后相交
        result.dynamics.collision_geoms = [
            {"type": "sphere", "link": "shoulder", "center": [0.5, 0, 0], "radius": 0.4},
            {"type": "sphere", "link": "shoulder", "center": [0.8, 0, 0], "radius": 0.4},
        ]
        store.save(result, model_id="fk_test")
        pairs = store.check_collision_at_config("fk_test", [0.0])
        assert len(pairs) == 1
