"""
Unit tests for forward kinematics and collision body transformation.

Zero external dependencies.
"""

import math

import pytest

from powermem.embodied.collision import AABB, Capsule, CollisionBody, Sphere
from powermem.embodied.kinematics import (
    Transform,
    dh_to_transform,
    forward_kinematics,
    forward_kinematics_poses,
    transform_collision_body,
    transform_collision_bodies,
)
from powermem.embodied.physical_model import DHParameter
from powermem.embodied.types import Vec3


class TestTransform:
    def test_identity(self):
        T = Transform.identity()
        v = Vec3(1, 2, 3)
        assert T.apply(v) == v

    def test_translation(self):
        T = Transform.from_translation(1, 2, 3)
        v = Vec3(0, 0, 0)
        assert T.apply(v) == Vec3(1, 2, 3)

    def test_rotation_z_90(self):
        # 绕 Z 轴旋转 90 度
        T = Transform(
            m00=0, m01=-1, m02=0, m03=0,
            m10=1, m11=0, m12=0, m13=0,
            m20=0, m21=0, m22=1, m23=0,
        )
        v = Vec3(1, 0, 0)
        r = T.apply(v)
        assert r.x == pytest.approx(0.0, abs=1e-9)
        assert r.y == pytest.approx(1.0, abs=1e-9)
        assert r.z == pytest.approx(0.0, abs=1e-9)

    def test_compose_translation(self):
        T1 = Transform.from_translation(1, 0, 0)
        T2 = Transform.from_translation(0, 2, 0)
        T = T1 @ T2
        v = Vec3(0, 0, 0)
        assert T.apply(v) == Vec3(1, 2, 0)

    def test_to_pose(self):
        T = Transform.from_translation(1, 2, 3)
        pose = T.to_pose()
        assert pose.position == Vec3(1, 2, 3)


class TestDHToTransform:
    def test_pure_translation_z(self):
        dh = DHParameter(d=1.0, theta=0.0, a=0.0, alpha=0.0)
        T = dh_to_transform(dh, 0.0)
        assert T.apply(Vec3(0, 0, 0)) == Vec3(0, 0, 1)

    def test_rotation_z(self):
        dh = DHParameter(d=0.0, theta=0.0, a=0.0, alpha=0.0)
        T = dh_to_transform(dh, math.pi / 2)
        v = Vec3(1, 0, 0)
        r = T.apply(v)
        assert r.x == pytest.approx(0.0, abs=1e-9)
        assert r.y == pytest.approx(1.0, abs=1e-9)

    def test_prismatic_joint(self):
        # prismatic: theta fixed, d varies (joint angle added to d)
        dh = DHParameter(d=0.5, theta=0.0, a=0.0, alpha=0.0)
        T = dh_to_transform(dh, 0.3)  # joint angle treated as added theta here
        # In our convention joint_angle adds to theta, not d
        # This test just verifies the transform is well-formed
        v = Vec3(0, 0, 0)
        r = T.apply(v)
        assert abs(r.z - 0.5) < 1e-9


class TestForwardKinematics:
    def test_single_joint_rotation(self):
        dh = [DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0)]
        angles = [0.0]
        transforms = forward_kinematics(dh, angles)
        assert len(transforms) == 1
        # End effector at (1, 0, 0)
        pos = transforms[0].apply(Vec3(0, 0, 0))
        assert pos.x == pytest.approx(1.0)
        assert pos.y == pytest.approx(0.0)

    def test_two_joint_planar_arm(self):
        # 2-DOF planar arm: both revolute about Z
        # Link 1: length 1 along X
        # Link 2: length 1 along X
        dh = [
            DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0),
            DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0),
        ]
        angles = [0.0, 0.0]
        transforms = forward_kinematics(dh, angles)
        assert len(transforms) == 2
        # Both links aligned with X axis
        end = transforms[1].apply(Vec3(0, 0, 0))
        assert end.x == pytest.approx(2.0)
        assert end.y == pytest.approx(0.0)

    def test_two_joint_bent_90(self):
        dh = [
            DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0),
            DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0),
        ]
        angles = [0.0, math.pi / 2]
        transforms = forward_kinematics(dh, angles)
        end = transforms[1].apply(Vec3(0, 0, 0))
        # First link along X, second link rotated 90deg about Z
        assert end.x == pytest.approx(1.0)
        assert end.y == pytest.approx(1.0)

    def test_empty_dh(self):
        assert forward_kinematics([], []) == []

    def test_missing_angles(self):
        dh = [DHParameter(), DHParameter()]
        transforms = forward_kinematics(dh, [0.0])
        assert len(transforms) == 2


class TestForwardKinematicsPoses:
    def test_returns_poses(self):
        dh = [DHParameter(d=0.0, theta=0.0, a=1.0, alpha=0.0)]
        poses = forward_kinematics_poses(dh, [0.0])
        assert len(poses) == 1
        assert poses[0].position.x == pytest.approx(1.0)


class TestTransformCollisionBody:
    def test_sphere_translation(self):
        body = CollisionBody(
            "s", "sphere",
            Sphere(center=Vec3(1, 0, 0), radius=0.1),
        )
        T = Transform.from_translation(0, 2, 0)
        world = transform_collision_body(body, T)
        assert world.geometry.center == Vec3(1, 2, 0)
        assert world.geometry.radius == pytest.approx(0.1)

    def test_capsule_rotation(self):
        body = CollisionBody(
            "c", "capsule",
            Capsule(a=Vec3(0, 0, 0), b=Vec3(1, 0, 0), radius=0.05),
        )
        # 90 deg about Z
        T = Transform(
            m00=0, m01=-1, m02=0, m03=0,
            m10=1, m11=0, m12=0, m13=0,
            m20=0, m21=0, m22=1, m23=0,
        )
        world = transform_collision_body(body, T)
        assert world.geometry.a.x == pytest.approx(0.0, abs=1e-9)
        assert world.geometry.a.y == pytest.approx(0.0, abs=1e-9)
        assert world.geometry.b.x == pytest.approx(0.0, abs=1e-9)
        assert world.geometry.b.y == pytest.approx(1.0, abs=1e-9)

    def test_aabb_rotation(self):
        body = CollisionBody(
            "b", "aabb",
            AABB(min=Vec3(-1, -1, -1), max=Vec3(1, 1, 1)),
        )
        # 45 deg about Z
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        T = Transform(
            m00=c, m01=-s, m02=0, m03=0,
            m10=s, m11=c, m12=0, m13=0,
            m20=0, m21=0, m22=1, m23=0,
        )
        world = transform_collision_body(body, T)
        # Rotated cube should have larger AABB
        assert world.geometry.min.x < -1.0
        assert world.geometry.max.x > 1.0


class TestTransformCollisionBodies:
    def test_batch_transform(self):
        bodies = [
            CollisionBody("b0", "sphere", Sphere(Vec3(1, 0, 0), 0.1), link_name="link0"),
            CollisionBody("b1", "sphere", Sphere(Vec3(1, 0, 0), 0.1), link_name="link1"),
        ]
        transforms = [
            Transform.from_translation(0, 0, 0),
            Transform.from_translation(0, 2, 0),
        ]
        link_map = {"link0": 0, "link1": 1}
        result = transform_collision_bodies(bodies, transforms, link_map)
        assert result[0].geometry.center == Vec3(1, 0, 0)
        assert result[1].geometry.center == Vec3(1, 2, 0)

    def test_unknown_link_keeps_original(self):
        body = CollisionBody("b", "sphere", Sphere(Vec3(1, 0, 0), 0.1), link_name="missing")
        transforms = [Transform.identity()]
        link_map = {"other": 0}
        result = transform_collision_bodies([body], transforms, link_map)
        assert result[0].geometry.center == Vec3(1, 0, 0)
