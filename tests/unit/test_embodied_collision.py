"""
Unit tests for lightweight collision detection.

Zero external dependencies — pure geometry verification.
"""

import math

import pytest

from powermem.embodied.collision import (
    AABB,
    Capsule,
    CollisionBody,
    CollisionChecker,
    Sphere,
    bodies_intersect,
    build_collision_bodies,
)
from powermem.embodied.parsers.base import ParseResult
from powermem.embodied.types import Vec3


class TestSphere:
    def test_contains_center(self):
        s = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        assert s.contains(Vec3(0, 0, 0))

    def test_contains_surface(self):
        s = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        assert s.contains(Vec3(1.0, 0, 0))

    def test_not_contains_outside(self):
        s = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        assert not s.contains(Vec3(2.0, 0, 0))

    def test_distance_to_inside(self):
        s = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        assert s.distance_to(Vec3(0, 0, 0)) == 0.0

    def test_distance_to_outside(self):
        s = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        assert s.distance_to(Vec3(3.0, 0, 0)) == pytest.approx(2.0)

    def test_aabb(self):
        s = Sphere(center=Vec3(1, 2, 3), radius=0.5)
        box = s.aabb()
        assert box.min == Vec3(0.5, 1.5, 2.5)
        assert box.max == Vec3(1.5, 2.5, 3.5)


class TestCapsule:
    def test_contains_on_segment(self):
        cap = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 2), radius=0.5)
        assert cap.contains(Vec3(0, 0, 1))

    def test_contains_near_segment(self):
        cap = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 2), radius=0.5)
        assert cap.contains(Vec3(0.3, 0, 1))

    def test_not_contains_far(self):
        cap = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 2), radius=0.5)
        assert not cap.contains(Vec3(2.0, 0, 1))

    def test_distance_to_on_surface(self):
        cap = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 2), radius=0.5)
        # point at radius distance from center of cylinder -> on surface
        d = cap.distance_to(Vec3(0.5, 0, 1))
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_distance_to_outside(self):
        cap = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 2), radius=0.5)
        d = cap.distance_to(Vec3(2.0, 0, 1))
        assert d == pytest.approx(1.5)

    def test_degenerate_to_sphere(self):
        cap = Capsule(a=Vec3(0, 0, 0), b=Vec3(0, 0, 0), radius=1.0)
        assert cap.contains(Vec3(0.5, 0, 0))
        assert not cap.contains(Vec3(2.0, 0, 0))

    def test_aabb(self):
        cap = Capsule(a=Vec3(0, 0, 0), b=Vec3(1, 2, 3), radius=0.5)
        box = cap.aabb()
        assert box.min == Vec3(-0.5, -0.5, -0.5)
        assert box.max == Vec3(1.5, 2.5, 3.5)


class TestAABB:
    def test_contains_inside(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(2, 2, 2))
        assert box.contains(Vec3(1, 1, 1))

    def test_contains_on_boundary(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(2, 2, 2))
        assert box.contains(Vec3(0, 1, 1))
        assert box.contains(Vec3(2, 2, 2))

    def test_not_contains_outside(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(2, 2, 2))
        assert not box.contains(Vec3(3, 1, 1))

    def test_intersects_true(self):
        a = AABB(min=Vec3(0, 0, 0), max=Vec3(2, 2, 2))
        b = AABB(min=Vec3(1, 1, 1), max=Vec3(3, 3, 3))
        assert a.intersects(b)

    def test_intersects_false(self):
        a = AABB(min=Vec3(0, 0, 0), max=Vec3(1, 1, 1))
        b = AABB(min=Vec3(2, 2, 2), max=Vec3(3, 3, 3))
        assert not a.intersects(b)

    def test_distance_to_inside(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(2, 2, 2))
        assert box.distance_to(Vec3(1, 1, 1)) == 0.0

    def test_distance_to_outside(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(2, 2, 2))
        d = box.distance_to(Vec3(3, 0, 0))
        assert d == pytest.approx(1.0)

    def test_distance_to_corner(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(1, 1, 1))
        d = box.distance_to(Vec3(2, 2, 2))
        assert d == pytest.approx(math.sqrt(3))

    def test_center(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(2, 4, 6))
        assert box.center() == Vec3(1, 2, 3)

    def test_diagonal(self):
        box = AABB(min=Vec3(0, 0, 0), max=Vec3(1, 1, 1))
        assert box.diagonal() == pytest.approx(math.sqrt(3))


class TestCollisionBody:
    def test_wrapper_sphere(self):
        s = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        body = CollisionBody(
            entity_id="robot:base",
            geom_type="sphere",
            geometry=s,
            link_name="base",
        )
        assert body.contains(Vec3(0, 0, 0))
        assert body.distance_to(Vec3(2, 0, 0)) == pytest.approx(1.0)


class TestBodiesIntersect:
    def test_sphere_sphere_intersect(self):
        a = CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 1.0))
        b = CollisionBody("b", "sphere", Sphere(Vec3(1.5, 0, 0), 1.0))
        assert bodies_intersect(a, b)

    def test_sphere_sphere_no_intersect(self):
        a = CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 0.5))
        b = CollisionBody("b", "sphere", Sphere(Vec3(2.0, 0, 0), 0.5))
        assert not bodies_intersect(a, b)

    def test_sphere_capsule_intersect(self):
        s = CollisionBody("s", "sphere", Sphere(Vec3(0, 0, 0), 0.5))
        cap = CollisionBody("c", "capsule", Capsule(Vec3(0, 0, 0), Vec3(0, 0, 2), 0.3))
        assert bodies_intersect(s, cap)

    def test_capsule_capsule_intersect_crossing(self):
        a = CollisionBody("a", "capsule", Capsule(Vec3(-1, 0, 0), Vec3(1, 0, 0), 0.2))
        b = CollisionBody("b", "capsule", Capsule(Vec3(0, -1, 0), Vec3(0, 1, 0), 0.2))
        assert bodies_intersect(a, b)

    def test_capsule_capsule_no_intersect(self):
        a = CollisionBody("a", "capsule", Capsule(Vec3(0, 0, 0), Vec3(1, 0, 0), 0.1))
        b = CollisionBody("b", "capsule", Capsule(Vec3(0, 2, 0), Vec3(1, 2, 0), 0.1))
        assert not bodies_intersect(a, b)

    def test_aabb_broadphase_miss(self):
        a = CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 0.1))
        b = CollisionBody("b", "sphere", Sphere(Vec3(10, 0, 0), 0.1))
        assert not bodies_intersect(a, b)


class TestCollisionChecker:
    def test_check_point(self):
        checker = CollisionChecker()
        checker.add_body(CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 1.0)))
        checker.add_body(CollisionBody("b", "sphere", Sphere(Vec3(3, 0, 0), 1.0)))
        hits = checker.check_point(Vec3(0.5, 0, 0))
        assert len(hits) == 1
        assert hits[0].entity_id == "a"

    def test_check_intersections(self):
        checker = CollisionChecker()
        checker.add_body(CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 1.0)))
        checker.add_body(CollisionBody("b", "sphere", Sphere(Vec3(1.5, 0, 0), 1.0)))
        checker.add_body(CollisionBody("c", "sphere", Sphere(Vec3(5, 0, 0), 1.0)))
        pairs = checker.check_intersections()
        assert len(pairs) == 1
        ids = {pairs[0][0].entity_id, pairs[0][1].entity_id}
        assert ids == {"a", "b"}

    def test_nearest_body(self):
        checker = CollisionChecker()
        checker.add_body(CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 0.5)))
        checker.add_body(CollisionBody("b", "sphere", Sphere(Vec3(3, 0, 0), 0.5)))
        nearest, dist = checker.nearest_body(Vec3(2, 0, 0))
        assert nearest.entity_id == "b"
        assert dist == pytest.approx(0.5)

    def test_nearest_body_empty(self):
        checker = CollisionChecker()
        assert checker.nearest_body(Vec3(0, 0, 0)) is None

    def test_clear(self):
        checker = CollisionChecker()
        checker.add_body(CollisionBody("a", "sphere", Sphere(Vec3(0, 0, 0), 1.0)))
        checker.clear()
        assert len(checker.bodies) == 0


class TestBuildCollisionBodies:
    def test_from_parse_result(self):
        result = ParseResult()
        result.links = [
            {"name": "base", "mass": 8.0, "com": {"x": 0, "y": 0, "z": 0.5}},
            {"name": "arm", "mass": 1.0, "com": {"x": 0.5, "y": 0, "z": 1.0}},
        ]
        result.source_hash = "abc123"
        bodies = build_collision_bodies(result, default_radius=0.05)
        assert len(bodies) == 2
        assert bodies[0].link_name == "base"
        assert bodies[0].geom_type == "sphere"
        # radius ~ mass^(1/3) * default_radius
        # 8^(1/3) = 2, so radius ~ 0.1
        assert bodies[0].geometry.radius == pytest.approx(0.1, abs=0.01)

    def test_empty_links(self):
        result = ParseResult()
        bodies = build_collision_bodies(result)
        assert len(bodies) == 0
