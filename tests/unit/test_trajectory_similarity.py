"""
Tests for trajectory similarity metrics (DTW, signatures, pre-filtering).
"""

from __future__ import annotations

import math

import pytest

from powermem.embodied.trajectory_similarity import (
    dtw_distance,
    dtw_distance_normalized,
    signature_compatible,
    trajectory_bounding_box_diagonal,
    trajectory_feature_signature,
    trajectory_principal_direction,
    trajectory_total_length,
)
from powermem.embodied.types import Vec3


class TestDTW:
    def test_identical_trajectories(self):
        traj = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)]
        assert dtw_distance(traj, traj) == 0.0

    def test_single_point(self):
        a = [Vec3(1, 2, 3)]
        b = [Vec3(1, 2, 3)]
        assert dtw_distance(a, b) == 0.0

    def test_different_shapes(self):
        a = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)]
        b = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        d = dtw_distance(a, b)
        assert d > 0.0

    def test_time_warping(self):
        """DTW should handle different sampling rates along the same path."""
        a = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0), Vec3(3, 0, 0)]
        b = [Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(3, 0, 0)]
        d = dtw_distance(a, b)
        # Same path, just missing one sample -> small distance
        assert d < 2.0

    def test_reverse_trajectory(self):
        a = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)]
        b = [Vec3(2, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 0)]
        d = dtw_distance(a, b)
        assert d > 0.0
        # DTW does NOT align reversed sequences well (unlike shape-based methods)
        assert d > 1.0

    def test_empty_trajectory(self):
        a = [Vec3(0, 0, 0)]
        b = []
        assert dtw_distance(a, b) == float("inf")

    def test_band_restriction(self):
        """Sakoe-Chiba band should affect warping freedom."""
        a = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0), Vec3(3, 0, 0)]
        b = [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(3, 0, 0), Vec3(3, 0, 0)]
        d_unbounded = dtw_distance(a, b, bandwidth=None)
        d_bounded = dtw_distance(a, b, bandwidth=1)
        # Band restriction makes matching harder = larger distance
        assert d_bounded >= d_unbounded

    def test_normalized_comparable_across_lengths(self):
        short = [Vec3(0, 0, 0), Vec3(1, 0, 0)]
        long = [Vec3(0, 0, 0), Vec3(0.5, 0, 0), Vec3(1, 0, 0)]
        d_short = dtw_distance_normalized(short, short)
        d_long = dtw_distance_normalized(long, long)
        assert d_short == 0.0
        assert d_long == 0.0


class TestTrajectoryFeatures:
    def test_total_length_straight_line(self):
        traj = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(1, 0, 0), 1.0),
            (Vec3(3, 0, 0), 2.0),
        ]
        positions = [wp[0] for wp in traj]
        assert trajectory_total_length(positions) == pytest.approx(3.0)

    def test_total_length_diagonal(self):
        traj = [(Vec3(0, 0, 0), 0.0), (Vec3(1, 1, 1), 1.0)]
        positions = [wp[0] for wp in traj]
        assert trajectory_total_length(positions) == pytest.approx(math.sqrt(3))

    def test_bbox_diagonal(self):
        traj = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(1, 2, 3), 1.0),
        ]
        positions = [wp[0] for wp in traj]
        assert trajectory_bounding_box_diagonal(positions) == pytest.approx(math.sqrt(14))

    def test_principal_direction(self):
        traj = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(3, 0, 0), 1.0),
        ]
        positions = [wp[0] for wp in traj]
        dx, dy, dz = trajectory_principal_direction(positions)
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(0.0)
        assert dz == pytest.approx(0.0)

    def test_principal_direction_single_point(self):
        assert trajectory_principal_direction([Vec3(1, 2, 3)]) == (0.0, 0.0, 0.0)

    def test_feature_signature(self):
        traj = [
            (Vec3(0, 0, 0), 0.0),
            (Vec3(1, 0, 0), 1.0),
            (Vec3(2, 0, 0), 2.0),
        ]
        sig = trajectory_feature_signature(traj)
        assert len(sig) == 8
        duration, total_len, bbox_diag, avg_speed, dx, dy, dz, count = sig
        assert duration == pytest.approx(2.0)
        assert total_len == pytest.approx(2.0)
        assert avg_speed == pytest.approx(1.0)
        assert dx == pytest.approx(1.0)
        assert count == 3.0

    def test_signature_empty(self):
        sig = trajectory_feature_signature([])
        assert sig == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class TestSignatureCompatibility:
    def test_identical_compatible(self):
        traj = [(Vec3(0, 0, 0), 0.0), (Vec3(1, 0, 0), 1.0)]
        sig = trajectory_feature_signature(traj)
        assert signature_compatible(sig, sig)

    def test_duration_mismatch_rejected(self):
        short = [(Vec3(0, 0, 0), 0.0), (Vec3(1, 0, 0), 1.0)]
        long = [(Vec3(0, 0, 0), 0.0), (Vec3(1, 0, 0), 10.0)]
        sig_s = trajectory_feature_signature(short)
        sig_l = trajectory_feature_signature(long)
        # duration ratio = 10.0 > default tol 2.0
        assert not signature_compatible(sig_s, sig_l)

    def test_direction_opposite_rejected(self):
        a = [(Vec3(0, 0, 0), 0.0), (Vec3(1, 0, 0), 1.0)]
        b = [(Vec3(0, 0, 0), 0.0), (Vec3(-1, 0, 0), 1.0)]
        sig_a = trajectory_feature_signature(a)
        sig_b = trajectory_feature_signature(b)
        # dot product = -1.0 < default min_direction_dot -0.5
        assert not signature_compatible(sig_a, sig_b)

    def test_perpendicular_accepted(self):
        a = [(Vec3(0, 0, 0), 0.0), (Vec3(1, 0, 0), 1.0)]
        b = [(Vec3(0, 0, 0), 0.0), (Vec3(0, 1, 0), 1.0)]
        sig_a = trajectory_feature_signature(a)
        sig_b = trajectory_feature_signature(b)
        # dot = 0.0 >= -0.5 -> accepted
        assert signature_compatible(sig_a, sig_b)

    def test_waypoint_count_ratio_rejected(self):
        a = [(Vec3(0, 0, 0), 0.0), (Vec3(1, 0, 0), 1.0)]
        b = [(Vec3(0, 0, 0), 0.0)] + [(Vec3(float(i) * 0.01, 0, 0), float(i)) for i in range(1, 20)]
        sig_a = trajectory_feature_signature(a)
        sig_b = trajectory_feature_signature(b)
        # count ratio = 19 / 2 = 9.5 > 5.0
        assert not signature_compatible(sig_a, sig_b)
