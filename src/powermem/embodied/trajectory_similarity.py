"""
Trajectory similarity metrics for embodied memory retrieval.

Provides:
- DTW (Dynamic Time Warping) for shape-aware trajectory comparison
- Lightweight trajectory feature signatures for coarse pre-filtering
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .types import Vec3


def _extract_positions(
    waypoints: List[Tuple[Vec3, float]],
) -> List[Vec3]:
    """Extract spatial positions from (Vec3, timestamp) waypoints."""
    return [wp[0] for wp in waypoints]


def _positions_to_array(positions: List[Vec3]) -> "np.ndarray":
    """Convert list of Vec3 to (N, 3) float64 numpy array."""
    arr = np.empty((len(positions), 3), dtype=np.float64)
    for i, p in enumerate(positions):
        arr[i, 0] = p.x
        arr[i, 1] = p.y
        arr[i, 2] = p.z
    return arr


def _euclidean_cost_matrix(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Vectorized pairwise Euclidean distance matrix.

    Args:
        a: (N, 3) array
        b: (M, 3) array

    Returns:
        (N, M) array of distances
    """
    # (N, 1, 3) - (1, M, 3) -> (N, M, 3) -> sqrt(sum) -> (N, M)
    return np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))


def dtw_distance(
    traj_a: List[Vec3],
    traj_b: List[Vec3],
    bandwidth: Optional[int] = None,
    max_distance: Optional[float] = None,
) -> float:
    """Dynamic Time Warping distance between two 3D trajectories (numpy-accelerated).

    Args:
        traj_a: List of Vec3 positions (length N)
        traj_b: List of Vec3 positions (length M)
        bandwidth: Sakoe-Chiba band width (max |i-j| allowed). None = unrestricted.
        max_distance: 若当前行最小值超过此阈值则提前返回 inf，避免无意义的完整计算。

    Returns:
        Scalar DTW distance (sum of Euclidean costs along optimal warping path)
    """
    n = len(traj_a)
    m = len(traj_b)
    if n == 0 or m == 0:
        return float("inf")
    if n == 1 and m == 1:
        return traj_a[0].distance_to(traj_b[0])

    # For very small trajectories, pure-Python distance_to is faster than numpy overhead.
    # Threshold determined empirically: numpy wins when n*m >= ~150.
    use_numpy = n * m >= 200
    if use_numpy:
        a_arr = _positions_to_array(traj_a)
        b_arr = _positions_to_array(traj_b)
        cost_matrix = _euclidean_cost_matrix(a_arr, b_arr)
    else:
        cost_matrix = None

    # Use two rows for O(min(N,M)) memory
    prev = [float("inf")] * m
    curr = [float("inf")] * m

    for i in range(n):
        left = float("inf")
        row_min = float("inf")
        row_costs = cost_matrix[i] if cost_matrix is not None else None
        for j in range(m):
            # Sakoe-Chiba band constraint
            if bandwidth is not None and abs(i - j) > bandwidth:
                curr[j] = float("inf")
                continue

            if row_costs is not None:
                cost = float(row_costs[j])
            else:
                cost = traj_a[i].distance_to(traj_b[j])
            if i == 0 and j == 0:
                curr[j] = cost
            elif i == 0:
                curr[j] = curr[j - 1] + cost
            elif j == 0:
                curr[j] = prev[j] + cost
            else:
                curr[j] = cost + min(prev[j], left, prev[j - 1])
            left = curr[j]
            if curr[j] < row_min:
                row_min = curr[j]

        # Early termination: 若当前行所有路径成本均已超过阈值，后续不可能更优
        if max_distance is not None and row_min > max_distance:
            return float("inf")
        prev, curr = curr, prev

    return prev[m - 1]


def dtw_distance_normalized(
    traj_a: List[Vec3],
    traj_b: List[Vec3],
    bandwidth: Optional[int] = None,
    max_distance: Optional[float] = None,
) -> float:
    """Normalized DTW distance = raw DTW / max(len_a, len_b).

    Makes scores comparable across different-length trajectories.
    """
    denom = max(len(traj_a), len(traj_b), 1)
    raw_max = max_distance * denom if max_distance is not None else None
    raw = dtw_distance(traj_a, traj_b, bandwidth=bandwidth, max_distance=raw_max)
    return raw / denom


def trajectory_total_length(positions: List[Vec3]) -> float:
    """Sum of Euclidean segment lengths."""
    total = 0.0
    for i in range(1, len(positions)):
        total += positions[i].distance_to(positions[i - 1])
    return total


def trajectory_bounding_box_diagonal(positions: List[Vec3]) -> float:
    """Diagonal of the axis-aligned bounding box."""
    if not positions:
        return 0.0
    min_x = min(p.x for p in positions)
    min_y = min(p.y for p in positions)
    min_z = min(p.z for p in positions)
    max_x = max(p.x for p in positions)
    max_y = max(p.y for p in positions)
    max_z = max(p.z for p in positions)
    return math.sqrt(
        (max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2
    )


def trajectory_principal_direction(positions: List[Vec3]) -> Tuple[float, float, float]:
    """Return normalized principal displacement vector (start -> end).

    Returns (0,0,0) for single-point trajectories.
    """
    if len(positions) < 2:
        return (0.0, 0.0, 0.0)
    dx = positions[-1].x - positions[0].x
    dy = positions[-1].y - positions[0].y
    dz = positions[-1].z - positions[0].z
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (dx / length, dy / length, dz / length)


def trajectory_feature_signature(
    waypoints: List[Tuple[Vec3, float]],
) -> Tuple[float, ...]:
    """Compute a lightweight signature for coarse trajectory pre-filtering.

    Returns:
        (duration_sec, total_length, bbox_diagonal, avg_speed,
         dir_x, dir_y, dir_z, waypoint_count)
    """
    positions = _extract_positions(waypoints)
    if not positions:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    timestamps = [wp[1] for wp in waypoints]
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    total_len = trajectory_total_length(positions)
    bbox_diag = trajectory_bounding_box_diagonal(positions)
    avg_speed = total_len / duration if duration > 0 else 0.0
    dir_x, dir_y, dir_z = trajectory_principal_direction(positions)

    return (
        float(duration),
        float(total_len),
        float(bbox_diag),
        float(avg_speed),
        float(dir_x),
        float(dir_y),
        float(dir_z),
        float(len(positions)),
    )


def signature_compatible(
    query_sig: Tuple[float, ...],
    candidate_sig: Tuple[float, ...],
    duration_ratio_tol: float = 2.0,
    length_ratio_tol: float = 3.0,
    bbox_ratio_tol: float = 3.0,
    min_direction_dot: float = -0.5,
) -> bool:
    """Quick reject heuristic: return False if candidate is clearly incompatible.

    A True result does NOT guarantee similarity; it just says "worth running DTW".
    """
    q_dur, q_len, q_bbox, _, q_dx, q_dy, q_dz, q_count = query_sig
    c_dur, c_len, c_bbox, _, c_dx, c_dy, c_dz, c_count = candidate_sig

    # Duration ratio check
    if q_dur > 0 and c_dur > 0:
        ratio = max(q_dur, c_dur) / min(q_dur, c_dur)
        if ratio > duration_ratio_tol:
            return False

    # Length ratio check
    if q_len > 0 and c_len > 0:
        ratio = max(q_len, c_len) / min(q_len, c_len)
        if ratio > length_ratio_tol:
            return False

    # BBox diagonal ratio check
    if q_bbox > 0 and c_bbox > 0:
        ratio = max(q_bbox, c_bbox) / min(q_bbox, c_bbox)
        if ratio > bbox_ratio_tol:
            return False

    # Direction alignment (allow opposite directions if min_direction_dot is negative)
    q_dir_len = math.sqrt(q_dx * q_dx + q_dy * q_dy + q_dz * q_dz)
    c_dir_len = math.sqrt(c_dx * c_dx + c_dy * c_dy + c_dz * c_dz)
    if q_dir_len > 0 and c_dir_len > 0:
        dot = (q_dx * c_dx + q_dy * c_dy + q_dz * c_dz) / (q_dir_len * c_dir_len)
        if dot < min_direction_dot:
            return False

    # Waypoint count ratio (extremely different sampling rates = noisy DTW)
    if q_count > 0 and c_count > 0:
        ratio = max(q_count, c_count) / min(q_count, c_count)
        if ratio > 5.0:
            return False

    return True
