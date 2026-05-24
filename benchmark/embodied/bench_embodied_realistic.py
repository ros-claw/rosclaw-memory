"""
Realistic embodied memory benchmark — event-driven ingestion model.

Simulates a service robot where:
- Raw sensor streams (RGB/depth/lidar at 10-30Hz) are filtered by SurprisalGate
- Only ~2-5% of sensor frames become memories (anomalies, novel observations)
- Trajectories are recorded per mission
- World objects are updated on detection/pose-change events
- Actions and outcomes are recorded on task execution

This models the real architecture:
    Sensor Frontend (100Hz) → Filter/SurprisalGate → EmbodiedMemory

Usage:
    PYTHONPATH=../../src python3 bench_embodied_realistic.py --scale 1.0 --output realistic.json
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import os
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.ingest_pipeline import SensorFrame
from powermem.embodied.memory_atom import MemoryAtom
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import (
    MemoryAction, Modality, Pose, Quaternion, TemporalInterval, Vec3, WorldObject,
)


class MockStorageAdapter:
    def __init__(self):
        self._store: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def add_memory(self, payload: Dict[str, Any]) -> int:
        mid = self._next_id
        self._next_id += 1
        self._store[mid] = {
            "id": mid,
            "content": payload.get("content", ""),
            "metadata": payload.get("metadata", {}),
            "user_id": payload.get("user_id", ""),
            "agent_id": payload.get("agent_id", ""),
        }
        return mid

    def get_memory(self, memory_id: int) -> Any:
        return self._store.get(memory_id)

    def delete_memory(self, memory_id: int, **kwargs) -> bool:
        return self._store.pop(memory_id, None) is not None

    def search_memories(self, **kwargs) -> list:
        limit = kwargs.get("limit", 30)
        results = []
        for mid, item in list(self._store.items())[:limit]:
            results.append({
                "id": mid,
                "memory": item["content"],
                "score": 0.9,
                "metadata": item.get("metadata", {}),
            })
        return results

    def update_memory(self, memory_id: int, content: str, **kwargs) -> Dict[str, Any]:
        item = self._store.get(memory_id)
        if item is None:
            raise KeyError(memory_id)
        item["content"] = content
        return item


class MockMemory:
    def __init__(self):
        self.storage = MockStorageAdapter()
        self.agent_id = "benchmark_agent"

    def add(self, content, **kwargs):
        return self.storage.add_memory({"content": content, "metadata": kwargs.get("metadata", {})})

    def search(self, query, **kwargs):
        results = self.storage.search_memories(limit=kwargs.get("limit", 30))
        return {"results": results, "relations": []}

    def get(self, memory_id, **kwargs):
        return self.storage.get_memory(memory_id)

    def delete(self, memory_id):
        return self.storage.delete_memory(memory_id)

    def update(self, memory_id, content, **kwargs):
        return self.storage.update_memory(memory_id, content)


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


# =============================================================================
# Scene definitions
# =============================================================================

ROOMS = {
    "living_room": {"center": Vec3(0, 0, 0), "size": (6.0, 5.0, 2.5)},
    "kitchen": {"center": Vec3(8, 0, 0), "size": (5.0, 4.0, 2.5)},
    "bedroom": {"center": Vec3(0, 6, 0), "size": (5.0, 5.0, 2.5)},
    "corridor": {"center": Vec3(4, 3, 0), "size": (8.0, 2.0, 2.5)},
}

FURNITURE_TEMPLATES = [
    {"name": "sofa", "type": "box", "size": (2.0, 0.8, 0.6)},
    {"name": "table", "type": "box", "size": (1.2, 0.8, 0.75)},
    {"name": "chair", "type": "box", "size": (0.5, 0.5, 0.9)},
    {"name": "shelf", "type": "box", "size": (1.0, 0.3, 1.8)},
    {"name": "tv_stand", "type": "box", "size": (1.5, 0.4, 0.5)},
]

OBJECT_TEMPLATES = [
    {"name": "mug", "type": "cylinder", "size": (0.08, 0.08, 0.12)},
    {"name": "book", "type": "box", "size": (0.15, 0.02, 0.21)},
    {"name": "plate", "type": "cylinder", "size": (0.15, 0.15, 0.02)},
    {"name": "bottle", "type": "cylinder", "size": (0.04, 0.04, 0.25)},
    {"name": "remote", "type": "box", "size": (0.04, 0.02, 0.15)},
]


def random_point_in_room(room_name: str) -> Vec3:
    room = ROOMS[room_name]
    cx, cy, cz = room["center"].x, room["center"].y, room["center"].z
    sx, sy, sz = room["size"]
    return Vec3(
        cx + random.uniform(-sx / 2, sx / 2),
        cy + random.uniform(-sy / 2, sy / 2),
        random.uniform(0.3, sz - 0.5),
    )


def random_trajectory_between(start_room: str, end_room: str, n_wp: int = 20) -> List[Tuple[Vec3, float]]:
    """Generate a smooth trajectory between two rooms via corridor."""
    start = random_point_in_room(start_room)
    end = random_point_in_room(end_room)
    via = Vec3(
        ROOMS["corridor"]["center"].x + random.uniform(-1, 1),
        ROOMS["corridor"]["center"].y + random.uniform(-0.5, 0.5),
        0.5,
    )
    waypoints = []
    duration = random.uniform(3.0, 8.0)
    for i in range(n_wp):
        t = i / (n_wp - 1)
        if t < 0.5:
            local_t = t * 2
            x = start.x + (via.x - start.x) * local_t
            y = start.y + (via.y - start.y) * local_t
            z = start.z + (via.z - start.z) * local_t
        else:
            local_t = (t - 0.5) * 2
            x = via.x + (end.x - via.x) * local_t
            y = via.y + (end.y - via.y) * local_t
            z = via.z + (end.z - via.z) * local_t
        waypoints.append((Vec3(x, y, z), t * duration))
    return waypoints


# =============================================================================
# Benchmark phases
# =============================================================================

def phase_setup_environment(em: EmbodiedMemory) -> Dict[str, Any]:
    """Initialize the world with rooms, furniture, and objects."""
    t0 = _now_ms()
    object_count = 0

    for room_name in ROOMS:
        for _ in range(random.randint(2, 4)):
            template = random.choice(FURNITURE_TEMPLATES)
            obj = WorldObject(
                obj_id=f"{room_name}_{template['name']}_{random.randint(1000, 9999)}",
                obj_type=template["type"],
                name=template["name"],
                pose=Pose(position=random_point_in_room(room_name)),
                size=template["size"],
                scene_id=room_name,
                semantic_tags=["furniture"],
            )
            em.add_world_object(obj)

        for _ in range(random.randint(3, 6)):
            template = random.choice(OBJECT_TEMPLATES)
            obj = WorldObject(
                obj_id=f"{room_name}_{template['name']}_{random.randint(1000, 9999)}",
                obj_type=template["type"],
                name=template["name"],
                pose=Pose(position=random_point_in_room(room_name)),
                size=template["size"],
                scene_id=room_name,
                semantic_tags=["graspable"],
            )
            em.add_world_object(obj)
            object_count += 1

    setup_ms = _now_ms() - t0
    return {"setup_ms": setup_ms, "object_count": object_count}


def phase_sensor_stream_with_surprisal(em: EmbodiedMemory, n_frames: int) -> Dict[str, Any]:
    """Simulate high-frequency sensor stream filtered by SurprisalGate.

    In real deployments:
    - RGB @ 30Hz, depth @ 10Hz, lidar @ 10Hz
    - SurprisalGate filters out 95-98% of frames as redundant
    - Only anomalies / novel observations become memories
    """
    t0 = _now_ms()
    modalities = [Modality.RGB, Modality.DEPTH, Modality.LIDAR]
    stored_count = 0
    filtered_count = 0

    for i in range(n_frames):
        room = random.choice(list(ROOMS.keys()))
        pos = random_point_in_room(room)
        modality = modalities[i % len(modalities)]

        # Simulate sensor data: mostly stable, occasional anomaly
        # Anomaly probability increases with scale to ensure some frames pass gate
        base_value = 0.5
        if random.random() < 0.03:  # 3% anomaly rate
            data_value = base_value + random.uniform(2.0, 5.0)  # large deviation
        else:
            data_value = base_value + random.uniform(-0.05, 0.05)  # normal noise

        frame = SensorFrame(
            modality=modality,
            timestamp_sec=float(i) * 0.033,
            data=[data_value] * 100,  # simplified feature array
            sensor_pose=Pose(position=pos),
            frame_id="world",
        )
        mid = em.ingest(frame, content=f"{modality.value} frame at {room}")
        if mid is not None:
            stored_count += 1
        else:
            filtered_count += 1

    # Flush any buffered frames
    flushed_id = em.flush_pipeline()
    if flushed_id is not None:
        stored_count += 1

    ingest_ms = _now_ms() - t0
    return {
        "ingest_ms": ingest_ms,
        "frames_streamed": n_frames,
        "frames_stored": stored_count,
        "frames_filtered": filtered_count,
        "storage_rate_pct": 100.0 * stored_count / n_frames if n_frames > 0 else 0,
    }


def phase_record_missions_with_events(
    em: EmbodiedMemory, n_missions: int
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Record missions with realistic event-driven writes during execution.

    Events injected during a mission:
    - collision / near-collision (5% probability)
    - detected new object (10% probability)
    - existing object moved (5% probability)
    - sensor anomaly at waypoint (5% probability)
    """
    t0 = _now_ms()
    room_names = list(ROOMS.keys())
    missions = []
    event_count = 0

    for i in range(n_missions):
        start_room = random.choice(room_names)
        end_room = random.choice([r for r in room_names if r != start_room])
        waypoints = random_trajectory_between(start_room, end_room)

        # Mission start
        action_id = em.record_action(
            f"Navigate from {start_room} to {end_room}",
            spatial=waypoints[0][0],
        )

        # Simulate events during traversal
        for wp_idx, (pos, ts) in enumerate(waypoints):
            # Event: near-collision with obstacle
            if random.random() < 0.05:
                em.record_action(
                    f"Near-collision detected at waypoint {wp_idx}",
                    spatial=pos,
                    outcome_status="warning",
                )
                event_count += 1

            # Event: detected new object
            if random.random() < 0.10:
                template = random.choice(OBJECT_TEMPLATES)
                new_obj = WorldObject(
                    obj_id=f"discovered_{template['name']}_{random.randint(1000, 9999)}",
                    obj_type=template["type"],
                    name=template["name"],
                    pose=Pose(position=pos),
                    size=template["size"],
                    scene_id=start_room,
                    semantic_tags=["graspable", "discovered"],
                )
                em.add_world_object(new_obj)
                event_count += 1

            # Event: existing object moved (update pose)
            if random.random() < 0.05:
                # Pick a random existing object and move it slightly
                scene_objs = em.world_object_store.list_by_scene(start_room, limit=10)
                if scene_objs:
                    moved_obj = random.choice(scene_objs)
                    new_pose = Pose(position=Vec3(
                        moved_obj.pose.position.x + random.uniform(-0.2, 0.2),
                        moved_obj.pose.position.y + random.uniform(-0.2, 0.2),
                        moved_obj.pose.position.z,
                    ))
                    em.update_world_object_pose(moved_obj.obj_id, new_pose, state="moved")
                    event_count += 1

        # Record trajectory summary
        traj_id = em.record_trajectory(
            f"mission_{i}: {start_room} -> {end_room}",
            waypoints,
        )

        # Mission outcome
        outcome = "success" if random.random() < 0.9 else "collision"
        outcome_id = em.record_outcome(
            action_id=action_id,
            content=f"Navigation ended with {outcome}",
            outcome_status=outcome,
            spatial=waypoints[-1][0],
        )
        missions.append({
            "traj_id": traj_id,
            "action_id": action_id,
            "outcome_id": outcome_id,
            "outcome": outcome,
        })

    mission_ms = _now_ms() - t0
    return {
        "mission_ms": mission_ms,
        "mission_count": n_missions,
        "event_count": event_count,
    }, missions


def phase_mixed_queries(
    em: EmbodiedMemory,
    n_queries: int,
    missions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run a realistic mix of read queries."""
    latencies: Dict[str, List[float]] = {
        "spatial": [],
        "temporal": [],
        "trajectory": [],
        "world_object": [],
        "scene_graph": [],
        "causal": [],
    }

    for _ in range(n_queries):
        query_type = random.choice([
            "spatial", "temporal", "trajectory", "world_object", "scene_graph", "causal",
        ])

        if query_type == "spatial":
            room = random.choice(list(ROOMS.keys()))
            center = ROOMS[room]["center"]
            t0 = _now_ms()
            em.search_near(center, radius=2.0)
            latencies["spatial"].append(_now_ms() - t0)

        elif query_type == "temporal":
            start = random.uniform(0.0, 50.0)
            interval = TemporalInterval(start_sec=start, end_sec=start + 5.0)
            t0 = _now_ms()
            em.search_temporal(interval)
            latencies["temporal"].append(_now_ms() - t0)

        elif query_type == "trajectory":
            room = random.choice(list(ROOMS.keys()))
            end_room = random.choice([r for r in ROOMS if r != room])
            query_wp = random_trajectory_between(room, end_room, n_wp=15)
            center = query_wp[len(query_wp) // 2][0]
            t_mid = query_wp[len(query_wp) // 2][1]
            interval = TemporalInterval(start_sec=t_mid - 5.0, end_sec=t_mid + 5.0)
            t0 = _now_ms()
            em.search_similar_trajectories(
                query_wp,
                spatial_center=center,
                spatial_radius=3.0,
                temporal_interval=interval,
                top_k=5,
            )
            latencies["trajectory"].append(_now_ms() - t0)

        elif query_type == "world_object":
            room = random.choice(list(ROOMS.keys()))
            center = ROOMS[room]["center"]
            t0 = _now_ms()
            em.search_world_objects(center, radius=2.0, scene_id=room)
            latencies["world_object"].append(_now_ms() - t0)

        elif query_type == "scene_graph":
            room = random.choice(list(ROOMS.keys()))
            t0 = _now_ms()
            em.auto_compute_relations(room, spatial_tolerance=0.05)
            latencies["scene_graph"].append(_now_ms() - t0)

        elif query_type == "causal":
            mission = random.choice(missions)
            action_id = mission["action_id"]
            t0 = _now_ms()
            effects = em.get_effects(action_id, limit=5)
            latencies["causal"].append(_now_ms() - t0)

    summary: Dict[str, Any] = {}
    for qtype, times in latencies.items():
        if times:
            summary[qtype] = {
                "count": len(times),
                "p50_ms": _percentile(times, 50),
                "p99_ms": _percentile(times, 99),
                "total_ms": sum(times),
            }
    return summary


def phase_write_burst(em: EmbodiedMemory, n_ops: int) -> Dict[str, Any]:
    """Single-threaded burst write of event atoms (SQLite-safe)."""
    t0 = _now_ms()
    for i in range(n_ops):
        room = random.choice(list(ROOMS.keys()))
        event_type = random.choice(["anomaly", "object_detected", "pose_update"])
        atom = MemoryAtom.from_observation(
            content=f"Event: {event_type} in {room}",
            sensor_pose=Pose(position=random_point_in_room(room)),
            modality=Modality.RGB,
            timestamp_sec=time.time(),
        )
        em.add_atom(atom)
    total_ms = _now_ms() - t0
    return {
        "burst_write_ms": total_ms,
        "total_ops": n_ops,
        "ops_per_sec": n_ops / (total_ms / 1000.0) if total_ms > 0 else 0,
    }


# =============================================================================
# Main runner
# =============================================================================

def run_realistic(scale_factor: float = 1.0) -> Dict[str, Any]:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    initialize_embodied_schema(conn)
    mock_mem = MockMemory()
    em = EmbodiedMemory(memory=mock_mem, db_conn=conn, enable_plugin=False)

    results: Dict[str, Any] = {"scale_factor": scale_factor}

    # Scale parameters
    n_sensor_frames = int(2000 * scale_factor)   # 2000 frames = ~1 min of 30Hz RGB
    n_missions = int(50 * scale_factor)
    n_queries = int(200 * scale_factor)
    n_burst_ops = int(200 * scale_factor)

    print("=== Realistic Embodied Memory Benchmark (Event-Driven) ===")
    print(f"Scale factor: {scale_factor}")
    print()

    print("[1/5] Setting up environment...")
    r = phase_setup_environment(em)
    print(f"      {r['object_count']} objects in {len(ROOMS)} rooms ({r['setup_ms']:.1f} ms)")
    results["setup"] = r

    print("[2/5] Sensor stream (SurprisalGate filtered)...")
    r = phase_sensor_stream_with_surprisal(em, n_sensor_frames)
    print(
        f"      {r['frames_streamed']} frames streamed, "
        f"{r['frames_stored']} stored ({r['storage_rate_pct']:.1f}%), "
        f"{r['frames_filtered']} filtered "
        f"({r['ingest_ms']:.1f} ms)"
    )
    results["sensor_ingest"] = r

    print("[3/5] Recording missions with events...")
    r, missions = phase_record_missions_with_events(em, n_missions)
    print(
        f"      {r['mission_count']} missions, "
        f"{r['event_count']} events injected "
        f"({r['mission_ms']:.1f} ms)"
    )
    results["missions"] = r

    print("[4/5] Mixed queries...")
    r = phase_mixed_queries(em, n_queries, missions)
    for qtype, stats in r.items():
        print(f"      {qtype:15s} count={stats['count']:3d}  p50={stats['p50_ms']:6.2f}ms  p99={stats['p99_ms']:6.2f}ms")
    results["mixed_queries"] = r

    print("[5/5] Event burst stress test...")
    r = phase_write_burst(em, n_burst_ops)
    print(f"      {r['total_ops']} events in {r['burst_write_ms']:.1f} ms ({r['ops_per_sec']:.0f} ops/sec)")
    results["burst_write"] = r

    # Summary
    total_atoms = len(mock_mem.storage._store)
    total_wo = sum(
        len(em.world_object_store.list_by_scene(room, limit=1000))
        for room in ROOMS
    )
    traj_atoms = sum(
        1 for mid in mock_mem.storage._store
        if mock_mem.storage._store[mid].get("metadata", {}).get("embodied_meta", {}).get("physical_type") == "trajectory"
    )
    print(f"\nTotal atoms in store: {total_atoms}")
    print(f"  - Trajectory memories: {traj_atoms}")
    print(f"  - Other atoms:         {total_atoms - traj_atoms}")
    print(f"Total world objects:     {total_wo}")

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Realistic embodied memory benchmark (event-driven)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor (1.0 = default load)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    results = run_realistic(scale_factor=args.scale)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
