"""
Embodied memory performance benchmarks.

Usage:
    python bench_embodied.py --scale 10000 --output results.json
    python bench_embodied.py --scale 100000 --output results.json
    python bench_embodied.py --scale 1000000 --output results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from typing import Any, Dict, List, Tuple

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.schema import initialize_embodied_schema
from powermem.embodied.types import MemoryAction, MemoryAtom, Pose, TemporalInterval, Vec3, WorldObject


class MockStorageAdapter:
    """Lightweight mock for PowerMem core storage."""

    def __init__(self):
        self._store: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def add_memory(self, payload: Dict[str, Any]) -> int:
        mid = self._next_id
        self._next_id += 1
        self._store[mid] = {"id": mid, "content": payload.get("content", ""), "metadata": payload.get("metadata", {})}
        return mid

    def get_memory(self, memory_id: int) -> Any:
        return self._store.get(memory_id)

    def delete_memory(self, memory_id: int, **kwargs) -> bool:
        return self._store.pop(memory_id, None) is not None

    def search_memories(self, **kwargs) -> list:
        limit = kwargs.get("limit", 30)
        results = []
        for mid, item in list(self._store.items())[:limit]:
            results.append({"id": mid, "memory": item["content"], "score": 0.9, "metadata": item.get("metadata", {})})
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


def benchmark_spatial_search(em: EmbodiedMemory, n_atoms: int, n_queries: int = 100) -> Dict[str, float]:
    """Benchmark spatial near search."""
    # Insert random atoms
    for i in range(n_atoms):
        atom = MemoryAtom(
            content=f"atom_{i}",
            spatial=Vec3(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)),
        )
        em.add_atom(atom)

    # Query
    latencies: List[float] = []
    for _ in range(n_queries):
        center = Vec3(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))
        t0 = _now_ms()
        em.search_near(center, radius=2.0, limit=30)
        latencies.append(_now_ms() - t0)

    return {
        "p50_ms": _percentile(latencies, 50),
        "p99_ms": _percentile(latencies, 99),
        "throughput_qps": n_queries / (sum(latencies) / 1000.0) if sum(latencies) > 0 else 0,
    }


def benchmark_world_objects(em: EmbodiedMemory, n_objects: int, n_queries: int = 100) -> Dict[str, float]:
    """Benchmark world object CRUD and scene graph."""
    scene_id = "bench_scene"
    # Insert
    t0 = _now_ms()
    for i in range(n_objects):
        obj = WorldObject(
            obj_id=f"obj_{i}",
            obj_type=random.choice(["box", "sphere", "cylinder"]),
            name=f"object_{i}",
            pose=Pose(position=Vec3(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(0, 2))),
            size=(random.uniform(0.05, 0.3), random.uniform(0.05, 0.3), random.uniform(0.05, 0.3)),
            scene_id=scene_id,
        )
        em.add_world_object(obj)
    insert_time_ms = _now_ms() - t0

    # Search
    latencies: List[float] = []
    for _ in range(n_queries):
        center = Vec3(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(0, 2))
        t0 = _now_ms()
        em.search_world_objects(center, radius=1.0, scene_id=scene_id, limit=30)
        latencies.append(_now_ms() - t0)

    # Scene graph build
    t0 = _now_ms()
    sg = em.get_scene_graph(scene_id)
    sg_build_ms = _now_ms() - t0

    return {
        "insert_total_ms": insert_time_ms,
        "insert_per_ms": insert_time_ms / n_objects if n_objects > 0 else 0,
        "search_p50_ms": _percentile(latencies, 50),
        "search_p99_ms": _percentile(latencies, 99),
        "scene_graph_build_ms": sg_build_ms,
    }


def benchmark_trajectory_similarity(em: EmbodiedMemory, n_trajectories: int, n_queries: int = 20) -> Dict[str, float]:
    """Benchmark DTW trajectory search."""
    # Insert random trajectories
    for i in range(n_trajectories):
        n_wp = random.randint(10, 50)
        waypoints = []
        x, y, z = random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(0, 2)
        for t in range(n_wp):
            waypoints.append((Vec3(x + t * 0.1, y, z), float(t) * 0.5))
        em.record_trajectory(f"traj_{i}", waypoints)

    # Query
    latencies: List[float] = []
    for _ in range(n_queries):
        query_wp = [(Vec3(random.uniform(-5, 5) + t * 0.1, random.uniform(-5, 5), random.uniform(0, 2)), float(t) * 0.5) for t in range(15)]
        t0 = _now_ms()
        em.search_similar_trajectories(query_wp, top_k=5)
        latencies.append(_now_ms() - t0)

    return {
        "p50_ms": _percentile(latencies, 50),
        "p99_ms": _percentile(latencies, 99),
        "throughput_qps": n_queries / (sum(latencies) / 1000.0) if sum(latencies) > 0 else 0,
    }


def run(scale: int) -> Dict[str, Any]:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    initialize_embodied_schema(conn)
    mock_mem = MockMemory()
    em = EmbodiedMemory(memory=mock_mem, db_conn=conn, enable_plugin=False)

    results: Dict[str, Any] = {"scale": scale}

    print(f"=== Spatial search benchmark (scale={scale}) ===")
    results["spatial"] = benchmark_spatial_search(em, n_atoms=scale, n_queries=100)
    print(json.dumps(results["spatial"], indent=2))

    print(f"=== World object benchmark (scale={scale}) ===")
    results["world_objects"] = benchmark_world_objects(em, n_objects=min(scale, 10000), n_queries=100)
    print(json.dumps(results["world_objects"], indent=2))

    print(f"=== Trajectory similarity benchmark (scale={scale}) ===")
    traj_scale = min(scale // 100, 1000)
    results["trajectory"] = benchmark_trajectory_similarity(em, n_trajectories=max(traj_scale, 10), n_queries=20)
    print(json.dumps(results["trajectory"], indent=2))

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Embodied memory benchmarks")
    parser.add_argument("--scale", type=int, default=10000, help="Number of atoms/objects to benchmark")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    results = run(args.scale)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
