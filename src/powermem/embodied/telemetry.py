"""
MemoryTelemetry — cache hit/miss and query latency metrics

Tracks operational metrics for EmbodiedMemory subsystems:
- Atom cache hit/miss rate
- WorldObject cache hit/miss rate
- SceneGraph cache hit/miss rate
- ModelStore cache hit/miss rate
- Query latencies (spatial, temporal, trajectory, atom operations)

Thread-safe: uses threading.Lock for counter updates.

Usage:
    telemetry = MemoryTelemetry()
    em = EmbodiedMemory(..., telemetry=telemetry)
    ...
    stats = telemetry.snapshot()
    print(stats["atom_cache_hit_rate"])  # 0.85
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class _CacheCounters:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        t = self.total
        return self.hits / t if t > 0 else 0.0


@dataclass
class _LatencyCounters:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        if ms < self.min_ms:
            self.min_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.count > 0 else 0.0,
            "max_ms": round(self.max_ms, 3),
        }


class MemoryTelemetry:
    """Collects cache and query metrics for EmbodiedMemory.

    Pass to EmbodiedMemory via `telemetry=telemetry` kwarg.
    Access metrics via `snapshot()` or Prometheus-format `prometheus_metrics()`.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._lock = threading.Lock()

        # Cache counters
        self._atom_cache = _CacheCounters()
        self._world_object_cache = _CacheCounters()
        self._scene_graph_cache = _CacheCounters()
        self._model_cache = _CacheCounters()
        self._collision_cache = _CacheCounters()

        # Latency counters
        self._spatial_query = _LatencyCounters()
        self._temporal_query = _LatencyCounters()
        self._trajectory_query = _LatencyCounters()
        self._atom_get = _LatencyCounters()
        self._atom_add = _LatencyCounters()
        self._scene_graph_build = _LatencyCounters()

        # Daemon stats snapshot (updated on demand)
        self._daemon_snapshot: Dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Cache instrumentation
    # ------------------------------------------------------------------

    def record_cache_hit(self, cache_name: str) -> None:
        """Record a cache hit. cache_name ∈ {atom, world_object, scene_graph, model, collision}"""
        if not self._enabled:
            return
        with self._lock:
            c = self._get_cache(cache_name)
            if c is not None:
                c.hits += 1

    def record_cache_miss(self, cache_name: str) -> None:
        """Record a cache miss."""
        if not self._enabled:
            return
        with self._lock:
            c = self._get_cache(cache_name)
            if c is not None:
                c.misses += 1

    def _get_cache(self, name: str) -> Optional[_CacheCounters]:
        return {
            "atom": self._atom_cache,
            "world_object": self._world_object_cache,
            "scene_graph": self._scene_graph_cache,
            "model": self._model_cache,
            "collision": self._collision_cache,
        }.get(name)

    # ------------------------------------------------------------------
    # Latency instrumentation
    # ------------------------------------------------------------------

    def record_latency(self, operation: str, ms: float) -> None:
        """Record a latency sample.

        operation ∈ {spatial_query, temporal_query, trajectory_query,
                     atom_get, atom_add, scene_graph_build}
        """
        if not self._enabled:
            return
        with self._lock:
            c = self._get_latency(operation)
            if c is not None:
                c.record(ms)

    def _get_latency(self, name: str) -> Optional[_LatencyCounters]:
        return {
            "spatial_query": self._spatial_query,
            "temporal_query": self._temporal_query,
            "trajectory_query": self._trajectory_query,
            "atom_get": self._atom_get,
            "atom_add": self._atom_add,
            "scene_graph_build": self._scene_graph_build,
        }.get(name)

    # ------------------------------------------------------------------
    # Daemon stats injection
    # ------------------------------------------------------------------

    def update_daemon_snapshot(self, stats: Dict[str, Any]) -> None:
        """Update cached daemon stats (called by EmbodiedMemory.get_daemon_stats)."""
        with self._lock:
            self._daemon_snapshot = dict(stats)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a complete metrics snapshot as a dict.

        Keys:
        - atom_cache_hits/misses/hit_rate/total
        - world_object_cache_*
        - scene_graph_cache_*
        - model_cache_*
        - collision_cache_*
        - spatial_query/temporal_query/trajectory_query/atom_get/atom_add/scene_graph_build
          (each: count, total_ms, avg_ms, min_ms, max_ms)
        - daemon: latest daemon stats
        """
        with self._lock:
            return {
                # Cache stats
                "atom_cache_hits": self._atom_cache.hits,
                "atom_cache_misses": self._atom_cache.misses,
                "atom_cache_hit_rate": round(self._atom_cache.hit_rate, 4),
                "atom_cache_total": self._atom_cache.total,
                "world_object_cache_hits": self._world_object_cache.hits,
                "world_object_cache_misses": self._world_object_cache.misses,
                "world_object_cache_hit_rate": round(self._world_object_cache.hit_rate, 4),
                "world_object_cache_total": self._world_object_cache.total,
                "scene_graph_cache_hits": self._scene_graph_cache.hits,
                "scene_graph_cache_misses": self._scene_graph_cache.misses,
                "scene_graph_cache_hit_rate": round(self._scene_graph_cache.hit_rate, 4),
                "scene_graph_cache_total": self._scene_graph_cache.total,
                "model_cache_hits": self._model_cache.hits,
                "model_cache_misses": self._model_cache.misses,
                "model_cache_hit_rate": round(self._model_cache.hit_rate, 4),
                "model_cache_total": self._model_cache.total,
                "collision_cache_hits": self._collision_cache.hits,
                "collision_cache_misses": self._collision_cache.misses,
                "collision_cache_hit_rate": round(self._collision_cache.hit_rate, 4),
                "collision_cache_total": self._collision_cache.total,
                # Latency stats
                "spatial_query": self._spatial_query.to_dict(),
                "temporal_query": self._temporal_query.to_dict(),
                "trajectory_query": self._trajectory_query.to_dict(),
                "atom_get": self._atom_get.to_dict(),
                "atom_add": self._atom_add.to_dict(),
                "scene_graph_build": self._scene_graph_build.to_dict(),
                # Daemon
                "daemon": dict(self._daemon_snapshot),
            }

    def prometheus_metrics(self) -> str:
        """Return metrics in Prometheus text exposition format.

        Suitable for scraping by Prometheus or exposition via an HTTP endpoint.
        """
        with self._lock:
            lines = []
            lines.append("# HELP powermem_cache_hits_total Total cache hits by cache name")
            lines.append("# TYPE powermem_cache_hits_total counter")
            for name, c in [
                ("atom", self._atom_cache),
                ("world_object", self._world_object_cache),
                ("scene_graph", self._scene_graph_cache),
                ("model", self._model_cache),
                ("collision", self._collision_cache),
            ]:
                lines.append(f'powermem_cache_hits_total{{cache="{name}"}} {c.hits}')

            lines.append("# HELP powermem_cache_misses_total Total cache misses by cache name")
            lines.append("# TYPE powermem_cache_misses_total counter")
            for name, c in [
                ("atom", self._atom_cache),
                ("world_object", self._world_object_cache),
                ("scene_graph", self._scene_graph_cache),
                ("model", self._model_cache),
                ("collision", self._collision_cache),
            ]:
                lines.append(f'powermem_cache_misses_total{{cache="{name}"}} {c.misses}')

            lines.append("# HELP powermem_query_duration_ms Query latency stats")
            lines.append("# TYPE powermem_query_duration_ms summary")
            for op, c in [
                ("spatial", self._spatial_query),
                ("temporal", self._temporal_query),
                ("trajectory", self._trajectory_query),
                ("atom_get", self._atom_get),
                ("atom_add", self._atom_add),
                ("scene_graph_build", self._scene_graph_build),
            ]:
                lines.append(f'powermem_query_duration_ms_count{{op="{op}"}} {c.count}')
                lines.append(f'powermem_query_duration_ms_sum{{op="{op}"}} {c.total_ms:.3f}')

            return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all counters (useful for benchmarking intervals)."""
        with self._lock:
            for c in [
                self._atom_cache,
                self._world_object_cache,
                self._scene_graph_cache,
                self._model_cache,
                self._collision_cache,
            ]:
                c.hits = 0
                c.misses = 0
            for c in [
                self._spatial_query,
                self._temporal_query,
                self._trajectory_query,
                self._atom_get,
                self._atom_add,
                self._scene_graph_build,
            ]:
                c.count = 0
                c.total_ms = 0.0
                c.min_ms = float("inf")
                c.max_ms = 0.0
            self._daemon_snapshot = {}


class _Timer:
    """Context manager for timing an operation, then recording to telemetry.

    Usage:
        with _Timer(telemetry, "spatial_query"):
            # do spatial query
    """
    __slots__ = ("_telemetry", "_operation", "_start")

    def __init__(self, telemetry: Optional[MemoryTelemetry], operation: str):
        self._telemetry = telemetry
        self._operation = operation

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._telemetry is not None and self._telemetry.enabled:
            elapsed_ms = (time.perf_counter() - self._start) * 1000.0
            self._telemetry.record_latency(self._operation, elapsed_ms)
        return False  # don't suppress exceptions
