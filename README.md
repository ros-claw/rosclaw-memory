# RosClaw Memory

**Embodied Intelligence Memory for Physical AI Robots**

Built on top of [PowerMem](https://github.com/oceanbase/powermem) + embedded SeekDB (OceanBase). Not a replacement — an extension layer that gives robots a brain-like memory system for physical world experiences.

---

## Design Philosophy

- **Brain-like architecture** — Memory is not a database table. It is a spatial-temporal-causal graph with predictive state, surprise detection, and associative retrieval.
- **Zero ROS dependency** — Robot stacks should not be forced into a specific middleware. gRPC + plain Python SDK. Use it from C++, Rust, Go, or Python.
- **First principles** — Every feature starts from "how does a physical agent remember?" rather than "how does a web app cache?"
- **Build ON TOP of PowerMem** — We inherit vector search, full-text retrieval, graph edges, LLM-driven extraction, and Ebbinghaus decay from PowerMem. We add the *embodied* dimension: space, time, physics, and body.

---

## What It Does

RosClaw Memory turns raw robot experiences — sensor frames, trajectories, collisions, constraints, causal outcomes — into queryable, generalizable memory atoms.

```
Sensor Stream → Surprisal Gate → Memory Atom → Spatial Index
                                                    ↓
                                           Temporal Index → Causal Graph
                                                    ↓
                                              EmbodiedMemory (gRPC / Python)
```

---

## Core Capabilities

### 1. Embodied Memory Atom

A unified memory primitive with six facets:

| Facet | Field | Example |
|-------|-------|---------|
| **Spatial** | `Vec3(x, y, z)` + frame_id | Where did this happen? |
| **Temporal** | `TemporalInterval(start, end)` + frame_id | When did this happen? |
| **Perceptual** | `Modality` (RGB, depth, lidar, tactile, audio, proprioception) | What was sensed? |
| **Physical** | `CollisionBody`, `JointLimit`, `PhysicalConstraint` | What body state was involved? |
| **Uncertainty** | `prediction_error`, `information_gain` | How surprising was this? |
| **Affective** | `affective_tags` (curiosity, pain, satisfaction) | What was the valence? |

```python
from powermem.embodied import MemoryAtom, Vec3, TemporalInterval, Modality

atom = MemoryAtom(
    content="grasped the red cube",
    spatial=Vec3(0.5, -0.2, 0.1),
    temporal=TemporalInterval(12.5, 14.0, frame_id="session_01"),
    modality=Modality.TACTILE,
    prediction_error=0.85,  # high surprise → strong memory encoding
)
```

### 2. Multi-Format Robot Model Parsing

Load robot descriptions without ROS:

```python
from powermem.embodied.parsers import parse_model

result = parse_model(open("franka.urdf").read())   # URDF
result = parse_model(open("anymal.xml").read())    # MJCF
result = parse_model(open("scene.usda").read())    # OpenUSD
```

Supported formats: **URDF, MJCF, SDF, Xacro, OpenUSD** (lazy-loaded modules).

### 3. Collision Detection & Kinematics

```python
from powermem.embodied.embodied_memory import EmbodiedMemory

em = EmbodiedMemory(memory=pmem, db_conn=conn)

# Forward kinematics
fk = em.forward_kinematics("panda", joint_angles=[0, -0.5, 0, -1.8, 0, 1.5, 0])

# Self-collision check
pairs = em.check_self_collision("panda")
# → [CollisionPair(link_a="panda_link4", link_b="panda_link6", distance=-0.012)]
```

Collision geometry: Sphere, Capsule, AABB. Broad-phase AABB tree + analytical narrow-phase.

### 4. Temporal Reasoning (Allen Interval Algebra)

13 interval relations for causal and temporal queries:

```python
from powermem.embodied.types import TemporalInterval, IntervalRelation

# Find all memories that happened DURING a specific session
results = em.search_temporal(
    interval=TemporalInterval(10.0, 20.0),
    relation=IntervalRelation.DURING,
)
```

### 5. Spatial Indexing (Voxel Hash)

O(1) spatial lookup for memory atoms:

```python
# Query all memories within 0.5m of a point
neighbors = em.search_near(center=Vec3(1.0, 0.0, 0.0), radius=0.5)
```

### 6. Trajectory Similarity Search (DTW)

Find historically similar trajectories — essential for "have I done this grasp before?"

```python
waypoints = [
    (Vec3(0, 0, 0), 0.0),
    (Vec3(0.1, 0.2, 0.3), 1.0),
    (Vec3(0.2, 0.4, 0.5), 2.0),
]

# Record and later retrieve by shape similarity
mid = em.record_trajectory("approach from left", waypoints)

similar = em.search_similar_trajectories(
    query_waypoints=new_waypoints,
    top_k=5,
    max_dtw_distance=0.3,
)
# → [(MemoryAtom, dtw_distance), ...]
```

- **Coarse filter** — trajectory feature signature (duration, length, bounding box, principal direction)
- **Fine ranking** — Dynamic Time Warping (DTW) with optional Sakoe-Chiba bandwidth
- **Normalized distance** — comparable across different-length trajectories

### 7. Causal Graph

Link actions to outcomes:

```python
cause_id = em.add_atom(MemoryAtom(content="motor overheated"))
effect_id = em.add_atom(MemoryAtom(content="gripper slipped", causal_parents=[cause_id]))

causes = em.get_causes(effect_id)   # → ["motor overheated"]
effects = em.get_effects(cause_id)  # → ["gripper slipped"]
```

### 8. Predictive State & Surprisal Gate

Only surprising experiences become long-term memory. A sliding Welford window computes a 3-sigma dynamic threshold.

```python
from powermem.embodied.ingest_pipeline import SensorFrame, Modality

frame = SensorFrame(
    modality=Modality.PROPRICEPTION,
    timestamp_sec=10.5,
    data=[0.12, 0.34, 0.56],
)
mid = em.ingest(frame, content="joint torque anomaly")
# If prediction_error > 3σ, it stores; otherwise it is gated out.
```

### 9. Physical Constraints as Memory

```python
constraint = PhysicalConstraint(
    constraint_type="no_fly_zone",
    region=AABB(min=Vec3(0,0,0), max=Vec3(1,1,1)),
)
em.add_constraint(constraint)
```

Constraints are stored as MemoryAtoms and indexed by spatial region.

### 10. gRPC Service

Expose the full `EmbodiedMemory` API to C++, Rust, Go, or any gRPC-capable stack:

```python
from powermem.embodied.grpc.server import serve

server = serve(memory=pmem, db_conn=conn, port=50051)
server.wait_for_termination()
```

Python client helper:

```python
from powermem.embodied.grpc.client import EmbodiedMemoryClient

with EmbodiedMemoryClient("localhost:50051") as client:
    mid = client.add_atom(atom)
    results = client.search_similar_trajectories(waypoints, top_k=5)
```

Supported RPCs: `AddAtom`, `GetAtom`, `DeleteAtom`, `Search`, `SearchNear`, `SearchTemporal`, `RecordTrajectory`, `SearchSimilarTrajectories`, `IngestSensorFrame`, `SaveModel`, `CheckSelfCollision`, `GetCauses`, `GetEffects`, `GetStats`.

---

## Quick Start

```bash
pip install powermem
```

```python
import sqlite3
from powermem.core.memory import Memory, auto_config
from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.schema import initialize_embodied_schema

# 1. PowerMem core
pmem = Memory(config=auto_config())

# 2. SQLite-backed embodied layer
conn = sqlite3.connect("embodied.db")
initialize_embodied_schema(conn)

# 3. Embodied memory
em = EmbodiedMemory(memory=pmem, db_conn=conn)

# 4. Add an experience
atom = MemoryAtom(
    content="object fell from table",
    spatial=Vec3(1.0, 0.5, 0.0),
    temporal=TemporalInterval(5.0, 6.5),
    prediction_error=2.1,
)
mid = em.add_atom(atom)

# 5. Query spatially
for atom in em.search_near(Vec3(1.0, 0.5, 0.0), radius=1.0):
    print(atom.content)
```

See [`docs/`](docs/) and [`tests/unit/`](tests/unit/) for full examples.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RosClaw Memory                           │
├─────────────────────────────────────────────────────────────┤
│  gRPC / Python SDK                                          │
├─────────────────────────────────────────────────────────────┤
│  EmbodiedMemory                                              │
│  ├── MemoryAtom (spatial · temporal · perceptual · physical)│
│  ├── SpatialIndex (VoxelHash)                               │
│  ├── TemporalIndex (Allen Interval Algebra)                 │
│  ├── CausalGraph (action → outcome edges)                   │
│  ├── TrajectoryStore (DTW similarity)                       │
│  ├── IngestPipeline (Surprisal Gate)                        │
│  ├── PhysicalModel (FK, collision, constraints)             │
│  └── PredictiveState (Welford sliding window)               │
├─────────────────────────────────────────────────────────────┤
│  PowerMem Core                                               │
│  ├── Vector + Full-text + Graph retrieval                   │
│  ├── LLM-driven extraction & distillation                   │
│  └── Ebbinghaus time decay                                  │
├─────────────────────────────────────────────────────────────┤
│  Storage                                                     │
│  ├── SeekDB (embedded OceanBase)  ←  default                │
│  ├── PostgreSQL / pgvector                                   │
│  └── SQLite                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Context

- **Upstream:** [oceanbase/powermem](https://github.com/oceanbase/powermem) — general-purpose persistent memory for AI agents.
- **This repo:** The embodied extension — everything needed for physical AI (robots, embodied agents, sim-to-real) to remember, reason, and generalize from real-world interaction.
- **License:** Apache 2.0 (same as PowerMem).

---

## Why "RosClaw"?

A claw is a physical end-effector. ROS is the lingua franca of robotics. RosClaw Memory is the *memory layer* that physical agents carry with them — no middleware required, just a brain-like store of what the body has done and felt.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
