# RosClaw Memory

**面向物理 AI 机器人的具身智能记忆系统**

基于 [PowerMem](https://github.com/oceanbase/powermem) + 嵌入式 SeekDB (OceanBase) 构建。不是替代，而是一个扩展层——为机器人提供脑式的物理世界经验记忆系统。

---

## 设计理念

- **类脑架构** —— 记忆不是数据库表，而是一个时空因果图，带有预测状态、惊奇检测与联想检索。
- **零 ROS 依赖** —— 不强绑定任何中间件。gRPC + 纯 Python SDK，可在 C++、Rust、Go、Python 中调用。
- **第一性原理** —— 每个功能都从"物理智能体如何记忆？"出发，而不是"Web 应用如何缓存？"
- **构建于 PowerMem 之上** —— 我们继承向量检索、全文检索、图边、LLM 驱动抽取与艾宾浩斯衰减。我们新增的是 *具身* 维度：空间、时间、物理与身体。

---

## 它能做什么

RosClaw Memory 将原始机器人经验——传感器帧、轨迹、碰撞、约束、因果结果——转化为可查询、可泛化的记忆原子。

```
传感器流 → 惊奇门控 → 记忆原子 → 空间索引
                                              ↓
                                     时间索引 → 因果图
                                              ↓
                                    具身记忆层 (gRPC / Python)
```

---

## 核心能力

### 1. 具身记忆原子 (MemoryAtom)

统一记忆原语，包含六个维度：

| 维度 | 字段 | 示例 |
|------|------|------|
| **空间** | `Vec3(x, y, z)` + frame_id | 这件事发生在哪里？ |
| **时间** | `TemporalInterval(start, end)` + frame_id | 这件事发生在何时？ |
| **感知** | `Modality` (RGB、深度、激光雷达、触觉、音频、本体感觉) | 感知到了什么？ |
| **物理** | `CollisionBody`、`JointLimit`、`PhysicalConstraint` | 涉及什么身体状态？ |
| **不确定性** | `prediction_error`、`information_gain` | 这件事有多令人惊讶？ |
| **情感** | `affective_tags` (好奇、疼痛、满足) | 它的情感效价是什么？ |

```python
from powermem.embodied import MemoryAtom, Vec3, TemporalInterval, Modality

atom = MemoryAtom(
    content="抓起了红色方块",
    spatial=Vec3(0.5, -0.2, 0.1),
    temporal=TemporalInterval(12.5, 14.0, frame_id="session_01"),
    modality=Modality.TACTILE,
    prediction_error=0.85,  # 高惊讶 → 强记忆编码
)
```

### 2. 多格式机器人模型解析

无需 ROS 即可加载机器人描述文件：

```python
from powermem.embodied.parsers import parse_model

result = parse_model(open("franka.urdf").read())   # URDF
result = parse_model(open("anymal.xml").read())    # MJCF
result = parse_model(open("scene.usda").read())    # OpenUSD
```

支持格式：**URDF、MJCF、SDF、Xacro、OpenUSD**（模块懒加载）。

### 3. 碰撞检测与运动学

```python
from powermem.embodied.embodied_memory import EmbodiedMemory

em = EmbodiedMemory(memory=pmem, db_conn=conn)

# 正运动学
fk = em.forward_kinematics("panda", joint_angles=[0, -0.5, 0, -1.8, 0, 1.5, 0])

# 自碰撞检测
pairs = em.check_self_collision("panda")
# → [CollisionPair(link_a="panda_link4", link_b="panda_link6", distance=-0.012)]
```

碰撞几何：球体、胶囊体、AABB。Broad-phase AABB 树 + 解析 narrow-phase。

### 4. 时间推理 (Allen 区间代数)

13 种区间关系，用于因果和时间查询：

```python
from powermem.embodied.types import TemporalInterval, IntervalRelation

# 查找发生在某段时间内的所有记忆
results = em.search_temporal(
    interval=TemporalInterval(10.0, 20.0),
    relation=IntervalRelation.DURING,
)
```

### 5. 空间索引 (Voxel Hash)

记忆原子的 O(1) 空间查询：

```python
# 查询某点半径 0.5m 内的所有记忆
neighbors = em.search_near(center=Vec3(1.0, 0.0, 0.0), radius=0.5)
```

### 6. 轨迹相似性检索 (DTW)

查找历史上相似的轨迹——对"我以前做过这个抓取动作吗？"至关重要。

```python
waypoints = [
    (Vec3(0, 0, 0), 0.0),
    (Vec3(0.1, 0.2, 0.3), 1.0),
    (Vec3(0.2, 0.4, 0.5), 2.0),
]

# 记录轨迹，之后按形状相似度检索
mid = em.record_trajectory("从左侧接近", waypoints)

similar = em.search_similar_trajectories(
    query_waypoints=new_waypoints,
    top_k=5,
    max_dtw_distance=0.3,
)
# → [(MemoryAtom, dtw_distance), ...]
```

- **粗过滤** —— 轨迹特征签名（持续时间、长度、包围盒、主方向）
- **精排序** —— 动态时间规整 (DTW)，可选 Sakoe-Chiba 带宽约束
- **归一化距离** —— 不同长度轨迹之间可比

### 7. 因果图

将动作与结果关联：

```python
cause_id = em.add_atom(MemoryAtom(content="电机过热"))
effect_id = em.add_atom(MemoryAtom(content="夹爪打滑", causal_parents=[cause_id]))

causes = em.get_causes(effect_id)   # → ["电机过热"]
effects = em.get_effects(cause_id)  # → ["夹爪打滑"]
```

### 8. 预测状态与惊奇门控

只有令人惊讶的经验才会成为长期记忆。滑动 Welford 窗口计算 3-sigma 动态阈值。

```python
from powermem.embodied.ingest_pipeline import SensorFrame, Modality

frame = SensorFrame(
    modality=Modality.PROPRICEPTION,
    timestamp_sec=10.5,
    data=[0.12, 0.34, 0.56],
)
mid = em.ingest(frame, content="关节力矩异常")
# 若 prediction_error > 3σ，则存储；否则被门控过滤
```

### 9. 物理约束记忆化

```python
constraint = PhysicalConstraint(
    constraint_type="no_fly_zone",
    region=AABB(min=Vec3(0,0,0), max=Vec3(1,1,1)),
)
em.add_constraint(constraint)
```

约束以 MemoryAtom 形式存储，并按空间区域索引。

### 10. gRPC 服务

将完整 `EmbodiedMemory` API 暴露给 C++、Rust、Go 或任何支持 gRPC 的栈：

```python
from powermem.embodied.grpc.server import serve

server = serve(memory=pmem, db_conn=conn, port=50051)
server.wait_for_termination()
```

Python 客户端助手：

```python
from powermem.embodied.grpc.client import EmbodiedMemoryClient

with EmbodiedMemoryClient("localhost:50051") as client:
    mid = client.add_atom(atom)
    results = client.search_similar_trajectories(waypoints, top_k=5)
```

支持的 RPC：`AddAtom`、`GetAtom`、`DeleteAtom`、`Search`、`SearchNear`、`SearchTemporal`、`RecordTrajectory`、`SearchSimilarTrajectories`、`IngestSensorFrame`、`SaveModel`、`CheckSelfCollision`、`GetCauses`、`GetEffects`、`GetStats`。

---

## 快速开始

```bash
pip install powermem
```

```python
import sqlite3
from powermem.core.memory import Memory, auto_config
from powermem.embodied.embodied_memory import EmbodiedMemory
from powermem.embodied.schema import initialize_embodied_schema

# 1. PowerMem 核心
pmem = Memory(config=auto_config())

# 2. SQLite 具身层
conn = sqlite3.connect("embodied.db")
initialize_embodied_schema(conn)

# 3. 具身记忆
em = EmbodiedMemory(memory=pmem, db_conn=conn)

# 4. 添加一条经验
atom = MemoryAtom(
    content="物体从桌上掉落",
    spatial=Vec3(1.0, 0.5, 0.0),
    temporal=TemporalInterval(5.0, 6.5),
    prediction_error=2.1,
)
mid = em.add_atom(atom)

# 5. 空间查询
for atom in em.search_near(Vec3(1.0, 0.5, 0.0), radius=1.0):
    print(atom.content)
```

完整示例见 [`docs/`](docs/) 和 [`tests/unit/`](tests/unit/)。

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     RosClaw Memory                           │
├─────────────────────────────────────────────────────────────┤
│  gRPC / Python SDK                                          │
├─────────────────────────────────────────────────────────────┤
│  EmbodiedMemory                                              │
│  ├── MemoryAtom (空间 · 时间 · 感知 · 物理)                 │
│  ├── SpatialIndex (VoxelHash)                               │
│  ├── TemporalIndex (Allen 区间代数)                         │
│  ├── CausalGraph (动作 → 结果 边)                           │
│  ├── TrajectoryStore (DTW 相似度)                           │
│  ├── IngestPipeline (惊奇门控)                              │
│  ├── PhysicalModel (正运动学、碰撞、约束)                   │
│  └── PredictiveState (Welford 滑动窗口)                     │
├─────────────────────────────────────────────────────────────┤
│  PowerMem Core                                               │
│  ├── 向量 + 全文 + 图检索                                   │
│  ├── LLM 驱动抽取与蒸馏                                     │
│  └── 艾宾浩斯时间衰减                                       │
├─────────────────────────────────────────────────────────────┤
│  Storage                                                     │
│  ├── SeekDB (嵌入式 OceanBase)  ←  默认                     │
│  ├── PostgreSQL / pgvector                                   │
│  └── SQLite                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目背景

- **上游项目：** [oceanbase/powermem](https://github.com/oceanbase/powermem) —— 面向 AI 智能体的通用持久化记忆。
- **本仓库：** 具身扩展层——为物理 AI（机器人、具身智能体、sim-to-real）提供记忆、推理与泛化真实世界交互所需的一切。
- **许可证：** Apache 2.0（与 PowerMem 一致）。

---

## 为什么叫 "RosClaw"？

Claw（爪）是物理末端执行器。ROS 是机器人领域的事实标准。RosClaw Memory 是物理智能体随身携带的 *记忆层*——不需要中间件，只是一个关于身体做过什么、感受过什么的脑式存储。

---

## 许可证

Apache License 2.0 —— 详见 [LICENSE](LICENSE)。
