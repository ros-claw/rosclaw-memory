# ROSClaw-Memory 整体架构文档

**版本**: v0.2.0  
**日期**: 2026-05-25  
**定位**: PowerMem 之上的具身智能记忆扩展层

---

## 1. 项目概述

ROSClaw-Memory 是为物理 AI 机器人（具身智能体）设计的记忆系统，基于 PowerMem + SeekDB 构建。它不是通用的向量数据库，而是专门为机器人场景优化的**具身记忆系统** —— 支持空间、时间、物理、因果、轨迹等多维度的记忆存储与检索。

核心设计哲学：
- **记忆不是数据的堆积，而是事件的沉淀** — 90%+ 的传感器数据被前端过滤，只有异常、事件和摘要进入记忆
- **空间与时间是一等公民** — 所有记忆原子都携带 3D 坐标和时间区间
- **物理一致性是根基** — 碰撞检测、运动学、场景图都是记忆系统的一部分

---

## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         外部调用方                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ 机器人本体 │  │ 仿真环境  │  │ 数据回放  │  │ 管理后台  │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
└───────┼────────────┼────────────┼────────────┼────────────────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         gRPC 服务层 (port 50051)                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  EmbodiedMemoryService (19 RPCs)                                  │   │
│  │  - AtomCRUD: AddAtom, GetAtom, DeleteAtom                         │   │
│  │  - Search: SearchNear, SearchTemporal, SearchSimilarTrajectories  │   │
│  │  - WorldObject: AddWorldObject, SearchWorldObjects, SceneGraph    │   │
│  │  - Trajectory: RecordTrajectory, SearchTrajectoryNear             │   │
│  │  - Causal: GetCauses, GetEffects                                  │   │
│  │  - Ingest: IngestSensorFrame                                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                         ▲                                               │
│                         │                                               │
│  ┌──────────────────────┴───────────────────────────────────────────┐   │
│  │                    EmbodiedMemory (核心封装类)                      │   │
│  │                                                                    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────────┐  │   │
│  │  │ 写入接口    │  │ 检索接口    │  │ 生命周期管理                │  │   │
│  │  │ add_atom() │  │ search()   │  │ delete_atom()              │  │   │
│  │  │ ingest()   │  │ search_near│  │ stats()                    │  │   │
│  │  │ record_*** │  │ search_temp│  │                            │  │   │
│  │  └────────────┘  └────────────┘  └────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                         │                                               │
└─────────────────────────┼───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         数据模型层                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │ MemoryAtom        │  │ WorldObject       │  │ SpatialRelation   │      │
│  │ ────────────────  │  │ ────────────────  │  │ ────────────────  │      │
│  │ content: str      │  │ obj_id: str       │  │ subject_id: str   │      │
│  │ spatial: Vec3     │  │ obj_type: str     │  │ object_id: str    │      │
│  │ temporal: Interval│  │ pose: Pose        │  │ relation: str     │      │
│  │ perceptual: Snap  │  │ size: Tuple       │  │ confidence: float │      │
│  │ physical: Invar   │  │ scene_id: str     │  └──────────────────┘      │
│  │ uncertainty: Est  │  │ state: str        │                            │
│  │ action: Action    │  │ memory_id: int    │                            │
│  │ embodied_meta: {} │  └──────────────────┘                            │
│  └──────────────────┘                                                   │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   索引层      │  │   存储层      │  │   插件层      │
│              │  │              │  │              │
│ SpatialIndex │  │ SQLite       │  │ EmbodiedInt  │
│ (Voxel Hash) │  │ (开发/测试)   │  │ ligencePlugin│
│              │  │              │  │              │
│ TemporalIndex│  │ SeekDB       │  │ IngestPipe-  │
│ (B-Tree)     │  │ (生产/SeekDB)│  │ line         │
│              │  │              │  │              │
│              │  │ WorldObject- │  │ Surprisal-   │
│              │  │ Store        │  │ Gate         │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 3. 核心组件详解

### 3.1 MemoryAtom — 具身记忆原子

统一的数据单元，所有写入记忆系统的数据都包装为 MemoryAtom。

```
MemoryAtom
├── content: str              # 人类可读的文本描述
├── spatial: Vec3            # 3D 世界坐标
├── temporal: TemporalInterval # 时间区间 [start_sec, end_sec]
├── perceptual: PerceptualSnapshot  # 传感器特征向量
├── physical: PhysicalInvariant     # 物理不变量（质量、DH参数等）
├── uncertainty: UncertaintyEstimate # 不确定性估计
├── affective: AffectiveTag         # 显著性/情感标记
├── action: MemoryAction           # OBSERVE | ACT | PREDICT | CORRECT
├── prediction_error: float        # 预测编码误差
├── causal_parents: List[int]      # 因果父记忆 ID
└── embodied_meta: Dict            # 扩展元数据（trajectory, world_object 等）
```

**工厂方法**:
- `from_observation()` — 从传感器观测创建
- `from_action()` — 从动作执行创建
- `from_trajectory()` — 从轨迹路点创建（含预计算签名）
- `from_world_object()` — 从世界对象创建
- `from_constraint()` — 从物理约束创建

### 3.2 EmbodiedMemory — 核心封装类

对外暴露的单一入口类，内部协调所有子系统。

**写入接口**:
| 方法 | 用途 | 对应 action |
|------|------|------------|
| `add_atom()` | 通用写入 | 任意 |
| `ingest()` | 传感器帧摄入 | OBSERVE |
| `record_action()` | 记录动作执行 | ACT |
| `record_outcome()` | 记录动作结果 | CORRECT |
| `record_trajectory()` | 记录运动轨迹 | ACT |
| `add_world_object()` | 添加世界对象 | OBSERVE |
| `update_world_object_pose()` | 更新对象位姿 | OBSERVE |
| `add_constraint()` | 添加物理约束 | PREDICT |

**检索接口**:
| 方法 | 维度 | 时间复杂度 |
|------|------|-----------|
| `search()` | 语义 + 空间 + 时间 | O(K) |
| `search_near()` | 纯空间范围 | O(1) 体素查询 |
| `search_temporal()` | 纯时间区间 | O(log N) B-Tree |
| `search_similar_trajectories()` | 轨迹相似度 (DTW) | O(M×N) 精排 |
| `search_world_objects()` | 世界对象空间查询 | O(K) |
| `get_scene_graph()` | 场景图 | O(N) |
| `auto_compute_relations()` | 空间关系推理 | O(N²) AABB |
| `get_causes()` / `get_effects()` | 因果图遍历 | O(1) 索引查 |

---

## 4. 存储层架构

### 4.1 双后端设计

| 后端 | 用途 | 特点 |
|------|------|------|
| **SQLite** | 开发/测试/边缘部署 | 零依赖，单文件，单线程写入 |
| **SeekDB/OceanBase** | 生产/云端 | 分布式，高并发，高可用 |

**切换方式**: 仅改变 `db_conn` 参数，上层代码完全透明。

### 4.2 数据库 Schema

```
embodied_memories          # MemoryAtom 扩展表（与 PowerMem memories 1:1）
├── memory_id (PK, FK)     # PowerMem 分配的 Snowflake ID
├── spatial_x/y/z          # 3D 坐标
├── spatial_voxel_key      # Voxel Hash 键
├── spatial_frame_id       # 坐标系（默认 "world"）
├── temporal_start/end     # 时间区间
├── modality               # 传感器模态
├── entity_id              # 物理实体标识
├── physical_type          # 'trajectory' | 'invariant' | 'snapshot' | 'constraint'
├── action_type            # 'observe' | 'act' | 'predict' | 'correct'
├── uncertainty_*          # 不确定性字段
├── salience/valence/arousal # 情感标记
├── embodied_meta (JSON)   # 扩展元数据
└── created_at/updated_at  # 时间戳

embodied_causal_edges      # 因果图边表
├── cause_memory_id (FK)
├── effect_memory_id (FK)
└── relation_type/strength

embodied_predictive_state  # SurprisalGate 持久化状态
├── predictor_id (PK)
├── window_mean/std/count  # Welford 在线统计
└── dynamic_threshold

embodied_world_objects     # 世界对象表
├── obj_id (PK)
├── obj_type/name          # box | sphere | cylinder | mesh
├── pos_x/y/z              # 位置
├── orient_w/x/y/z         # 四元数
├── size_json/color_json   # 几何属性
├── scene_id               # 所属场景
├── parent_obj_id          # 场景图父节点
├── state                  # present | moved | removed | occluded
└── memory_id (FK)         # 链接到 embodied_memories

embodied_spatial_relations # 空间关系表
├── subject_id/object_id (FK)
├── relation               # on | in | next_to | above | below | touching
└── confidence

embodied_physical_models   # 机器人/环境物理模型
├── model_id (PK)
├── model_type             # robot | environment | object
├── joint_names/dh_params  # 运动学参数
├── mass_matrix/coriolis_matrix/gravity_vector
├── collision_geoms        # 碰撞几何体 JSON
└── source_urdf_hash
```

### 4.3 索引设计

```
空间索引:
  idx_emb_spatial_xyz    (spatial_x, spatial_y, spatial_z)   # 精确坐标
  idx_emb_voxel          (spatial_voxel_key)                 # Voxel 桶查
  idx_emb_spatial_frame  (spatial_frame_id)                  # 坐标系过滤

时间索引:
  idx_emb_temp_start     (temporal_start)                    # 区间起始
  idx_emb_temp_cover     (temporal_frame_id, start, end)     # 复合覆盖

语义索引:
  idx_emb_physical_type  (physical_type)                     # 类型过滤
  idx_emb_action         (action_type)                       # 动作过滤
  idx_emb_modality       (modality)                          # 模态过滤
  idx_emb_entity         (entity_id)                         # 实体查

世界对象索引:
  idx_wo_scene           (scene_id)
  idx_wo_parent          (parent_obj_id)                     # 场景图遍历
  idx_wo_pos             (pos_x, pos_y, pos_z)

因果索引:
  idx_causal_pair        (cause_memory_id, effect_memory_id)
```

---

## 5. 索引层详解

### 5.1 SpatialIndex — Voxel Hash

**核心设计**: 内存中的 3D 体素哈希表 + DB 持久化坐标

```
写入流程:
  Vec3(x, y, z) → floor(x/voxel_size) → voxel_key="vx:vy:vz:frame_id"
  → 存入内存 Dict[voxel_key, Set[memory_id]]
  → DB UPDATE embodied_memories SET spatial_voxel_key=...

查询流程:
  1. 计算查询中心所在体素键
  2. 扩展半径 r 覆盖的邻接体素 (r/voxel_size 个桶)
  3. 取所有候选 memory_id 的并集
  4. 精确欧氏距离过滤
  5. 按距离升序返回

复杂度:
  - 插入: O(1)
  - 点查: O(1)
  - 范围查: O(k³) k = r/voxel_size (常数，通常 k≤10)
  - 重建: O(N)
```

### 5.2 TemporalIndex — B-Tree

基于 Allen Interval Algebra（13 种区间拓扑关系）的时间查询。

```
支持的关系:
  OVERLAPS, DURING, STARTS, FINISHES, EQUALS,
  CONTAINS, STARTED_BY, FINISHED_BY, BEFORE, AFTER,
  MEETS, MET_BY, OVERLAPPED_BY

查询模式:
  - query_overlapping(): 任意重叠（最常用）
  - query(): 指定 Allen 关系
```

### 5.3 轨迹相似度 — 签名预过滤 + DTW

```
存储阶段:
  waypoints → trajectory_feature_signature()
  → (duration, total_length, bbox_diagonal, avg_speed,
     dir_x, dir_y, dir_z, waypoint_count)
  → 存入 embodied_meta["trajectory"]["signature"]

查询阶段:
  1. 空间粗筛: spatial_index.query_radius(center, radius, limit=100)
  2. 时间粗筛: temporal_index.query_overlapping(interval)
  3. 签名预过滤: signature_compatible(query_sig, candidate_sig)
     - 时长比、长度比、包围盒比、方向点积、路点数比
     - 通过 → 进入 DTW
     - 拒绝 → 跳过（O(1) 成本）
  4. DTW 精排: dtw_distance_normalized(query, candidate)
  5. 返回 top_k

优化效果: 1000 候选 → 签名过滤后约 50-200 条 → DTW
```

---

## 6. 物理层架构

### 6.1 物理模型存储 (ModelStore)

支持多格式机器人模型解析：

```
URDF ──┐
MJCF ──┼──→ ParseResult ──→ ModelStore.save() ──→ embodied_physical_models
SDF ───┤      (统一中间格式)
XACRO ─┤
USD ───┘
```

存储内容:
- 关节名称、DH 参数、质量矩阵
- 碰撞几何体（AABB、Sphere、Capsule、Box、Mesh）
- 关节限制（位置、速度、力矩）

### 6.2 碰撞检测 (CollisionChecker)

```
场景: check_self_collision(model_id)
流程:
  1. 从 ModelStore 加载模型
  2. 构建碰撞体列表 (CollisionBody[])
  3. forward_kinematics() 计算世界坐标系下的碰撞体位置
  4. 遍历所有碰撞体对
  5. 按类型分发: AABB-AABB, Sphere-Sphere, Sphere-Capsule...
  6. 返回碰撞列表 (body_a, body_b, contact_point, penetration_depth)
```

### 6.3 正运动学 (Kinematics)

```
输入: DH 参数 + 关节角 q[]
输出: 各连杆的世界坐标系位姿
算法: 逐关节累积 4×4 齐次变换矩阵
      T_i = T_{i-1} × DH(d_i, θ_i, a_i, α_i)
```

---

## 7. 世界对象层架构

### 7.1 WorldObjectStore

```
CRUD:
  save(obj: WorldObject)      → obj_id (INSERT/REPLACE)
  load(obj_id)                → WorldObject
  list_by_scene(scene_id)     → List[WorldObject]
  list_by_type(obj_type)      → List[WorldObject]
  update_pose(obj_id, pose)   → bool
  delete(obj_id)              → bool

关系管理:
  add_relation(rel: SpatialRelation)    → int
  get_relations(obj_id, direction)      → List[SpatialRelation]
  get_scene_graph(scene_id)             → Dict[str, List[Relation]]
```

### 7.2 SceneGraph

```
场景图构建:
  1. WorldObjectStore.load_many(scene_id) → 所有对象
  2. 按 parent_obj_id 构建树
  3. compute_relations() → AABB 两两比较
     - "on": A 的底面与 B 的顶面接触 (z_overlap + x/y_within)
     - "in": A 完全在 B 的包围盒内
     - "next_to": 水平距离 < tolerance
     - "above": A 在 B 上方，水平投影重叠
  4. 返回 SpatialRelation 列表
```

**注意**: 当前实现是 O(n²) 两两比较，5x 规模下 p50 从 1.7ms → 37ms。优化方向：空间分桶（按体素分组，只比较同/邻桶对象）。

---

## 8. 插件与管线层

### 8.1 EmbodiedIntelligencePlugin

PowerMem 插件接口的实现，在记忆生命周期钩子中注入具身智能逻辑。

```
on_add(content, metadata):
  1. 惊奇门控评估 → 是否产生记忆
  2. 显著性评估 → salience 打分
  3. 不确定性校验 → confidence < 阈值告警
  4. 自动生成体素键 → compute_voxel_key()
  5. 返回增强后的 metadata

on_get(memory_id, metadata):
  1. 预测编码更新 → 更新内部模型
  2. 显著性衰减 → salience *= (1 - decay_rate)
  3. 物理一致性检查 → 验证 spatial/temporal 一致性

on_search(query, candidates):
  1. 时空相关性重排序 → 优先返回最近空间/时间的
  2. 不确定性加权 → 高不确定性候选降权
```

### 8.2 IngestPipeline — 传感器接入管线

```
传感器帧 (SensorFrame)
  ↓
特征提取 (FeatureExtractor)
  → 统计特征: mean, std, min, max, median, percentile
  → 可选频域特征
  ↓
SurprisalGate.filter_atom()
  → 预测器: ZeroOrderHold | Linear
  → 动态阈值: μ + kσ (默认 k=3)
  → 误差 > 阈值 → 通过
  → 误差 ≤ 阈值 → 丢弃 (99%+ 帧在此被过滤)
  ↓
缓冲/合并
  → 连续同模态帧合并为 TemporalInterval
  → 融合不确定性
  ↓
memory_store(MemoryAtom) → PowerMem
```

**Welford 在线算法**:
```python
count += 1
delta = value - mean
mean += delta / count
delta2 = value - mean
m2 += delta * delta2
std = sqrt(m2 / count)
```

---

## 9. gRPC 服务层

### 9.1 服务定义

```protobuf
service EmbodiedMemoryService {
  // Atom CRUD
  rpc AddAtom(AddAtomRequest) returns (AddAtomResponse);
  rpc GetAtom(GetAtomRequest) returns (GetAtomResponse);
  rpc DeleteAtom(DeleteAtomRequest) returns (DeleteAtomResponse);

  // Search
  rpc SearchNear(SearchNearRequest) returns (SearchNearResponse);
  rpc SearchTemporal(SearchTemporalRequest) returns (SearchTemporalResponse);
  rpc SearchSimilarTrajectories(SearchSimilarTrajectoriesRequest) returns (...);

  // Trajectory
  rpc RecordTrajectory(RecordTrajectoryRequest) returns (RecordTrajectoryResponse);
  rpc SearchTrajectoryNear(SearchTrajectoryNearRequest) returns (...);

  // World Object
  rpc AddWorldObject(AddWorldObjectRequest) returns (AddWorldObjectResponse);
  rpc GetWorldObject(GetWorldObjectRequest) returns (GetWorldObjectResponse);
  rpc SearchWorldObjects(SearchWorldObjectsRequest) returns (...);
  rpc GetSceneGraph(GetSceneGraphRequest) returns (GetSceneGraphResponse);
  rpc ComputeRelations(ComputeRelationsRequest) returns (ComputeRelationsResponse);

  // Causal
  rpc GetCauses(GetCausesRequest) returns (GetCausesResponse);
  rpc GetEffects(GetEffectsRequest) returns (GetEffectsResponse);

  // Ingest
  rpc IngestSensorFrame(IngestSensorFrameRequest) returns (IngestSensorFrameResponse);

  // Stats
  rpc GetStats(GetStatsRequest) returns (GetStatsResponse);
}
```

### 9.2 部署方式

```
# 方式 1: 直接运行
powermem-embodied-server --host 0.0.0.0 --port 50051 --db-path /data/embodied.db

# 方式 2: Docker
docker build -f docker/Dockerfile.embodied -t rosclaw-memory:embodied .
docker run -p 50051:50051 rosclaw-memory:embodied

# 方式 3: Docker Compose
docker-compose -f docker/docker-compose.embodied.yml up
```

---

## 10. 性能基准

### 10.1 组件极限测试 (bench_embodied.py)

| 组件 | 规模 | p50 | p99 | 吞吐 |
|------|------|-----|-----|------|
| 空间搜索 | 1M 原子 | 71.6ms | 78.8ms | 14.1 qps |
| 世界对象插入 | 1M 对象 | 0.63ms/个 | — | — |
| 世界对象搜索 | 1M 对象 | 0.88ms | 0.93ms | — |
| 场景图构建 | 1M 对象 | 150ms | — | — |
| 轨迹 DTW | 1K 轨迹 | 8.1ms | 9.7ms | 121.3 qps |

### 10.2 真实场景测试 (bench_embodied_realistic.py)

**1x 规模负载**: 2000 传感器帧 → 2 条记忆, 50 任务, 183 事件

| 查询类型 | p50 | p99 | 说明 |
|---------|-----|-----|------|
| 空间近邻 | 37.9ms | 39.4ms | Voxel Hash |
| 时间区间 | 0.06ms | 0.35ms | B-Tree |
| 轨迹相似度 | 127.4ms | 136.3ms | 签名+DTW |
| 世界对象 | 0.38ms | 0.82ms | Store 查询 |
| 场景图 | 1.66ms | 3.91ms | AABB O(n²) |
| 因果查询 | 0.03ms | 0.05ms | 直接索引 |
| 事件写入 | — | — | 1704 ops/sec |

**5x 规模**: 10000 帧 → 7 条记忆, 250 任务, 1073 事件, 2870 原子

| 查询类型 | p50 | 趋势 |
|---------|-----|------|
| 空间近邻 | 37.0ms | ✅ 持平 |
| 时间区间 | 0.19ms | ✅ 微增 |
| 轨迹相似度 | 121.4ms | ✅ 持平 |
| 场景图 | **37.4ms** | ⚠️ O(n²) 恶化 |
| 因果查询 | 0.03ms | ✅ 持平 |

---

## 11. 技术栈

| 层级 | 技术 |
|------|------|
| **语言** | Python 3.11+ |
| **核心依赖** | PowerMem, SeekDB (OceanBase), SQLite |
| **gRPC** | grpcio, protobuf |
| **向量运算** | numpy (特征提取) |
| **物理模型解析** | xml.etree (URDF/MJCF/SDF/Xacro/USD) |
| **部署** | Docker, Docker Compose |
| **测试** | pytest, MagicMock |
| **通信** | gRPC (port 50051) |

---

## 12. 已知问题与优化方向

| 优先级 | 问题 | 影响 | 优化方案 |
|--------|------|------|---------|
| 🔴 P0 | SceneGraph O(n²) | 5x 规模下 37ms | 空间分桶：按体素分组，只比较邻桶 |
| 🟡 P1 | 轨迹搜索空间过滤 | 扫描非轨迹原子 | 复合索引: (voxel_key, physical_type) |
| 🟡 P1 | SQLite 单线程写入 | 8K→15K ops/sec 上限 | 批量写入 API + WAL 模式 |
| 🟢 P2 | 缺少 gRPC 压测 | 网络开销未知 | 并发客户端 benchmark |
| 🟢 P2 | Docker 未实际构建 | 可能构建失败 | CI 构建验证 |
| 🟢 P3 | SeekDB 后端未测试 | 生产性能未知 | SeekDB 集成 benchmark |

---

## 13. 模块依赖图

```
embodied_memory.py (中心协调)
    ├── memory_atom.py
    ├── spatial_index.py → VoxelHash
    ├── temporal_index.py
    ├── model_store.py
    ├── world_object_store.py → scene_graph.py
    ├── trajectory_similarity.py
    ├── ingest_pipeline.py → surprisal_gate.py
    ├── embodied_plugin.py
    ├── collision.py → kinematics.py
    ├── physical_model.py
    ├── uncertainty.py
    ├── parsers/ (urdf, mjcf, sdf, xacro, usd)
    ├── schema.py
    └── grpc/ (servicer.py, client.py, server.py, cli.py)
```

---

## 14. 未来路线图

### Phase 8: 性能优化
- [ ] SceneGraph 空间分桶（O(n²) → O(n log n)）
- [ ] 复合索引 (voxel_key + physical_type)
- [ ] 批量写入 API
- [ ] SQLite WAL 模式配置

### Phase 9: 生产就绪
- [ ] SeekDB 后端集成测试
- [ ] gRPC 并发压测
- [ ] Docker 镜像 CI 构建
- [ ] 健康检查端点 (/health)

### Phase 10: 高级功能
- [ ] 轨迹在线学习（从相似轨迹中提取最优路径）
- [ ] 预测编码驱动的主动探索（高不确定性区域优先观测）
- [ ] 多机器人记忆共享（分布式场景图合并）
- [ ] LLM 驱动的记忆摘要与蒸馏

---

*文档生成于 2026-05-25，基于 commit be73ac4*
