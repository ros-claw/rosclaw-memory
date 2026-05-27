"""
ROSClaw-Memory 具身扩展 Schema — DDL 与初始化

本模块定义了在 PowerMem 之上新增的具身智能专用表结构。
核心设计原则：
1. 不修改 PowerMem 的 memories 主表，通过 `memory_id` 外键关联
2. 空间/时间索引使用 SeekDB 的 B-Tree + 虚拟列能力
3. 所有表均为幂等创建（IF NOT EXISTS）
"""

from __future__ import annotations

import logging
from typing import List

try:
    import pyseekdb
    _HAS_SEEKDB = True
except ImportError:
    _HAS_SEEKDB = False

logger = logging.getLogger(__name__)


# --- 具身记忆扩展表 ---

# 存储 MemoryAtom 的具身专属字段，与 PowerMem memories 表 1:1
_EMBODIED_MEMORIES_DDL = """
CREATE TABLE IF NOT EXISTS embodied_memories (
    memory_id           BIGINT PRIMARY KEY,
    -- 空间（虚拟列将在下方索引中引用）
    spatial_x           DOUBLE,
    spatial_y           DOUBLE,
    spatial_z           DOUBLE,
    spatial_voxel_key   VARCHAR(64),
    spatial_frame_id    VARCHAR(64) DEFAULT 'world',
    -- 时间
    temporal_start      DOUBLE,
    temporal_end        DOUBLE,
    temporal_frame_id   VARCHAR(64) DEFAULT 'wall_clock',
    -- 感知
    modality            VARCHAR(32) DEFAULT 'rgb',
    feature_vec_hash    VARCHAR(64),      -- feature_vec 的哈希，用于去重
    raw_data_hash       VARCHAR(64),      -- 原始传感器数据的哈希
    -- 物理
    entity_id           VARCHAR(128),     -- 物体/连杆标识
    physical_type       VARCHAR(32),      -- 'invariant' | 'snapshot' | 'constraint'
    -- 不确定性
    uncertainty_type    VARCHAR(32) DEFAULT 'aleatoric',
    uncertainty_std     DOUBLE DEFAULT 0.0,
    uncertainty_confidence DOUBLE DEFAULT 1.0,
    -- 情感/显著性
    salience            DOUBLE DEFAULT 0.5,
    valence             DOUBLE DEFAULT 0.0,
    arousal             DOUBLE DEFAULT 0.0,
    -- 动作与预测编码
    action_type         VARCHAR(32) DEFAULT 'observe',
    prediction_error    DOUBLE DEFAULT 0.0,   -- 预测误差（Surprisal Gate 用）
    -- 元数据（JSON）
    embodied_meta       TEXT DEFAULT '{}',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
)
"""

# --- 空间索引 ---

_SPATIAL_INDEXES: List[str] = [
    # B-Tree 精确坐标查询
    "CREATE INDEX IF NOT EXISTS idx_emb_spatial_xyz ON embodied_memories(spatial_x, spatial_y, spatial_z)",
    # Voxel key 快速桶查询
    "CREATE INDEX IF NOT EXISTS idx_emb_voxel ON embodied_memories(spatial_voxel_key)",
    # 范围查询（圆柱形范围搜索的前置条件）
    "CREATE INDEX IF NOT EXISTS idx_emb_spatial_x ON embodied_memories(spatial_x)",
    "CREATE INDEX IF NOT EXISTS idx_emb_spatial_y ON embodied_memories(spatial_y)",
    "CREATE INDEX IF NOT EXISTS idx_emb_spatial_z ON embodied_memories(spatial_z)",
    # frame_id 过滤
    "CREATE INDEX IF NOT EXISTS idx_emb_spatial_frame ON embodied_memories(spatial_frame_id)",
]

# --- 时间索引 ---

_TEMPORAL_INDEXES: List[str] = [
    # 时间区间起始点
    "CREATE INDEX IF NOT EXISTS idx_emb_temp_start ON embodied_memories(temporal_start)",
    # 时间区间结束点
    "CREATE INDEX IF NOT EXISTS idx_emb_temp_end ON embodied_memories(temporal_end)",
    # 复合索引：frame + start（最常用查询模式）
    "CREATE INDEX IF NOT EXISTS idx_emb_temp_frame_start ON embodied_memories(temporal_frame_id, temporal_start)",
    # 覆盖查询：frame + start + end
    "CREATE INDEX IF NOT EXISTS idx_emb_temp_cover ON embodied_memories(temporal_frame_id, temporal_start, temporal_end)",
]

# --- 语义/业务索引 ---

_SEMANTIC_INDEXES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_emb_modality ON embodied_memories(modality)",
    "CREATE INDEX IF NOT EXISTS idx_emb_entity ON embodied_memories(entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_emb_physical_type ON embodied_memories(physical_type)",
    "CREATE INDEX IF NOT EXISTS idx_emb_uncertainty_type ON embodied_memories(uncertainty_type)",
    "CREATE INDEX IF NOT EXISTS idx_emb_salience ON embodied_memories(salience)",
    "CREATE INDEX IF NOT EXISTS idx_emb_action ON embodied_memories(action_type)",
    "CREATE INDEX IF NOT EXISTS idx_emb_raw_hash ON embodied_memories(raw_data_hash) WHERE raw_data_hash IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_emb_feature_hash ON embodied_memories(feature_vec_hash) WHERE feature_vec_hash IS NOT NULL",
]

# --- 因果图边表（轻量级，替代完整图数据库） ---

_CAUSAL_EDGES_DDL = """
CREATE TABLE IF NOT EXISTS embodied_causal_edges (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    cause_memory_id     BIGINT NOT NULL,
    effect_memory_id    BIGINT NOT NULL,
    relation_type       VARCHAR(32) DEFAULT 'causal',  -- causal | temporal | spatial | predictive
    strength            DOUBLE DEFAULT 1.0,            -- 边强度 [0, 1]
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cause_memory_id) REFERENCES embodied_memories(memory_id) ON DELETE CASCADE,
    FOREIGN KEY (effect_memory_id) REFERENCES embodied_memories(memory_id) ON DELETE CASCADE
)
"""

_CAUSAL_INDEXES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_causal_cause ON embodied_causal_edges(cause_memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_causal_effect ON embodied_causal_edges(effect_memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_causal_relation ON embodied_causal_edges(relation_type)",
    "CREATE INDEX IF NOT EXISTS idx_causal_pair ON embodied_causal_edges(cause_memory_id, effect_memory_id)",
]

# --- 预测编码状态表（Surprisal Gate 的在线状态） ---

_PREDICTIVE_STATE_DDL = """
CREATE TABLE IF NOT EXISTS embodied_predictive_state (
    predictor_id        VARCHAR(128) PRIMARY KEY,     -- 如 "rgb_cam_front", "joint_6dof"
    last_prediction     TEXT,                          -- JSON 编码的上次预测
    last_update_sec     DOUBLE DEFAULT 0.0,
    dynamic_threshold   DOUBLE DEFAULT 0.0,            -- 当前 3-sigma 动态阈值
    window_mean         DOUBLE DEFAULT 0.0,            -- 滑动窗口均值
    window_std          DOUBLE DEFAULT 0.0,            -- 滑动窗口标准差
    window_count        INT DEFAULT 0,
    update_count        INT DEFAULT 0,                 -- 总更新次数
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
)
"""

# --- 世界对象表（物体身份追踪 + 场景图） ---

_WORLD_OBJECTS_DDL = """
CREATE TABLE IF NOT EXISTS embodied_world_objects (
    obj_id              VARCHAR(128) PRIMARY KEY,
    obj_type            VARCHAR(32),
    name                VARCHAR(128),
    pos_x               DOUBLE,
    pos_y               DOUBLE,
    pos_z               DOUBLE,
    orient_w            DOUBLE,
    orient_x            DOUBLE,
    orient_y            DOUBLE,
    orient_z            DOUBLE,
    size_json           TEXT,
    color_json          TEXT,
    mesh_path           VARCHAR(256),
    physics_props_json  TEXT,
    semantic_tags_json  TEXT,
    scene_id            VARCHAR(128),
    parent_obj_id       VARCHAR(128),
    state               VARCHAR(32) DEFAULT 'present',
    memory_id           BIGINT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES embodied_memories(memory_id) ON DELETE SET NULL
)
"""

_WORLD_OBJECT_INDEXES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_wo_scene ON embodied_world_objects(scene_id)",
    "CREATE INDEX IF NOT EXISTS idx_wo_parent ON embodied_world_objects(parent_obj_id)",
    "CREATE INDEX IF NOT EXISTS idx_wo_type ON embodied_world_objects(obj_type)",
    "CREATE INDEX IF NOT EXISTS idx_wo_state ON embodied_world_objects(state)",
    "CREATE INDEX IF NOT EXISTS idx_wo_memory ON embodied_world_objects(memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_wo_pos ON embodied_world_objects(pos_x, pos_y, pos_z)",
]

# --- 空间关系表（场景图边） ---

_SPATIAL_RELATIONS_DDL = """
CREATE TABLE IF NOT EXISTS embodied_spatial_relations (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    subject_id  VARCHAR(128) NOT NULL,
    object_id   VARCHAR(128) NOT NULL,
    relation    VARCHAR(32) NOT NULL,
    confidence  DOUBLE DEFAULT 1.0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (subject_id) REFERENCES embodied_world_objects(obj_id) ON DELETE CASCADE,
    FOREIGN KEY (object_id) REFERENCES embodied_world_objects(obj_id) ON DELETE CASCADE
)
"""

_SPATIAL_RELATION_INDEXES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_rel_subject ON embodied_spatial_relations(subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_rel_object ON embodied_spatial_relations(object_id)",
    "CREATE INDEX IF NOT EXISTS idx_rel_pair ON embodied_spatial_relations(subject_id, object_id)",
]

# --- 物理经验图（Memory of World 的关系边） ---
# 边类型限定为物理世界原生关系，拒绝通用语义边（如 related_to）

_EXPERIENCE_GRAPH_DDL = """
CREATE TABLE IF NOT EXISTS embodied_experience_graph (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    source_memory_id    BIGINT NOT NULL,
    target_memory_id    BIGINT NOT NULL,
    edge_type           VARCHAR(32) NOT NULL,
    strength            DOUBLE DEFAULT 1.0,
    spatial_context_json TEXT,                        -- 关系发生时的空间上下文
    temporal_context_json TEXT,                       -- 关系发生时的时间上下文
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_memory_id) REFERENCES embodied_memories(memory_id) ON DELETE CASCADE,
    FOREIGN KEY (target_memory_id) REFERENCES embodied_memories(memory_id) ON DELETE CASCADE,
    CHECK (edge_type IN ('causes', 'precedes', 'supports', 'contains', 'instantiates', 'part_of', 'adjacent_to', 'overlaps_temporally'))
)
"""

_EXPERIENCE_GRAPH_INDEXES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_expgraph_source ON embodied_experience_graph(source_memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_expgraph_target ON embodied_experience_graph(target_memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_expgraph_type ON embodied_experience_graph(edge_type)",
    "CREATE INDEX IF NOT EXISTS idx_expgraph_pair ON embodied_experience_graph(source_memory_id, target_memory_id)",
]

# --- 多维度抽象索引（经验抽象树的分形层级） ---
# 一个 memory_id 可在不同 dimension 中属于不同 concept

_CONCEPT_INDEX_DDL = """
CREATE TABLE IF NOT EXISTS embodied_concept_index (
    memory_id       BIGINT NOT NULL,
    dimension       VARCHAR(32) NOT NULL,           -- 'task' | 'physics' | 'spatial_region' | 'social'
    layer           INT NOT NULL,                   -- 层级深度（1=最具体）
    concept_id      VARCHAR(128) NOT NULL,
    confidence      DOUBLE DEFAULT 1.0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (memory_id, dimension, concept_id),
    FOREIGN KEY (memory_id) REFERENCES embodied_memories(memory_id) ON DELETE CASCADE
)
"""

_CONCEPT_INDEX_INDEXES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_concept_dim_layer ON embodied_concept_index(dimension, layer)",
    "CREATE INDEX IF NOT EXISTS idx_concept_concept ON embodied_concept_index(concept_id)",
    "CREATE INDEX IF NOT EXISTS idx_concept_memory ON embodied_concept_index(memory_id)",
]

# --- 物理模型库表（机器人/环境骨架） ---

_PHYSICAL_MODELS_DDL = """
CREATE TABLE IF NOT EXISTS embodied_physical_models (
    model_id            VARCHAR(128) PRIMARY KEY,      -- 如 "ur5_tabletop_v1"
    model_type          VARCHAR(32) DEFAULT 'robot',   -- robot | environment | object
    -- 纯数学描述，不依赖 ROS/e-URDF 运行时
    joint_names         TEXT,                          -- JSON: ["j1", "j2", ...]
    dh_params           TEXT,                          -- JSON: [{d, theta, a, alpha}, ...]
    mass_matrix         TEXT,                          -- JSON: M(q) 的参数化表示
    coriolis_matrix     TEXT,                          -- JSON: C(q, qdot) 的参数化表示
    gravity_vector      TEXT,                          -- JSON: G(q) 的参数化表示
    joint_limits        TEXT,                          -- JSON: [{min, max, vel_max, torque_max}, ...]
    link_masses         TEXT,                          -- JSON: [m1, m2, ...]
    link_inertias       TEXT,                          -- JSON: [[ixx, ixy, ...], ...]
    collision_geoms     TEXT,                          -- JSON: 碰撞体列表
    -- 来源
    source_urdf_hash    VARCHAR(64),                   -- 原始 URDF 的哈希（溯源用）
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
)
"""

_PHYSICAL_MODEL_INDEXES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_phys_model_type ON embodied_physical_models(model_type)",
]

# --- 聚合 ---

ALL_DDL = [
    _EMBODIED_MEMORIES_DDL,
    _CAUSAL_EDGES_DDL,
    _PREDICTIVE_STATE_DDL,
    _WORLD_OBJECTS_DDL,
    _SPATIAL_RELATIONS_DDL,
    _EXPERIENCE_GRAPH_DDL,
    _CONCEPT_INDEX_DDL,
    _PHYSICAL_MODELS_DDL,
]

ALL_INDEXES = (
    _SPATIAL_INDEXES
    + _TEMPORAL_INDEXES
    + _SEMANTIC_INDEXES
    + _CAUSAL_INDEXES
    + _WORLD_OBJECT_INDEXES
    + _SPATIAL_RELATION_INDEXES
    + _EXPERIENCE_GRAPH_INDEXES
    + _CONCEPT_INDEX_INDEXES
    + _PHYSICAL_MODEL_INDEXES
)


def initialize_embodied_schema(conn) -> None:
    """初始化具身扩展 Schema

    Args:
        conn: 数据库连接对象（pyseekdb Connection 或兼容 DB-API 的连接）

    幂等操作 — 重复调用安全。
    """
    # 自动检测方言
    dialect = "seekdb"
    conn_module = getattr(getattr(conn, "__class__", None), "__module__", "").lower()
    if "sqlite" in conn_module:
        dialect = "sqlite"
    elif "mysql" in conn_module or "pymysql" in conn_module:
        dialect = "mysql"
    elif "postgres" in conn_module or "psycopg" in conn_module:
        dialect = "postgres"

    ddl_list, idx_list = get_dialect_ddl(dialect)
    cursor = conn.cursor()
    for ddl in ddl_list:
        try:
            cursor.execute(ddl)
            logger.debug("Executed DDL: %s", ddl[:60])
        except Exception as e:
            logger.warning("DDL execution skipped: %s | error: %s", ddl[:60], e)

    for idx_ddl in idx_list:
        try:
            cursor.execute(idx_ddl)
            logger.debug("Executed INDEX: %s", idx_ddl[:60])
        except Exception as e:
            logger.warning("INDEX execution skipped: %s | error: %s", idx_ddl[:60], e)

    conn.commit()
    logger.info("ROSClaw-Memory embodied schema initialized (%s)", dialect)


def get_dialect_ddl(dialect: str = "seekdb") -> tuple[List[str], List[str]]:
    """获取特定方言的 DDL 列表

    Args:
        dialect: "seekdb" | "sqlite" | "mysql" | "postgres"

    Returns:
        (ddl_list, index_list)
    """
    if dialect == "seekdb":
        return ALL_DDL, ALL_INDEXES

    if dialect == "sqlite":
        # SQLite 适配：去掉 ON UPDATE CURRENT_TIMESTAMP，使用 TEXT 时间戳
        sqlite_ddl = [
            _EMBODIED_MEMORIES_DDL
            .replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
            .replace("ON UPDATE CURRENT_TIMESTAMP", "")
            .replace("BIGINT AUTO_INCREMENT PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            .replace("BIGINT PRIMARY KEY", "INTEGER PRIMARY KEY")
            .replace("DOUBLE", "REAL")
            .replace("VARCHAR(64)", "TEXT")
            .replace("VARCHAR(128)", "TEXT")
            .replace("VARCHAR(32)", "TEXT")
            .replace("INT", "INTEGER"),
            _CAUSAL_EDGES_DDL
            .replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
            .replace("BIGINT AUTO_INCREMENT PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            .replace("BIGINT", "INTEGER")
            .replace("DOUBLE", "REAL")
            .replace("VARCHAR(32)", "TEXT")
            .replace("VARCHAR(64)", "TEXT"),
            _PREDICTIVE_STATE_DDL
            .replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
            .replace("ON UPDATE CURRENT_TIMESTAMP", "")
            .replace("VARCHAR(128)", "TEXT")
            .replace("INT", "INTEGER")
            .replace("DOUBLE", "REAL"),
            _WORLD_OBJECTS_DDL
            .replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
            .replace("ON UPDATE CURRENT_TIMESTAMP", "")
            .replace("VARCHAR(128)", "TEXT")
            .replace("VARCHAR(32)", "TEXT")
            .replace("VARCHAR(256)", "TEXT")
            .replace("BIGINT", "INTEGER")
            .replace("DOUBLE", "REAL"),
            _SPATIAL_RELATIONS_DDL
            .replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
            .replace("BIGINT AUTO_INCREMENT PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            .replace("VARCHAR(128)", "TEXT")
            .replace("VARCHAR(32)", "TEXT")
            .replace("DOUBLE", "REAL"),
            _EXPERIENCE_GRAPH_DDL
            .replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
            .replace("BIGINT AUTO_INCREMENT PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            .replace("BIGINT", "INTEGER")
            .replace("VARCHAR(32)", "TEXT")
            .replace("DOUBLE", "REAL")
            .replace("    CHECK (edge_type IN ('causes', 'precedes', 'supports', 'contains', 'instantiates', 'part_of', 'adjacent_to', 'overlaps_temporally'))\n", "")
            .replace(",\n)", "\n)"),
            _CONCEPT_INDEX_DDL
            .replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
            .replace("BIGINT", "INTEGER")
            .replace("VARCHAR(32)", "TEXT")
            .replace("VARCHAR(128)", "TEXT")
            .replace("INT", "INTEGER")
            .replace("DOUBLE", "REAL"),
            _PHYSICAL_MODELS_DDL
            .replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
            .replace("ON UPDATE CURRENT_TIMESTAMP", "")
            .replace("VARCHAR(128)", "TEXT")
            .replace("VARCHAR(32)", "TEXT")
            .replace("VARCHAR(64)", "TEXT")
            .replace("INT", "INTEGER"),
        ]
        sqlite_indexes = [
            idx if "IF NOT EXISTS" in idx else idx.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ")
            for idx in ALL_INDEXES
            if "WHERE" not in idx  # SQLite 部分版本不支持部分索引
        ]
        return sqlite_ddl, sqlite_indexes

    # 默认返回 seekdb 方言
    return ALL_DDL, ALL_INDEXES
