# RosClaw-Memory MCP Server

让 Claude 通过 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 直接读取机器人的具身记忆。

## 功能

Claude 可以通过 MCP Tools 查询：

- **`search_near`** — 空间范围搜索记忆原子
- **`search_temporal`** — 按时间区间搜索记忆
- **`search_similar_trajectories`** — 轨迹相似度检索（DTW）
- **`get_scene_graph`** — 获取场景图（对象 + 空间关系）
- **`search_world_objects`** — 按位置搜索物理世界对象
- **`get_world_object`** — 获取特定对象的详细信息
- **`get_causal_chain`** — 查询因果链（causes / effects）
- **`get_stats`** — 记忆系统统计

## 安装

```bash
pip install mcp  # 或 pip install -e ".[mcp]" 如果 pyproject.toml 已更新
```

## 启动

```bash
# 方式 1: CLI
powermem-mcp-server --db-path ./embodied.db

# 方式 2: Python 模块
python -m powermem.mcp.cli --db-path ./embodied.db
```

## Claude Desktop 配置

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) 或对应平台配置：

```json
{
    "mcpServers": {
        "rosclaw-memory": {
            "command": "powermem-mcp-server",
            "args": ["--db-path", "/path/to/embodied.db"]
        }
    }
}
```

## Claude Code 配置

编辑 `~/.claude/settings.json`：

```json
{
    "mcpServers": {
        "rosclaw-memory": {
            "command": "powermem-mcp-server",
            "args": ["--db-path", "/path/to/embodied.db"]
        }
    }
}
```

## 使用示例

配置完成后，在 Claude 中直接提问：

> "机器人在客厅里发现了什么物体？"

Claude 会自动调用 MCP tools：
1. `search_world_objects(center=<客厅中心>, radius=3m, scene_id="living_room")`
2. 返回对象列表和位姿
3. Claude 用自然语言回答

> "找一条与从厨房到卧室的路径最相似的历史轨迹"

Claude 会自动：
1. 生成查询轨迹路点
2. 调用 `search_similar_trajectories`
3. 返回最相似的轨迹和 DTW 距离
