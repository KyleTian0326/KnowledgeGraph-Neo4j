# 石化 GraphRAG

当前项目已适配 Linux/bash 容器环境。项目根目录为：

```bash
cd /root/石化大模型/Neo4j/工程文件
```

常用命令：

```bash
./start_neo4j_foreground.sh
./.venv/bin/python scripts/check_neo4j.py
./.venv/bin/python scripts/ingest_source.py
./.venv/bin/python scripts/web_graphrag_chat.py
```

在当前容器里，Neo4j 需要保持一个终端会话运行 `./start_neo4j_foreground.sh`。看到 `Started.` 后，再开另一个终端运行入库或 Web 命令。

也可以直接启动 Web 聊天页面：

```bash
./start_web_chat.sh
```

浏览器访问：

```text
http://127.0.0.1:7860
```

完整说明见 `README_GRAPHRAG_VSCODE.md`。
