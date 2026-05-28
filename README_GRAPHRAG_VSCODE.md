# 石化 GraphRAG 本地运行说明

这套工程用于演示：

1. 从石化资料中抽取文本；
2. 用 DeepSeek 抽取实体和关系；
3. 写入本地 Neo4j 知识图谱；
4. 给文本切 Chunk、做 embedding、创建 Neo4j 向量索引；
5. 启动 Web 聊天页面，让用户像 ChatGPT 一样提问；
6. 后台自动使用 Neo4j 图谱事实 + 官方 `VectorRetriever` 检索片段，再交给 DeepSeek 回答。

## 当前版本快速指南

当前工程已经切换到：

- LLM：DeepSeek；
- 知识图谱抽取：DeepSeek + schema 约束 + 证据校验 + 置信度过滤；
- 本地 embedding：`Qwen3-Embedding-0.6B`；
- 向量维度：`1024`；
- Neo4j 向量索引：`chunk_vector_index`。

日常使用时，不需要重新配置。每次只需要：

```powershell
cd "E:\Desktop\石化大模型\Neo4j\工程文件"
.\.venv\Scripts\python.exe scripts\web_graphrag_chat.py
```

如果你新增了资料，把 `.pdf`、`.txt`、`.md` 放进 `source` 文件夹，然后运行：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source.py
```

如果你要清空旧图谱和旧向量，从头重建，先启动 Neo4j，再按下面顺序运行：

```powershell
.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; from pathlib import Path; import os; from neo4j import GraphDatabase; load_dotenv(Path('.env')); driver=GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))); db=os.getenv('NEO4J_DATABASE','neo4j'); driver.execute_query('DROP INDEX chunk_vector_index IF EXISTS', database_=db); driver.execute_query('MATCH ()-[r]->() WHERE r.deepseek_kg = true DELETE r', database_=db); driver.execute_query('MATCH (n) WHERE n.deepseek_kg = true DETACH DELETE n', database_=db); driver.execute_query('MATCH (c:Chunk) WHERE c.vector_kg = true DETACH DELETE c', database_=db); driver.execute_query('MATCH (d:Document) WHERE NOT (d)--() DELETE d', database_=db); driver.close(); print('Old KG and vector chunks cleared.')"
.\.venv\Scripts\python.exe scripts\build_kg_with_deepseek.py --input data\ingested_20260514_170608 --output output\rebuild_qwen3_kg_extraction.json
.\.venv\Scripts\python.exe scripts\audit_kg_quality.py output\rebuild_qwen3_kg_extraction.json
.\.venv\Scripts\python.exe scripts\build_vector_chunks.py --input data\ingested_20260514_170608 --reset
```

## 0. 项目目录

在 PowerShell 里先进入工程目录：

```powershell
cd "E:\Desktop\石化大模型\Neo4j\工程文件"
```

后面的所有命令都默认在这个目录下执行。

## 1. 启动本地 Neo4j

先打开 Neo4j Desktop / Neo4j App，启动你的本地数据库。

当前 `.env` 默认连接：

```env
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j
```

如果你在 Neo4j 里改过密码，要同步修改 `.env`。

启动后先测试连接：

```powershell
.\.venv\Scripts\python.exe scripts\check_neo4j.py
```

如果成功，说明 Python 已经能连上 Neo4j。

## 2. 检查 Python 环境

本工程使用本地虚拟环境：

```text
.venv\Scripts\python.exe
```

如果依赖缺失，运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. 检查 DeepSeek 配置

`.env` 里需要有：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
LLM_MODEL=deepseek-v4-flash
```

注意：不要把真实 API Key 发到公开文档、PPT 或 GitHub。

可以用下面命令测试模型是否能调用：

```powershell
.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; from openai import OpenAI; from pathlib import Path; import os; load_dotenv(Path('.env')); client=OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com')); model=os.getenv('DEEPSEEK_MODEL','deepseek-v4-flash'); extra={'extra_body': {'thinking': {'type': 'disabled'}}} if model.startswith('deepseek-v4') else {}; response=client.chat.completions.create(model=model, messages=[{'role':'user','content':'Reply with exactly: ok'}], max_tokens=10, **extra); print(response.choices[0].message.content)"
```

正常输出：

```text
ok
```

## 4. 推荐工作流：source 文件夹一条命令增量入库

以后新收集的资料统一放到：

```text
source
```

支持的文件类型：

```text
.pdf
.txt
.md
```

如果没有 `source` 文件夹，先创建：

```powershell
mkdir source
```

然后把新资料复制进去，例如：

```text
source\催化裂化装置操作指南.pdf
source\工艺说明.txt
```

### 4.1 默认处理 source 里的全部文件

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source.py
```

默认行为：

- 扫描 `source` 文件夹里的 `.pdf`、`.txt`、`.md`；
- PDF 如果有文本层，就直接提取文本；
- PDF 如果是扫描版，就自动 OCR；
- 提取后的文本保存到 `data\ingested_时间戳\`；
- PDF 会按页保存为 `资料名_page_0030.txt` 这类文件，文件内容开头也会写入 `===== PAGE 30 =====`；
- 调用 DeepSeek 抽取实体和关系；
- 把结果增量写入 Neo4j；
- 调用 embedding，把原文切成 `Chunk` 并生成向量；
- 把向量写入 `Chunk.embedding`；
- 创建或更新 Neo4j 向量索引 `chunk_vector_index`；
- 抽取 JSON 保存到 `output\ingested_时间戳_kg_extraction.json`；
- 同时更新一份最新结果到 `output\deepseek_kg_extraction.json`；
- `Chunk` 和图谱关系会保存 `document`、`page_start`、`page_end`、`page`、`source_ref`，方便问答时显示资料出处。

这条命令不会清空原来的知识图谱。Neo4j 里用 `MERGE` 合并节点和关系，所以你的知识库会在原来的基础上逐渐积累。

### 4.1.1 页码出处追溯

当前版本会把 PDF 页码贯穿到整条 GraphRAG 链路：

- PDF 提取或 OCR 后，每一页都会写成单独文本，例如 `催化裂化装置操作指南_page_0030.txt`；
- 切 Chunk 时，会把页码写入 `Chunk.document`、`Chunk.page_start`、`Chunk.page_end`、`Chunk.page`、`Chunk.source_ref`；
- DeepSeek 抽取出的图谱关系边会保存同样的页码字段，并额外累积 `source_refs`，用于追溯多个支持证据；
- Web 问答返回引用时，会优先显示 `资料名，第 N 页`，这样可以回到原 PDF 对照确认。

注意：旧数据库里已经存在的 Chunk 和图谱关系不会自动补齐新字段。如果你要让旧资料也带准确页码，建议先清空旧 KG/向量，再从 `source` 里的 PDF 重新运行 `scripts\ingest_source.py`。如果直接复用旧 `data` 文件夹，只有文本里本身带 `===== PAGE N =====` 标记的资料才能恢复页码。

这里要区分两种数据：

```text
实体关系图谱：
原文 → DeepSeek → 实体/关系 → Neo4j 节点和关系

向量检索索引：
原文 → 切 Chunk → embedding → Chunk.embedding → Neo4j 向量索引
```

也就是说，embedding 不参与“实体关系抽取”，它参与的是“根据用户问题召回相关原文片段”。

当前版本默认使用本地 `Qwen3-Embedding-0.6B` 语义向量模型：

```text
scripts\local_embeddings.py
```

模型路径在 `.env` 中配置：

```env
LOCAL_EMBEDDING_PROVIDER=qwen
LOCAL_EMBEDDING_MODEL_PATH=C:\Users\Kyle\.cache\modelscope\hub\models\Qwen\Qwen3-Embedding-0___6B
LOCAL_EMBEDDING_DEVICE=cpu
LOCAL_EMBEDDING_DIMENSIONS=1024
LOCAL_EMBEDDING_NORMALIZE=true
```

`HashingEmbedder` 仍然保留为兜底 demo 模式，但日常使用不推荐。Qwen3 生成的是 1024 维语义向量，质量明显高于旧的 384 维哈希向量。

### 4.2 只处理 source 里的某一个文件

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source.py --file "催化裂化装置操作指南.pdf"
```

注意：`--file` 后面写的是 `source` 文件夹里的文件名，不需要写完整路径。

### 4.3 只处理 PDF 的指定页数

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source.py --file "催化裂化装置操作指南.pdf" --pages 30-35
```

也可以写多个页码或区间：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source.py --file "催化裂化装置操作指南.pdf" --pages 1,5,20-30
```

如果不写 `--pages`，默认处理 PDF 全部页。

### 4.4 强制 OCR

有些 PDF 虽然看起来有文字层，但提取质量很差，可以强制 OCR：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source.py --file "催化裂化装置操作指南.pdf" --pages 30-35 --force-ocr
```

### 4.5 只测试抽取，不写入 Neo4j

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source.py --file "催化裂化装置操作指南.pdf" --pages 30-35 --dry-run
```

dry-run 会生成抽取 JSON，但不会更新 Neo4j，也不会写入向量 Chunk。

### 4.6 重新生成全部 Chunk 向量索引

一般不要加这个参数，因为你的目标是增量积累。

只有当你想清空旧 Chunk 并重新构建向量检索时，才运行：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source.py --reset-vector
```

### 4.7 快速模式：OCR 和 DeepSeek 同时跑

普通 `ingest_source.py` 是稳妥顺序执行：

```text
先 OCR/提取完全部文本
→ 再 DeepSeek 抽取
→ 再写 Chunk embedding
```

如果 PDF 页数很多，可以用流水线脚本：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source_pipeline.py --file "催化裂化装置操作指南.pdf" --pages 30-80 --workers 3 --force-ocr
```

流水线模式会：

```text
OCR 完第 1 页 → 立刻交给 DeepSeek worker 抽取
OCR 第 2 页继续跑 → DeepSeek 同时处理第 1 页
OCR 后续页持续入队 → 多个 DeepSeek worker 并发抽取
每个 chunk 完成后 → 立即写入 Neo4j 和 Chunk.embedding
```

运行时会显示进度条：

```text
Pipeline 35/100 [########----------------] 12:30/35:42
```

含义：

- `35/100`：已完成页数或文档数 / 总页数或总文档数；
- `[########----------------]`：进度条；
- `12:30/35:42`：已花费时间 / 预计总耗时。

参数说明：

- `--workers 3`：DeepSeek 并发抽取数量，建议先用 2 或 3。
- `--pages 30-80`：只处理需要的页，速度提升最明显。
- `--force-ocr`：扫描版 PDF 建议加上。
- `--reset-vector`：只有想清空旧 Chunk 重新建向量时才加。

注意：并发越高不一定越快，可能遇到 API 限速，也会增加同时请求数量。建议从：

```powershell
--workers 2
```

或：

```powershell
--workers 3
```

开始试。

## 5. 增量入库后的检查

检查知识图谱关系：

```cypher
MATCH p=(a)-[r]->(b) WHERE r.deepseek_kg = true RETURN p LIMIT 100
```

检查文本 Chunk：

```cypher
MATCH (c:Chunk) RETURN c.source, c.index, left(c.text, 300) AS preview LIMIT 10
```

检查 Chunk 是否已经写入 embedding：

```cypher
MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN c.source, c.index, size(c.embedding) AS embedding_dimensions LIMIT 10
```

当前默认维度应该是：

```text
1024
```

检查向量索引：

```cypher
SHOW VECTOR INDEXES
```

正常应该能看到：

```text
chunk_vector_index
```

状态是：

```text
ONLINE
```

## 6. 手动流程：如果 PDF 是扫描版，先 OCR 成文本

催化裂化装置电子书是扫描版 PDF，`pypdf` 直接提取不到文字，所以要先 OCR。

例如只 OCR 第 30 到 35 页：

```powershell
.\.venv\Scripts\python.exe scripts\ocr_pdf_to_text.py --input "data\催化裂化装置操作指南 第2版 (张韩，刘英聚编著) (z-library.sk, 1lib.sk, z-lib.sk) (1).pdf" --output data\fcc_ocr_pages_30_35.txt --pages 30-35
```

如果要 OCR 更多页，可以改 `--pages`：

```powershell
--pages 30-80
```

OCR 后的文本会保存到：

```text
data\fcc_ocr_pages_30_35.txt
```

同时会额外生成按页文本，例如：

```text
data\fcc_ocr_pages_30_35_page_0030.txt
data\fcc_ocr_pages_30_35_page_0031.txt
```

这些按页文本的文件名和正文页码标记会被后续 Chunk 与知识图谱关系继承，用于 Web 问答里的出处引用。

## 7. 手动流程：用 DeepSeek 抽取知识图谱并写入 Neo4j

对 OCR 后的文本抽取实体和关系：

```powershell
.\.venv\Scripts\python.exe scripts\build_kg_with_deepseek.py --input data\fcc_ocr_pages_30_35.txt
```

这个脚本会：

- 读取文本；
- 分块；
- 调用 DeepSeek；
- 抽取实体和关系；
- 写入 Neo4j；
- 在 `output\deepseek_kg_extraction.json` 保存抽取结果。

如果你想先看抽取质量，不写入 Neo4j，可以 dry-run：

```powershell
.\.venv\Scripts\python.exe scripts\build_kg_with_deepseek.py --input data\fcc_ocr_pages_30_35.txt --dry-run --output output\deepseek_kg_extraction.json
```

## 8. 手动流程：创建 Chunk、Embedding 和向量索引

官方 `VectorRetriever` 需要 Neo4j 里有：

- `Chunk` 节点；
- `text` 文本属性；
- `embedding` 向量属性；
- Neo4j 向量索引。

运行：

```powershell
.\.venv\Scripts\python.exe scripts\build_vector_chunks.py --input data\fcc_ocr_pages_30_35.txt --reset
```

这个脚本会创建：

```text
(:Document)
(:Chunk {text, source, index, embedding})
```

并创建向量索引：

```text
chunk_vector_index
```

当前默认使用本地 `Qwen3-Embedding-0.6B` 生成 1024 维语义向量。旧的 `HashingEmbedder` 仅作为兜底 demo 模式保留，不建议日常使用。

## 9. 手动流程：在 Neo4j 里检查结果

打开 Neo4j 的 Query 页面，执行：

```cypher
MATCH p=(a)-[r]->(b) WHERE r.deepseek_kg = true RETURN p LIMIT 100
```

查看 Chunk：

```cypher
MATCH (c:Chunk) RETURN c.source, c.index, left(c.text, 300) AS preview LIMIT 10
```

查看向量索引：

```cypher
SHOW VECTOR INDEXES
```

正常应该能看到：

```text
chunk_vector_index
```

状态是：

```text
ONLINE
```

## 10. 召回评价系统

更完整的 KG / GraphRAG 评价说明见：

```text
eval\README_KG_ANSWER_EVAL.md
```

三类正式检查：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_kg_gold.py --extraction output\deepseek_kg_extraction.json --gold eval\kg_gold.example.jsonl --output output\kg_gold_eval.json
.\.venv\Scripts\python.exe scripts\compare_graphrag_modes.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --judge-answer --output output\graphrag_mode_compare.json
.\.venv\Scripts\python.exe scripts\audit_page_continuity.py --input data\pipeline_20260522_110658 --output output\page_continuity_audit.json
```

评测脚本：

```text
scripts\evaluate_graphrag_retrieval.py
```

样例评测集：

```text
eval\retrieval_eval.example.jsonl
```

快速跑三层召回评价：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_graphrag_retrieval.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --output output\retrieval_eval_example.json --csv output\retrieval_eval_example.csv
```

主要指标：

```text
向量 Chunk 召回：page_recall@K、chunk_recall@K、mrr_page@K、mrr_chunk@K、ndcg_page@K、citation_hit@K
图谱关系召回：entity_recall@K、triple_recall@K、relation_precision@K、noise_edge_rate@K、relation_evidence_coverage@K
端到端回答质量：answer_supported_rate、citation_accuracy@K、missing_evidence_rate、hallucination_rate
```

如果要调用 DeepSeek 生成答案并评估答案关键词覆盖：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_graphrag_retrieval.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --run-answer --output output\retrieval_eval_answer.json
```

如果要让 DeepSeek 作为裁判判断答案支撑性、引用准确性、缺证据和幻觉：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_graphrag_retrieval.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --judge-answer --output output\retrieval_eval_judge.json
```

建议先把 `page_recall@5` 作为主指标，同时看 `citation_accuracy@5` 和 `triple_recall@10`。标注格式见：

```text
eval\README_EVAL.md
```

## 11. 启动 Web 聊天页面

启动后端：

```powershell
.\.venv\Scripts\python.exe scripts\web_graphrag_chat.py
```

或者双击 / 运行：

```powershell
.\start_web_chat.bat
```

然后浏览器打开：

```text
http://127.0.0.1:7860
```

聊天页面的后台流程是：

```text
用户提问
→ VectorRetriever 检索 Chunk 文本片段
→ Neo4j 查询相关实体关系事实
→ 拼成上下文
→ DeepSeek 生成回答
→ 页面展示答案和证据
```

## 12. 手动流程的一键顺序命令示例

如果 Neo4j 已经启动，并且 `.env` 配置正确，可以按这个顺序跑：

```powershell
cd "E:\Desktop\石化大模型\Neo4j\工程文件"
.\.venv\Scripts\python.exe scripts\check_neo4j.py
.\.venv\Scripts\python.exe scripts\ocr_pdf_to_text.py --input "data\催化裂化装置操作指南 第2版 (张韩，刘英聚编著) (z-library.sk, 1lib.sk, z-lib.sk) (1).pdf" --output data\fcc_ocr_pages_30_35.txt --pages 30-35
.\.venv\Scripts\python.exe scripts\build_kg_with_deepseek.py --input data\fcc_ocr_pages_30_35.txt
.\.venv\Scripts\python.exe scripts\build_vector_chunks.py --input data\fcc_ocr_pages_30_35.txt --reset
.\.venv\Scripts\python.exe scripts\web_graphrag_chat.py
```

最后打开：

```text
http://127.0.0.1:7860
```

## 13. 常见问题

### 1. Neo4j 连不上

先确认 Neo4j App 里的数据库是 Running。

再检查 `.env`：

```env
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
```

然后重新运行：

```powershell
.\.venv\Scripts\python.exe scripts\check_neo4j.py
```

### 2. DeepSeek 返回空内容

`deepseek-v4-flash` 默认可能输出 `reasoning_content`。本工程已经在脚本里关闭思考模式：

```python
extra_body={"thinking": {"type": "disabled"}}
```

如果你自己新写脚本，也要加这个参数。

### 3. PDF 抽不到文字

扫描版 PDF 不能直接用 `pypdf` 抽文字，要先运行：

```powershell
.\.venv\Scripts\python.exe scripts\ocr_pdf_to_text.py --input 你的PDF --output data\xxx.txt --pages 30-50
```

### 4. Web 页面没有引用资料

先确认已经运行过：

```powershell
.\.venv\Scripts\python.exe scripts\build_vector_chunks.py --input data\fcc_ocr_pages_30_35.txt --reset
```

再到 Neo4j 里执行：

```cypher
MATCH (c:Chunk) RETURN count(c)
```

如果结果是 0，说明 Chunk 还没写进去。
