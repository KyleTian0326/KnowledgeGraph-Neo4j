# KG / GraphRAG 评价说明

这个文件夹用于保存 GraphRAG 评价集和人工标注 gold 文件。当前评价分三类：

1. KG 准确性：把抽取出的实体和三元组与人工 gold 文件对比。
2. 回答增益：比较 vector-only、graph-only、hybrid GraphRAG 对回答的帮助。
3. 跨页连续性：检查 PDF 相邻页边界是否被 chunk overlap 保留下来。

## 1. KG Gold Evaluation

用途：检查知识图谱本身是否准确。评价不依赖大模型回答，只看实体和三元组是否抽对。

演示样本：

- 固定页面：《催化裂化装置操作指南》第 44 页。
- 样本文本：`eval\kg_gold_demo\kg_gold_demo_page_0044.txt`，内容为第 44 页完整 OCR 文本，不是人工摘录版。
- 人工 gold：`eval\kg_gold.example.jsonl`，按完整 OCR 页的稳定抽取事实标注。
- 演示目的：验证完整 OCR 页中，钝化剂、金属陷阱、海泡石、轻烯烃助剂、ZSM-5、β 沸石、轻烯烃产率、辛烷值等实体和关系能否被准确抽取，并展示 evidence 定位和质量门控。

第一步，基于演示页重新抽取 KG，使用 `--dry-run` 只生成 JSON，不写入 Neo4j：

```powershell
.\.venv\Scripts\python.exe scripts\build_kg_with_deepseek.py --input eval\kg_gold_demo\kg_gold_demo_page_0044.txt --dry-run --output output\kg_gold_demo_page44_extraction.json
```

第二步，运行 KG Gold Evaluation：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_kg_gold.py --extraction output\kg_gold_demo_page44_extraction.json --gold eval\kg_gold.example.jsonl --output output\kg_gold_demo_page44_eval.json
```

如果要评价你自己的大批量抽取结果，把 `--extraction` 换成对应 pipeline 输出：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_kg_gold.py --extraction output\pipeline_20260522_122022_kg_extraction.json --gold eval\kg_gold.example.jsonl --output output\kg_gold_eval.json
```

参数说明：

| 标志位 | 含义 | 示例 |
|---|---|---|
| `scripts\build_kg_with_deepseek.py` | DeepSeek KG 抽取脚本入口。用于从演示页文本中重新抽取实体和关系。 | 固定写法 |
| `--input` | 要抽取的 TXT/PDF 文件或文件夹。演示中固定为第 44 页样本。 | `eval\kg_gold_demo\kg_gold_demo_page_0044.txt` |
| `--dry-run` | 只输出抽取 JSON，不写入 Neo4j。适合做 PPT demo 和离线评估。 | 开关，无参数值 |
| `scripts\evaluate_kg_gold.py` | KG gold 评价脚本入口。 | 固定写法 |
| `--extraction` | KG 抽取结果 JSON，演示中来自 `build_kg_with_deepseek.py --dry-run`。 | `output\kg_gold_demo_page44_extraction.json` |
| `--gold` | 人工标注 gold 文件，JSONL 格式，每行包含 `entities` 和 `triples`。 | `eval\kg_gold.example.jsonl` |
| `--output` | 抽取 JSON 或评价报告输出位置，取决于所运行的脚本。 | `output\kg_gold_demo_page44_eval.json` |

核心指标：

| 指标 | 含义 |
|---|---|
| `entity.precision` | 抽出的实体里有多少是 gold 中正确实体。 |
| `entity.recall` | gold 中应抽出的实体有多少被抽到。 |
| `entity.f1` | 实体 precision 和 recall 的综合分数。 |
| `triple.precision` | 抽出的三元组里有多少是 gold 中正确三元组。 |
| `triple.recall` | gold 中应抽出的三元组有多少被抽到。 |
| `triple.f1` | 三元组 precision 和 recall 的综合分数。 |
| `evidence.has_evidence_rate` | 图谱关系中有 evidence 字段的比例。 |
| `evidence.evidence_support_rate` | evidence 能在原 chunk 中定位到的比例。 |

gold 文件示例：

```jsonl
{"id":"kg_demo_page44_full_ocr","source":"催化裂化装置操作指南 第2版","page":44,"entities":["催化裂化装置","钝化剂","金属陷阱","海泡石","碱氮","轻烯烃助剂","ZSM-5","β沸石","轻烯烃产率","辛烷值","凹凸棒石","锂蒙脱石","捕集剂","硫转移添加剂","氢气与甲烷的比值","氢气量","转化率"],"triples":[["催化裂化装置","USES_MATERIAL","钝化剂"],["催化裂化装置","USES_MATERIAL","金属陷阱"],["海泡石","USED_FOR","碱氮"],["催化裂化装置","USES_MATERIAL","轻烯烃助剂"],["轻烯烃助剂","HAS_COMPONENT","ZSM-5"],["轻烯烃助剂","HAS_COMPONENT","β沸石"],["ZSM-5","AFFECTS","轻烯烃产率"],["ZSM-5","AFFECTS","辛烷值"]]}
```

## 2. Retrieval / Answer Evaluation

用途：检查 GraphRAG 是否真的帮助回答。它同时评价向量 chunk 召回、图谱事实召回、引用页码、答案支撑性和幻觉。

命令：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_graphrag_retrieval.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --mode all --judge-answer --output output\graphrag_mode_compare.json
```

这条命令会一次跑完三种模式：
- `vector`：只用向量 chunk。
- `graph`：只用 Neo4j 图谱事实。
- `hybrid`：同时使用向量 chunk 和图谱事实。

输出文件：
- `output\graphrag_mode_compare.json`：完整三模式报告。
- `output\graphrag_mode_compare.md`：适合 PPT 截图的 Markdown 对比表。
- `output\graphrag_mode_compare.csv`：适合 Excel 打开的对比表。
- `output\graphrag_mode_compare_parts\retrieval_eval_vector.json`：vector-only 明细。
- `output\graphrag_mode_compare_parts\retrieval_eval_graph.json`：graph-only 明细。
- `output\graphrag_mode_compare_parts\retrieval_eval_hybrid.json`：hybrid 明细。

参数说明：

| 标志位 | 含义 | 示例 |
|---|---|---|
| `scripts\evaluate_graphrag_retrieval.py` | 检索和回答评价脚本入口。 | 固定写法 |
| `--dataset` | 问题评价集，JSONL 格式。每行包含 `question`、`expected_pages`、`expected_entities`、`expected_triples` 等字段。 | `eval\retrieval_eval.example.jsonl` |
| `--ks` | 向量 chunk 和 citation 的 Top-K 列表。`3,5,10` 表示同时计算 @3、@5、@10。 | `3,5,10` |
| `--graph-ks` | 图谱关系召回的 Top-K 列表。通常比 `--ks` 稍大，因为图谱事实更短。 | `5,10` |
| `--mode` | 证据来源模式。`vector` 只用向量 chunk，`graph` 只用图谱事实，`hybrid` 同时使用二者，`all` 一次性运行三种模式并生成对比报告。默认是 `hybrid`。 | `--mode all` |
| `--min-rel-confidence` | 图谱关系最低置信度过滤阈值。低于该值的边不参与图谱召回。 | `0.70` |
| `--run-answer` | 调用 DeepSeek 生成答案，并用关键词规则近似评价答案覆盖和幻觉。 | 开关，无参数值 |
| `--judge-answer` | 调用 DeepSeek 作为裁判，判断答案是否被证据支持、引用是否准确、是否缺证据、是否幻觉。该开关会自动生成答案。 | 开关，无参数值 |
| `--output` | JSON 评价报告输出位置。`--mode all` 时是三模式汇总报告位置。 | `output\graphrag_mode_compare.json` |
| `--csv` | 可选，额外输出 CSV 汇总表。 | `output\retrieval_eval.csv` |

常用模式：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_graphrag_retrieval.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --mode vector --output output\retrieval_eval_vector.json
.\.venv\Scripts\python.exe scripts\evaluate_graphrag_retrieval.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --mode graph --output output\retrieval_eval_graph.json
.\.venv\Scripts\python.exe scripts\evaluate_graphrag_retrieval.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --mode hybrid --judge-answer --output output\retrieval_eval_hybrid_judge.json
```

主要指标：

| 指标 | 含义 |
|---|---|
| `page_recall@K` | 正确页码是否出现在 Top-K 向量 chunk 中。 |
| `chunk_recall@K` | 正确 chunk 是否出现在 Top-K 向量 chunk 中。 |
| `mrr_page@K` | 第一个正确页码的倒数排名，越高越好。 |
| `ndcg_page@K` | 考虑强弱相关性的页码排序质量。 |
| `citation_hit@K` | 最终引用里是否命中正确页码。 |
| `citation_accuracy@K` | 引用页码中有多少比例属于正确页码。 |
| `entity_recall@K` | Top-K 图谱关系覆盖了多少关键实体。 |
| `triple_recall@K` | Top-K 图谱关系覆盖了多少 gold 三元组。 |
| `relation_precision@K` | Top-K 图谱关系里有多少是 gold 三元组。 |
| `noise_edge_rate@K` | `1 - relation_precision@K`，表示噪声边比例。 |
| `relation_evidence_coverage@K` | Top-K 图谱关系里有 evidence 和页码的比例。 |
| `judge_answer_supported_rate` | 裁判判断答案主要结论是否被上下文支持。 |
| `judge_citation_accuracy` | 裁判判断引用页是否能支撑答案。 |
| `judge_missing_evidence_rate` | 裁判判断答案是否缺关键证据。 |
| `judge_hallucination_rate` | 裁判判断答案是否出现上下文不支持的事实。 |

## 3. Mode Comparison / Ablation

用途：一次性比较 `vector`、`graph`、`hybrid` 三种模式，用来证明知识图谱是否真的带来增益。

命令：

```powershell
.\.venv\Scripts\python.exe scripts\compare_graphrag_modes.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --judge-answer --output output\graphrag_mode_compare.json
```

也可以通过统一评价入口运行，效果相同：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_graphrag_retrieval.py --dataset eval\retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --mode all --judge-answer --output output\graphrag_mode_compare.json
```

参数说明：

| 标志位 | 含义 | 示例 |
|---|---|---|
| `scripts\compare_graphrag_modes.py` | 三模式对照实验脚本入口。内部会多次调用 `evaluate_graphrag_retrieval.py`。 | 固定写法 |
| `--dataset` | 问题评价集。 | `eval\retrieval_eval.example.jsonl` |
| `--ks` | 向量 chunk 和 citation 的 Top-K 列表。 | `3,5,10` |
| `--graph-ks` | 图谱关系召回的 Top-K 列表。 | `5,10` |
| `--modes` | 要比较的模式，默认是 `vector,graph,hybrid`。 | `vector,graph,hybrid` |
| `--min-rel-confidence` | 图谱边最低置信度阈值。 | `0.70` |
| `--run-answer` | 每种模式都生成答案，但不启用 LLM 裁判。 | 开关，无参数值 |
| `--judge-answer` | 每种模式都生成答案，并启用 LLM 裁判。 | 开关，无参数值 |
| `--python` | 指定 Python 解释器，默认使用当前工程 `.venv`。 | `.\.venv\Scripts\python.exe` |
| `--output` | 三模式汇总报告输出位置。 | `output\graphrag_mode_compare.json` |
| `--markdown` | 可选，额外指定 Markdown 对比表输出位置。不写时自动生成同名 `.md` 文件。 | `output\graphrag_mode_compare.md` |
| `--csv` | 可选，额外指定 CSV 对比表输出位置。不写时自动生成同名 `.csv` 文件。 | `output\graphrag_mode_compare.csv` |
| `--display-metrics` | 可选，指定展示表中要显示的指标，逗号分隔。不写时展示默认核心指标。 | `page_recall@5,triple_recall@5,judge_answer_supported_rate` |

判断方法：

- 如果 `hybrid` 的 `page_recall@5` 和 `judge_answer_supported_rate` 不低于 `vector`；
- 同时 `hybrid` 的 `entity_recall@K`、`triple_recall@K` 高于 `vector`；
- 且 `judge_hallucination_rate` 不升高；

就可以说明图谱事实对回答有正向帮助。

## 4. Page Continuity Audit

用途：检查 PDF 分页导致的句子断裂是否被 chunk overlap 保留。尤其适合扫描版 PDF，因为一页一个 txt 时，跨页句子容易被切断。

当前默认修复策略已经接入 ingest 流程：PDF 会先按连续页窗口合并，再进入 KG 抽取和向量 Chunk 构建。默认参数是 `--page-window 20 --page-window-overlap 1`，表示每 20 页合并成一个带 `===== PAGE n =====` 标记的文本，相邻窗口重叠 1 页。这样既保留跨页句子，又不会等整本 PDF OCR 完才开始 DeepSeek 抽取。

如果审计发现 `risk_rate` 仍然偏高，可以加大窗口重叠：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source_pipeline.py --file "催化裂化装置操作指南.pdf" --pages 30-80 --workers 3 --force-ocr --page-window 20 --page-window-overlap 2
```

如果需要临时恢复旧的一页一个 txt 行为：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_source_pipeline.py --file "催化裂化装置操作指南.pdf" --pages 30-80 --workers 3 --force-ocr --page-window 0
```

命令：

```powershell
.\.venv\Scripts\python.exe scripts\audit_page_continuity.py --input data\pipeline_20260522_110658 --output output\page_continuity_audit.json
```

参数说明：

| 标志位 | 含义 | 示例 |
|---|---|---|
| `scripts\audit_page_continuity.py` | 跨页连续性审计脚本入口。 | 固定写法 |
| `--input` | 准备好的文本文件或文件夹，通常是 `data\pipeline_时间戳` 或 `data\ingested_时间戳`。 | `data\pipeline_20260522_110658` |
| `--output` | 审计报告输出位置。 | `output\page_continuity_audit.json` |
| `--tail-chars` | 从上一页末尾取多少字符作为边界尾部证据。默认 `80`。 | `80` |
| `--head-chars` | 从下一页开头取多少字符作为边界开头证据。默认 `80`。 | `80` |
| `--min-match-chars` | 至少多少连续字符要出现在同一个 chunk 中，才算跨页桥接成功。默认 `24`。 | `24` |

指标解释：

| 指标 | 含义 |
|---|---|
| `bridge_rate` | 相邻页边界中，页尾和下一页页头被同一个跨页 chunk 覆盖的比例。越高越好。 |
| `risk_rate` | 可能没有被 chunk overlap 覆盖的页边界比例。越低越好。 |
| `likely_sentence_split` | 上一页没有以句号/问号/感叹号等结束，说明下一页可能延续同一句。 |

## 5. 建议的汇报顺序

1. 先跑 `evaluate_kg_gold.py`：说明图谱抽取质量。
2. 再跑 `audit_page_continuity.py`：说明页码和跨页上下文没有丢。
3. 最后跑 `compare_graphrag_modes.py`：说明图谱是否真的帮助最终回答。
