# GraphRAG 召回评价说明

这个文件夹用于保存 GraphRAG 评测集。评测集采用 `jsonl`，一行一个问题。

## 快速运行

推荐先跑三模式对照，一次性比较 `vector`、`graph`、`hybrid`：

```bash
./.venv/bin/python scripts/evaluate_graphrag_retrieval.py --dataset eval/retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --mode all --judge-answer --output output/graphrag_mode_compare.json
```

运行后会生成：
- `output/graphrag_mode_compare.json`：三模式完整 JSON 报告。
- `output/graphrag_mode_compare.md`： 对比表。
- `output/graphrag_mode_compare.csv`：适合 Excel 打开的指标表。
- `output/graphrag_mode_compare_parts/retrieval_eval_vector.json`、`retrieval_eval_graph.json`、`retrieval_eval_hybrid.json`：三种模式各自的明细报告。

如果只想跑单一 hybrid 模式：

```bash
./.venv/bin/python scripts/evaluate_graphrag_retrieval.py --dataset eval/retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --output output/retrieval_eval_example.json --csv output/retrieval_eval_example.csv
```

如果要同时调用 DeepSeek 生成回答并评估答案关键词覆盖：

```bash
./.venv/bin/python scripts/evaluate_graphrag_retrieval.py --dataset eval/retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --run-answer --output output/retrieval_eval_answer.json
```

如果要让 DeepSeek 作为裁判判断答案支撑性、引用准确性、缺证据和幻觉：

```bash
./.venv/bin/python scripts/evaluate_graphrag_retrieval.py --dataset eval/retrieval_eval.example.jsonl --ks 3,5,10 --graph-ks 5,10 --judge-answer --output output/retrieval_eval_judge.json
```

## 标注格式

```json
{"id":"q001","question":"催化剂选择性表示什么？","expected_pages":[41],"relevant_pages":{"41":3},"expected_entities":["催化剂","选择性"],"expected_triples":[["催化剂","HAS_PARAMETER","选择性"]],"expected_answer_keywords":["选择性"],"forbidden_answer_keywords":[]}
```

字段含义：

- `question`：评测问题。
- `expected_pages`：正确来源页码，用于 `Page Recall@K`、`MRR Page@K`、`Citation Hit@K`。
- `relevant_pages`：带强弱相关性的页码，分数越高越相关，用于 `nDCG Page@K`。
- `expected_chunks` 或 `expected_source_refs`：正确 chunk 标识，用于 `Chunk Recall@K`、`MRR Chunk@K`、`nDCG Chunk@K`。
- `expected_entities`：问题应召回的关键实体，用于 `Entity Recall@K`。
- `expected_triples`：问题应召回的三元组，用于 `Triple Recall@K`、`Relation Precision@K`、`Noise Edge Rate@K`。
- `expected_answer_keywords`：答案里应覆盖的关键词，用于不调用裁判模型时的 `Answer Supported Rate` 近似评估。
- `forbidden_answer_keywords`：答案里不应出现的关键词，用于 `Hallucination Rate` 的规则近似评估。

## 指标解释

- `page_recall@K`：Top-K 向量 chunk 覆盖了多少正确页码。
- `chunk_recall@K`：Top-K 向量 chunk 覆盖了多少正确 chunk。
- `mrr_page@K` / `mrr_chunk@K`：第一个正确页或 chunk 的倒数排名。
- `ndcg_page@K` / `ndcg_chunk@K`：考虑相关性强弱的排序质量。
- `citation_hit@K`：最终引用里是否命中正确页码。
- `citation_accuracy@K`：引用页里有多少比例属于标注正确页。
- `entity_recall@K`：Top-K 图谱关系覆盖了多少关键实体。
- `triple_recall@K`：Top-K 图谱关系覆盖了多少标注三元组。
- `relation_precision@K`：Top-K 图谱关系里有多少属于标注正确三元组。
- `noise_edge_rate@K`：`1 - relation_precision@K`。
- `relation_evidence_coverage@K`：Top-K 图谱关系里有原文证据和页码的比例。
- `answer_supported_rate`：规则模式下，答案覆盖了多少应答关键词。
- `missing_evidence_rate`：规则模式下，`1 - answer_supported_rate`。
- `hallucination_rate`：规则模式下，答案命中了多少 forbidden 关键词。

实际建设时，建议把 `Page Recall@5` 作为主指标，同时看 `Citation Accuracy@5` 和 `Triple Recall@10`。
