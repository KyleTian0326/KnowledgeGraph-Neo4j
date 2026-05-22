# 石化 GraphRAG 全流程图

> 说明：DeepSeek 只负责从文本中生成“候选实体/关系 JSON”。过滤、证据校验、实体规范化、置信度计算、规则补抽都由本地 Python 模块 `kg_quality.py` 独立完成，不是同一个 LLM 再判断一遍。

```mermaid
flowchart TD
    A[PDF / TXT 原始资料<br/>source/] --> B{输入类型}

    B -->|PDF 有可抽取文本| C[PyMuPDF / pypdf 提取文本]
    B -->|扫描版 PDF 或 --force-ocr| D[RapidOCR OCR 识别<br/>按页输出文本]
    B -->|TXT| E[直接读取文本]

    C --> F[写入 data/pipeline_xxx/<br/>每页一个 txt]
    D --> F
    E --> F

    F --> G[split_text 切 chunk<br/>保留 document / page_start / page_end / source_ref]

    subgraph FAST[快速模式 ingest_source_pipeline.py]
        P[Producer<br/>OCR / 提取页文本] --> Q[Queue<br/>PreparedDocument]
        Q --> W1[Worker 1]
        Q --> W2[Worker 2]
        Q --> W3[Worker N]
    end

    G --> H1[KG 抽取分支]
    G --> H2[向量 Chunk 分支]

    subgraph LLM[LLM 候选抽取: build_kg_with_deepseek.py]
        H1 --> I[build_prompt<br/>限定实体类型 / 关系类型 / 必须复制 evidence]
        I --> J[DeepSeek API<br/>输出候选 nodes + relationships JSON]
        J --> K{JSON 是否合法}
        K -->|合法| L[normalize_extraction]
        K -->|不合法| M[DeepSeek JSON repair<br/>只修格式]
        M --> L
    end

    subgraph QUALITY[本地质量门控: kg_quality.py<br/>独立 Python 规则，不是 LLM]
        L --> N[实体清洗 clean_name<br/>去标点 / 空格 / 泛化尾词]
        N --> O[canonical_key<br/>实体规范化与去冗余]
        O --> P1{实体是否合法}
        P1 -->|否| DROP1[丢弃节点<br/>bad_name / unknown_label / name_not_found]
        P1 -->|是| R[保留候选节点]

        L --> S[关系类型映射<br/>RELATION_ALIASES]
        S --> T{关系类型是否合法}
        T -->|否| DROP2[丢弃关系<br/>unknown_type]
        T -->|是| U[端点校验<br/>source / target 必须出现在 chunk]

        U --> V{端点是否存在}
        V -->|否| DROP3[丢弃关系<br/>endpoint_not_found_in_chunk]
        V -->|是| X[evidence 校验<br/>证据必须能在 chunk 中定位]

        X --> Y{evidence 是否可信}
        Y -->|否| DROP4[丢弃关系<br/>missing_evidence / evidence_not_found]
        Y -->|是| Z[schema 约束<br/>RELATION_SCHEMA 检查主宾类型]

        Z --> AA[score_relationship<br/>重新计算 confidence]
        AA --> AB{confidence >= 0.70}
        AB -->|否| DROP5[丢弃关系<br/>low_confidence]
        AB -->|是| AC[保留高置信关系<br/>写入 quality_flags / support_count]

        R --> AD[规则补抽<br/>组分 / 质量指标]
        AC --> AD
        AD --> AE[rule_based 关系<br/>confidence=0.92]
        DROP1 --> QLOG[_quality.dropped]
        DROP2 --> QLOG
        DROP3 --> QLOG
        DROP4 --> QLOG
        DROP5 --> QLOG
    end

    subgraph NEO[Neo4j 入库]
        R --> NF[MERGE 节点<br/>按 canonical_key 合并]
        AC --> RF[MERGE 关系<br/>保存 evidence / confidence / page / source_ref]
        AE --> RF
        RF --> RG[同一关系多次出现<br/>support_count 累加<br/>保留最高置信 evidence]
    end

    subgraph VECTOR[向量索引: build_vector_chunks.py / pipeline]
        H2 --> VE[本地 Qwen3-Embedding-0.6B<br/>sentence-transformers]
        VE --> VC[写入 Chunk 节点<br/>text / page / source_ref / embedding]
        VC --> VI[Neo4j chunk_vector_index<br/>cosine / 1024 dim]
    end

    QLOG --> OUT[output/*_kg_extraction.json<br/>包含 accepted + dropped 质检信息]
    NF --> DB[(Neo4j<br/>实体 / 关系 / Chunk / Document)]
    RG --> DB
    VI --> DB

    OUT --> AUDIT[audit_kg_quality.py<br/>抽取后质量审计]
    AUDIT --> AR[统计结果<br/>Accepted / Dropped / dropped reasons<br/>low confidence / alias groups]

    subgraph CHAT[Web GraphRAG 问答: web_graphrag_chat.py]
        USER[用户问题] --> EMB[本地 Qwen 向量化问题]
        EMB --> VR[VectorRetriever<br/>召回相关 Chunk]
        USER --> GR[图谱关系召回<br/>只取 confidence >= 0.70]
        GR --> GS[关系精排<br/>匹配实体/证据/support_count/confidence]
        VR --> CTX[组合上下文<br/>Chunk + 图谱事实 + 页码引用]
        GS --> CTX
        CTX --> ANS[DeepSeek 生成回答<br/>带引用页码]
    end

    DB --> VR
    DB --> GR

    subgraph EVAL[召回评价: evaluate_graphrag_retrieval.py]
        DATA[eval/*.jsonl 标注集] --> EV1[向量 Chunk 评价<br/>Page Recall / MRR / nDCG / Citation Hit]
        DATA --> EV2[图谱关系评价<br/>Entity Recall / Triple Recall / Noise Edge Rate]
        DATA --> EV3[端到端答案评价<br/>run-answer / judge-answer]
        DB --> EV1
        DB --> EV2
        EV1 --> REPORT[output/retrieval_eval*.json/csv]
        EV2 --> REPORT
        EV3 --> REPORT
    end

    classDef llm fill:#ffe9cc,stroke:#d88900,color:#222;
    classDef local fill:#e8f4ff,stroke:#2878b8,color:#222;
    classDef db fill:#eaf7e9,stroke:#3a8f3a,color:#222;
    classDef drop fill:#ffe8e8,stroke:#c63b3b,color:#222;
    classDef eval fill:#f1e8ff,stroke:#7652c7,color:#222;

    class I,J,K,L,M,ANS llm;
    class N,O,P1,R,S,T,U,V,X,Y,Z,AA,AB,AC,AD,AE,QLOG,VE,VC,VI,AUDIT,AR local;
    class DB,NF,RF,RG db;
    class DROP1,DROP2,DROP3,DROP4,DROP5 drop;
    class EV1,EV2,EV3,REPORT eval;
```

## 核心回答

- **LLM 抽取和质量过滤不是同一个步骤。**
- DeepSeek 做的是候选抽取：根据 prompt 从 chunk 中输出 JSON。
- `kg_quality.py` 做的是本地确定性质量门控：证据定位、端点检查、schema 约束、置信度重算、低置信过滤。
- 只有通过 `kg_quality.py` 的节点和关系才会写入 Neo4j。
- `audit_kg_quality.py` 不参与入库，只读取 `output/*_kg_extraction.json` 做抽取后审计。

## 快速模式并行关系

快速模式 `ingest_source_pipeline.py` 不是先完整 OCR 再完整抽取，而是：

```text
Producer 一边按页 OCR/提取文本 -> Queue
多个 Worker 一边消费页面文本 -> DeepSeek 抽取 + kg_quality 过滤 + Neo4j 写入 + Qwen 向量写入
```

所以页码信息是在最早的页面文本阶段就写入，并一路传递到 chunk、关系证据、Chunk 向量节点和最终引用。
