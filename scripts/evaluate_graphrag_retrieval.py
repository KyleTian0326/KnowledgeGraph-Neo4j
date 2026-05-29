import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever

from local_embeddings import build_embedder


load_dotenv(Path(".env"))

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunk_vector_index")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_EXTRA_BODY = (
    {"thinking": {"type": "disabled"}}
    if DEEPSEEK_MODEL.startswith("deepseek-v4")
    else None
)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip() and int(part.strip()) > 0]


def normalize_key(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def normalize_page(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(text))
    except ValueError:
        return text


def expand_pages(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    pages: set[str] = set()
    for part in re.split(r"[,，]", text):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                pages.update(str(page) for page in range(int(start), int(end) + 1))
            except ValueError:
                pages.add(normalize_page(part))
        else:
            pages.add(normalize_page(part))
    return {page for page in pages if page}


def normalize_pages(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (str, int)):
        return expand_pages(values)
    pages: set[str] = set()
    for value in values:
        pages.update(expand_pages(value))
    return pages


def normalize_chunk_refs(values: Any) -> set[str]:
    if not values:
        return set()
    if isinstance(values, str):
        return {values.strip()}
    refs = set()
    for value in values:
        if isinstance(value, dict):
            ref = value.get("source_ref") or value.get("id") or value.get("chunk_id")
        else:
            ref = value
        if ref:
            refs.add(str(ref).strip())
    return refs


def normalize_triple(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, dict):
        source = value.get("source")
        rel_type = value.get("type") or value.get("relation")
        target = value.get("target")
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        source, rel_type, target = value[:3]
    elif isinstance(value, str):
        match = re.match(r"\s*(.*?)\s*[-=]+>\s*(.*?)\s*[-=]+>\s*(.*?)\s*$", value)
        if not match:
            return None
        source, rel_type, target = match.groups()
    else:
        return None

    if not source or not rel_type or not target:
        return None
    return (normalize_key(source), str(rel_type).strip().upper(), normalize_key(target))


def normalize_triples(values: Any) -> set[tuple[str, str, str]]:
    triples = set()
    for value in values or []:
        triple = normalize_triple(value)
        if triple:
            triples.add(triple)
    return triples


def parse_relevance(raw: Any, expected_values: set[str]) -> dict[str, float]:
    relevance: dict[str, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            relevance[str(key)] = float(value)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                key = item.get("id") or item.get("page") or item.get("source_ref")
                score = item.get("relevance", item.get("score", 1))
                if key is not None:
                    relevance[str(key)] = float(score)
            else:
                relevance[str(item)] = 1.0
    for value in expected_values:
        relevance.setdefault(str(value), 1.0)
    return relevance


def dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def ndcg_at_k(retrieved_gains: list[float], ideal_gains: list[float], k: int) -> float | None:
    if not ideal_gains:
        return None
    actual = dcg(retrieved_gains[:k])
    ideal = dcg(sorted(ideal_gains, reverse=True)[:k])
    if ideal == 0:
        return None
    return actual / ideal


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def metric_value(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        row.setdefault("id", f"q{line_no:04d}")
        rows.append(row)
    return rows


def load_graph(driver, min_confidence: float) -> dict[str, list[dict[str, Any]]]:
    node_records, _, _ = driver.execute_query(
        """
        MATCH (n)
        WHERE n.name IS NOT NULL
        RETURN elementId(n) AS id,
               labels(n) AS labels,
               n.name AS name,
               properties(n) AS properties
        """,
        database_=NEO4J_DATABASE,
    )
    rel_records, _, _ = driver.execute_query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.name IS NOT NULL
          AND b.name IS NOT NULL
          AND coalesce(r.confidence, 1.0) >= $min_confidence
        RETURN elementId(r) AS id,
               a.name AS source,
               type(r) AS type,
               b.name AS target,
               properties(r) AS properties
        ORDER BY coalesce(r.support_count, 1) DESC, coalesce(r.confidence, 1.0) DESC
        """,
        min_confidence=min_confidence,
        database_=NEO4J_DATABASE,
    )
    return {
        "nodes": [dict(record) for record in node_records],
        "relationships": [dict(record) for record in rel_records],
    }


def tokenize(text: str, graph: dict[str, list[dict[str, Any]]]) -> set[str]:
    lowered = text.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9_]+", lowered))
    tokens.update(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    for node in graph["nodes"]:
        name = str(node.get("name") or "")
        if name and name in text:
            tokens.add(name)
    return {token for token in tokens if token}


def retrieve_facts(question: str, graph: dict[str, list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    tokens = tokenize(question, graph)
    matched_names = {
        str(node.get("name"))
        for node in graph["nodes"]
        if node.get("name") and str(node.get("name")) in question
    }

    scored = []
    for rel in graph["relationships"]:
        source = str(rel.get("source") or "")
        target = str(rel.get("target") or "")
        rel_type = str(rel.get("type") or "")
        properties = rel.get("properties", {})
        evidence = str(properties.get("evidence") or "")
        haystack = f"{source} {target} {rel_type} {evidence}".lower()

        score = 0.0
        if source in matched_names or target in matched_names:
            score += 8
        for token in tokens:
            if token.lower() in haystack:
                score += 3 if len(token) >= 3 else 1
        for char in set(question):
            if "\u4e00" <= char <= "\u9fff" and char in haystack:
                score += 0.2

        confidence = float(properties.get("confidence") or 0.75)
        support_count = int(properties.get("support_count") or 1)
        if evidence:
            score += 1.5
        score *= max(0.1, min(confidence, 1.0))
        score += min(support_count, 5) * 0.4
        if score > 0:
            item = dict(rel)
            item["_score"] = round(score, 6)
            scored.append((score, item))

    if not scored:
        scored = [(1.0, dict(rel, _score=1.0)) for rel in graph["relationships"][:limit]]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [rel for _, rel in scored[:limit]]


def make_vector_formatter():
    def formatter(record):
        node = record.get("node")
        score = record.get("score")
        content = json.dumps(
            {
                "id": node.get("id") if node else "",
                "source": node.get("source") if node else "",
                "document": node.get("document") if node else "",
                "page": node.get("page") if node else "",
                "page_start": node.get("page_start") if node else None,
                "page_end": node.get("page_end") if node else None,
                "source_ref": node.get("source_ref") if node else "",
                "index": node.get("index") if node else "",
                "score": score,
                "text": node.get("text") if node else "",
            },
            ensure_ascii=False,
        )
        from neo4j_graphrag.types import RetrieverResultItem

        return RetrieverResultItem(
            content=content,
            metadata={"score": score},
        )

    return formatter


def retrieve_vector_context(driver, embedder, question: str, limit: int) -> list[dict[str, Any]]:
    retriever = VectorRetriever(
        driver,
        VECTOR_INDEX_NAME,
        embedder=embedder,
        result_formatter=make_vector_formatter(),
        neo4j_database=NEO4J_DATABASE,
    )
    result = retriever.search(query_text=question, top_k=limit)
    items = []
    for rank, item in enumerate(result.items, start=1):
        try:
            parsed = json.loads(str(item.content))
        except json.JSONDecodeError:
            parsed = {"text": str(item.content)}
        parsed["rank"] = rank
        items.append(parsed)
    return items


def item_pages(item: dict[str, Any]) -> set[str]:
    page = item.get("page")
    pages = expand_pages(page)
    if pages:
        return pages
    start = item.get("page_start")
    end = item.get("page_end")
    if start is None:
        return set()
    if end is None:
        end = start
    try:
        return {str(page_no) for page_no in range(int(start), int(end) + 1)}
    except (TypeError, ValueError):
        return {normalize_page(start)}


def relation_pages(rel: dict[str, Any]) -> set[str]:
    return item_pages(rel.get("properties", {}))


def item_chunk_refs(item: dict[str, Any]) -> set[str]:
    refs = set()
    if item.get("source_ref"):
        refs.add(str(item["source_ref"]))
    if item.get("id"):
        refs.add(str(item["id"]))
    source = item.get("source")
    index = item.get("index")
    if source and index:
        refs.add(f"{source}::{index}")
    return refs


def relation_triple(rel: dict[str, Any]) -> tuple[str, str, str] | None:
    return normalize_triple(
        {
            "source": rel.get("source"),
            "type": rel.get("type"),
            "target": rel.get("target"),
        }
    )


def relation_entities(rel: dict[str, Any]) -> set[str]:
    return {normalize_key(rel.get("source")), normalize_key(rel.get("target"))} - {""}


def build_citations(vector_items: list[dict[str, Any]], rels: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for item in vector_items[:k]:
        citations.append(
            {
                "type": "chunk",
                "document": item.get("document"),
                "page": item.get("page"),
                "source_ref": item.get("source_ref"),
            }
        )
    for rel in rels[:k]:
        props = rel.get("properties", {})
        citations.append(
            {
                "type": "relationship",
                "document": props.get("document"),
                "page": props.get("page"),
                "source_ref": props.get("source_ref"),
            }
        )
    unique = []
    seen = set()
    for citation in citations:
        key = (citation.get("document"), citation.get("page"), citation.get("source_ref"))
        if key not in seen:
            seen.add(key)
            unique.append(citation)
    return unique


def page_recall(vector_items: list[dict[str, Any]], expected_pages: set[str], k: int) -> float | None:
    if not expected_pages:
        return None
    retrieved = set()
    for item in vector_items[:k]:
        retrieved.update(item_pages(item))
    return safe_div(len(expected_pages & retrieved), len(expected_pages))


def chunk_recall(vector_items: list[dict[str, Any]], expected_chunks: set[str], k: int) -> float | None:
    if not expected_chunks:
        return None
    retrieved = set()
    for item in vector_items[:k]:
        retrieved.update(item_chunk_refs(item))
    return safe_div(len(expected_chunks & retrieved), len(expected_chunks))


def reciprocal_rank_page(vector_items: list[dict[str, Any]], expected_pages: set[str], k: int) -> float | None:
    if not expected_pages:
        return None
    for index, item in enumerate(vector_items[:k], start=1):
        if item_pages(item) & expected_pages:
            return 1.0 / index
    return 0.0


def reciprocal_rank_chunk(vector_items: list[dict[str, Any]], expected_chunks: set[str], k: int) -> float | None:
    if not expected_chunks:
        return None
    for index, item in enumerate(vector_items[:k], start=1):
        if item_chunk_refs(item) & expected_chunks:
            return 1.0 / index
    return 0.0


def ndcg_page(vector_items: list[dict[str, Any]], page_relevance: dict[str, float], k: int) -> float | None:
    if not page_relevance:
        return None
    gains = []
    for item in vector_items[:k]:
        pages = item_pages(item)
        gains.append(max((page_relevance.get(page, 0.0) for page in pages), default=0.0))
    return ndcg_at_k(gains, list(page_relevance.values()), k)


def ndcg_chunk(vector_items: list[dict[str, Any]], chunk_relevance: dict[str, float], k: int) -> float | None:
    if not chunk_relevance:
        return None
    gains = []
    for item in vector_items[:k]:
        refs = item_chunk_refs(item)
        gains.append(max((chunk_relevance.get(ref, 0.0) for ref in refs), default=0.0))
    return ndcg_at_k(gains, list(chunk_relevance.values()), k)


def citation_metrics(
    vector_items: list[dict[str, Any]],
    rels: list[dict[str, Any]],
    expected_pages: set[str],
    k: int,
) -> dict[str, float | None]:
    if not expected_pages:
        return {f"citation_hit@{k}": None, f"citation_accuracy@{k}": None}
    citations = build_citations(vector_items, rels, k)
    cited_pages = set()
    for citation in citations:
        cited_pages.update(expand_pages(citation.get("page")))
    hit = 1.0 if cited_pages & expected_pages else 0.0
    accuracy = safe_div(len(cited_pages & expected_pages), len(cited_pages)) if cited_pages else 0.0
    return {f"citation_hit@{k}": hit, f"citation_accuracy@{k}": accuracy}


def entity_recall(rels: list[dict[str, Any]], expected_entities: set[str], k: int) -> float | None:
    if not expected_entities:
        return None
    retrieved = set()
    for rel in rels[:k]:
        retrieved.update(relation_entities(rel))
    return safe_div(len(expected_entities & retrieved), len(expected_entities))


def triple_recall(rels: list[dict[str, Any]], expected_triples: set[tuple[str, str, str]], k: int) -> float | None:
    if not expected_triples:
        return None
    retrieved = {triple for rel in rels[:k] if (triple := relation_triple(rel))}
    return safe_div(len(expected_triples & retrieved), len(expected_triples))


def relation_precision(rels: list[dict[str, Any]], expected_triples: set[tuple[str, str, str]], k: int) -> float | None:
    if not expected_triples:
        return None
    retrieved = [triple for rel in rels[:k] if (triple := relation_triple(rel))]
    if not retrieved:
        return 0.0
    correct = sum(1 for triple in retrieved if triple in expected_triples)
    return safe_div(correct, len(retrieved))


def relation_evidence_coverage(rels: list[dict[str, Any]], k: int) -> float | None:
    selected = rels[:k]
    if not selected:
        return None
    supported = 0
    for rel in selected:
        props = rel.get("properties", {})
        if props.get("evidence") and props.get("page"):
            supported += 1
    return safe_div(supported, len(selected))


def keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    hits = sum(1 for keyword in keywords if str(keyword) and str(keyword) in answer)
    return safe_div(hits, len(keywords))


def keyword_hallucination_rate(answer: str, forbidden_keywords: list[str]) -> float | None:
    if not forbidden_keywords:
        return None
    hits = sum(1 for keyword in forbidden_keywords if str(keyword) and str(keyword) in answer)
    return safe_div(hits, len(forbidden_keywords))


def build_context(vector_items: list[dict[str, Any]], rels: list[dict[str, Any]], vector_k: int, graph_k: int) -> str:
    parts = []
    for item in vector_items[:vector_k]:
        source_label = item.get("document") or item.get("source") or "unknown"
        if item.get("page"):
            source_label = f"{source_label} 第 {item.get('page')} 页"
        parts.append(f"[文档] {source_label}\n{item.get('text') or ''}")
    for rel in rels[:graph_k]:
        props = rel.get("properties", {})
        evidence = props.get("evidence")
        fact = f"{rel.get('source')} --{rel.get('type')}--> {rel.get('target')}"
        if evidence:
            fact += f"。证据：{evidence}"
        parts.append(f"[图谱] {fact}")
    return "\n\n".join(parts)


def filter_context_mode(
    vector_items: list[dict[str, Any]],
    rels: list[dict[str, Any]],
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if mode == "vector":
        return vector_items, []
    if mode == "graph":
        return [], rels
    return vector_items, rels


def call_answer_llm(question: str, context: str) -> str:
    if LLM_PROVIDER != "deepseek" or not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-your-"):
        return ""
    from openai import OpenAI

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你负责给出简洁、可信、基于资料的中文回答。"},
            {
                "role": "user",
                "content": (
                    "请只依据下面的知识库上下文回答问题；如果上下文不足，要说明现有资料不足以确认。\n\n"
                    f"知识库上下文：\n{context}\n\n问题：\n{question}"
                ),
            },
        ],
        extra_body=DEEPSEEK_EXTRA_BODY,
    )
    return response.choices[0].message.content or ""


def call_judge_llm(sample: dict[str, Any], answer: str, context: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    if LLM_PROVIDER != "deepseek" or not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-your-"):
        return {}
    from openai import OpenAI

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    rubric = {
        "question": sample.get("question"),
        "expected_pages": sample.get("expected_pages", []),
        "expected_entities": sample.get("expected_entities", []),
        "expected_triples": sample.get("expected_triples", []),
        "answer": answer,
        "citations": citations,
        "context": context[:6000],
    }
    prompt = f"""
你是RAG评测员。请根据标注、引用页和上下文判断回答质量，只输出JSON。
字段：
- answer_supported: 0或1，回答主要结论是否被上下文支持
- citation_accurate: 0或1，引用页是否能支撑回答
- missing_evidence: 0或1，是否有应答但缺少证据的关键点
- hallucination: 0或1，是否出现上下文没有支持的事实
- reason: 简短中文原因

输入：
{json.dumps(rubric, ensure_ascii=False)}
""".strip()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你只输出严格JSON，不要输出解释性前后缀。"},
            {"role": "user", "content": prompt},
        ],
        extra_body=DEEPSEEK_EXTRA_BODY,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        return json.loads(match.group(0)) if match else {}


def evaluate_sample(
    sample: dict[str, Any],
    vector_items: list[dict[str, Any]],
    rels: list[dict[str, Any]],
    ks: list[int],
    graph_ks: list[int],
    run_answer: bool,
    judge_answer: bool,
    mode: str = "hybrid",
) -> dict[str, Any]:
    expected_pages = normalize_pages(sample.get("expected_pages"))
    expected_chunks = normalize_chunk_refs(sample.get("expected_chunks") or sample.get("expected_source_refs"))
    expected_entities = {normalize_key(value) for value in sample.get("expected_entities", []) if value}
    expected_triples = normalize_triples(sample.get("expected_triples"))
    page_relevance = parse_relevance(sample.get("relevant_pages"), expected_pages)
    chunk_relevance = parse_relevance(sample.get("relevant_chunks"), expected_chunks)
    vector_items, rels = filter_context_mode(vector_items, rels, mode)

    metrics: dict[str, float | None] = {}
    for k in ks:
        metrics[f"page_recall@{k}"] = metric_value(page_recall(vector_items, expected_pages, k))
        metrics[f"page_hit@{k}"] = None if not expected_pages else (1.0 if metrics[f"page_recall@{k}"] and metrics[f"page_recall@{k}"] > 0 else 0.0)
        metrics[f"chunk_recall@{k}"] = metric_value(chunk_recall(vector_items, expected_chunks, k))
        metrics[f"mrr_page@{k}"] = metric_value(reciprocal_rank_page(vector_items, expected_pages, k))
        metrics[f"mrr_chunk@{k}"] = metric_value(reciprocal_rank_chunk(vector_items, expected_chunks, k))
        metrics[f"ndcg_page@{k}"] = metric_value(ndcg_page(vector_items, page_relevance, k))
        metrics[f"ndcg_chunk@{k}"] = metric_value(ndcg_chunk(vector_items, chunk_relevance, k))
        metrics.update({key: metric_value(value) for key, value in citation_metrics(vector_items, rels, expected_pages, k).items()})

    for k in graph_ks:
        metrics[f"entity_recall@{k}"] = metric_value(entity_recall(rels, expected_entities, k))
        metrics[f"triple_recall@{k}"] = metric_value(triple_recall(rels, expected_triples, k))
        precision = relation_precision(rels, expected_triples, k)
        metrics[f"relation_precision@{k}"] = metric_value(precision)
        metrics[f"noise_edge_rate@{k}"] = metric_value(None if precision is None else 1.0 - precision)
        metrics[f"relation_evidence_coverage@{k}"] = metric_value(relation_evidence_coverage(rels, k))

    answer = ""
    judge = {}
    context = ""
    citations = build_citations(vector_items, rels, max(ks) if ks else 5)
    if run_answer:
        context = build_context(vector_items, rels, max(ks) if ks else 5, max(graph_ks) if graph_ks else 10)
        answer = call_answer_llm(sample["question"], context)
        expected_keywords = sample.get("expected_answer_keywords", [])
        forbidden_keywords = sample.get("forbidden_answer_keywords", [])
        supported = keyword_coverage(answer, expected_keywords)
        metrics["answer_supported_rate"] = metric_value(supported)
        metrics["missing_evidence_rate"] = metric_value(None if supported is None else 1.0 - supported)
        metrics["hallucination_rate"] = metric_value(keyword_hallucination_rate(answer, forbidden_keywords))
        if judge_answer:
            judge = call_judge_llm(sample, answer, context, citations)
            if judge:
                metrics["judge_answer_supported_rate"] = float(judge.get("answer_supported", 0))
                metrics["judge_citation_accuracy"] = float(judge.get("citation_accurate", 0))
                metrics["judge_missing_evidence_rate"] = float(judge.get("missing_evidence", 0))
                metrics["judge_hallucination_rate"] = float(judge.get("hallucination", 0))

    return {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "metrics": metrics,
        "expected": {
            "pages": sorted(expected_pages, key=lambda x: int(x) if x.isdigit() else x),
            "chunks": sorted(expected_chunks),
            "entities": sorted(expected_entities),
            "triples": [list(triple) for triple in sorted(expected_triples)],
        },
        "retrieved": {
            "vector": [
                {
                    "rank": item.get("rank"),
                    "document": item.get("document"),
                    "page": item.get("page"),
                    "source_ref": item.get("source_ref"),
                    "score": item.get("score"),
                    "preview": (item.get("text") or "")[:160],
                }
                for item in vector_items
            ],
            "relationships": [
                {
                    "rank": index,
                    "source": rel.get("source"),
                    "type": rel.get("type"),
                    "target": rel.get("target"),
                    "score": rel.get("_score"),
                    "page": (rel.get("properties") or {}).get("page"),
                    "confidence": (rel.get("properties") or {}).get("confidence"),
                    "evidence": ((rel.get("properties") or {}).get("evidence") or "")[:160],
                }
                for index, rel in enumerate(rels, start=1)
            ],
            "citations": citations,
        },
        "answer": answer,
        "judge": judge,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    for result in results:
        for key, value in result["metrics"].items():
            if value is not None:
                buckets.setdefault(key, []).append(float(value))
    summary = {}
    for key, values in sorted(buckets.items()):
        summary[key] = {
            "mean": round(sum(values) / len(values), 6),
            "count": len(values),
        }
    return summary


def write_outputs(report: dict[str, Any], output_path: Path, csv_path: Path | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not csv_path:
        return
    keys = sorted({key for row in report["results"] for key in row["metrics"]})
    lines = [",".join(["id", "question"] + keys)]
    for row in report["results"]:
        values = [str(row["id"]), json.dumps(row["question"], ensure_ascii=False)]
        for key in keys:
            value = row["metrics"].get(key)
            values.append("" if value is None else str(value))
        lines.append(",".join(value.replace("\n", " ") for value in values))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG vector, graph, and answer retrieval quality.")
    parser.add_argument("--dataset", required=True, help="JSONL file with evaluation questions and expected evidence.")
    parser.add_argument("--output", default=None, help="JSON report path. Defaults to output/retrieval_eval_<time>.json.")
    parser.add_argument("--csv", default=None, help="Optional CSV summary path.")
    parser.add_argument("--ks", default="3,5,10", help="Vector/citation K values, e.g. 3,5,10.")
    parser.add_argument("--graph-ks", default=None, help="Graph K values. Defaults to --ks.")
    parser.add_argument("--min-rel-confidence", type=float, default=0.70, help="Minimum relationship confidence used by graph retrieval.")
    parser.add_argument("--mode", choices=["hybrid", "vector", "graph", "all"], default="hybrid", help="Which evidence source to evaluate. Use all to compare vector, graph, and hybrid in one run.")
    parser.add_argument("--run-answer", action="store_true", help="Also call the configured LLM and compute answer-level metrics.")
    parser.add_argument("--judge-answer", action="store_true", help="Use the configured LLM as a judge for answer support and hallucination.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output) if args.output else Path("output") / f"graphrag_mode_compare_{run_id}.json"
        command = [
            sys.executable,
            "scripts/compare_graphrag_modes.py",
            "--dataset",
            args.dataset,
            "--ks",
            args.ks,
            "--graph-ks",
            args.graph_ks or args.ks,
            "--min-rel-confidence",
            str(args.min_rel_confidence),
            "--output",
            str(output_path),
        ]
        if args.judge_answer:
            command.append("--judge-answer")
        elif args.run_answer:
            command.append("--run-answer")
        if args.csv:
            command.extend(["--csv", args.csv])
        subprocess.run(command, check=True)
        return

    dataset_path = Path(args.dataset)
    rows = load_dataset(dataset_path)
    if not rows:
        raise RuntimeError(f"No evaluation rows found: {dataset_path}")

    ks = sorted(set(parse_int_list(args.ks)))
    graph_ks = sorted(set(parse_int_list(args.graph_ks or args.ks)))
    max_vector_k = max(ks)
    max_graph_k = max(graph_ks) if graph_ks else 0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path("output") / f"retrieval_eval_{run_id}.json"
    csv_path = Path(args.csv) if args.csv else None

    results = []
    embedder = build_embedder()
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        graph = load_graph(driver, args.min_rel_confidence) if max_graph_k > 0 else {"nodes": [], "relationships": []}
        for index, row in enumerate(rows, start=1):
            question = str(row.get("question") or "").strip()
            if not question:
                raise ValueError(f"Missing question for row {row.get('id')}")
            print(f"[{index}/{len(rows)}] {row.get('id')} {question}")
            vector_items = retrieve_vector_context(driver, embedder, question, max_vector_k)
            rels = retrieve_facts(question, graph, max_graph_k) if max_graph_k > 0 else []
            result = evaluate_sample(
                row,
                vector_items,
                rels,
                ks,
                graph_ks,
                run_answer=args.run_answer or args.judge_answer,
                judge_answer=args.judge_answer,
                mode=args.mode,
            )
            results.append(result)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "neo4j": {
            "uri": NEO4J_URI,
            "database": NEO4J_DATABASE,
            "vector_index": VECTOR_INDEX_NAME,
            "min_relationship_confidence": args.min_rel_confidence,
            "mode": args.mode,
        },
        "ks": ks,
        "graph_ks": graph_ks,
        "summary": summarize(results),
        "results": results,
    }
    write_outputs(report, output_path, csv_path)

    primary_key = "page_recall@5" if 5 in ks else f"page_recall@{ks[-1]}"
    primary = report["summary"].get(primary_key, {})
    print("\nEvaluation finished.")
    print(f"Questions: {len(results)}")
    if primary:
        print(f"Primary {primary_key}: {primary['mean']:.4f} over {primary['count']} labeled questions")
    print(f"Report: {output_path}")
    if csv_path:
        print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
