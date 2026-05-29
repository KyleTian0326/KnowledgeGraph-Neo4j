import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from pypdf import PdfReader

from kg_quality import validate_extraction
from progress import ProgressTracker
from source_metadata import clean_document_name, find_page_spans, page_range_for_offsets, page_label, source_ref


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_EXTRA_BODY = (
    {"thinking": {"type": "disabled"}}
    if DEEPSEEK_MODEL.startswith("deepseek-v4")
    else None
)

NODE_TYPES = [
    "Company",
    "Plant",
    "Unit",
    "Equipment",
    "Material",
    "Product",
    "Process",
    "Parameter",
    "Risk",
    "Standard",
    "Operation",
    "Procedure",
    "Condition",
    "Measure",
    "Cause",
    "Fault",
    "Phenomenon",
    "ControlAction",
    "Purpose",
    "Document",
]

RELATIONSHIP_TYPES = [
    "OWNS",
    "HAS_UNIT",
    "USES_MATERIAL",
    "HAS_COMPONENT",
    "USES_PROCESS",
    "PRODUCES",
    "HAS_EQUIPMENT",
    "CONTROLS",
    "USED_FOR",
    "HAS_RISK",
    "COMPLIES_WITH",
    "HAS_PARAMETER",
    "HAS_QUALITY_INDEX",
    "HAS_RANGE",
    "OPERATES_AT",
    "HAS_CONDITION",
    "REQUIRES",
    "HAS_STEP",
    "PRECEDES",
    "CAUSES",
    "AFFECTS",
    "MITIGATED_BY",
    "MONITORS",
    "LOCATED_IN",
    "PART_OF",
    "REFERS_TO",
]

ALLOWED_LABELS = set(NODE_TYPES)
ALLOWED_REL_TYPES = set(RELATIONSHIP_TYPES)


@dataclass
class Chunk:
    source: str
    index: int
    text: str
    document: str = ""
    page_start: int | None = None
    page_end: int | None = None


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_index}]\n{text}")
    return "\n\n".join(pages)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_file(path)
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    raise ValueError(f"Unsupported file type: {path}")


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for pattern in ("*.txt", "*.md", "*.pdf"):
        files.extend(path.rglob(pattern))
    return sorted(files)


def split_text(text: str, source: str, chunk_size: int = 2600, overlap: int = 250) -> list[Chunk]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    document = clean_document_name(source)
    page_spans = find_page_spans(normalized)
    chunks = []
    start = 0
    index = 1
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        page_start, page_end = page_range_for_offsets(source, normalized, start, end, page_spans)
        chunks.append(
            Chunk(
                source=source,
                index=index,
                text=normalized[start:end],
                document=document,
                page_start=page_start,
                page_end=page_end,
            )
        )
        if end == len(normalized):
            break
        start = max(0, end - overlap)
        index += 1
    return chunks


def extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {content[:300]}")

    return json.loads(content[start : end + 1])


def normalize_extraction(value: dict[str, Any]) -> dict[str, Any]:
    nodes = value.get("nodes") if isinstance(value.get("nodes"), list) else []
    relationships = value.get("relationships") if isinstance(value.get("relationships"), list) else []
    return {"nodes": nodes, "relationships": relationships}


def build_prompt(chunk: Chunk) -> str:
    return f"""
You are a petrochemical knowledge graph extraction engine.
Extract only facts explicitly supported by the provided text.
Return valid JSON only. Do not include Markdown or explanations.

Allowed entity labels:
{", ".join(NODE_TYPES)}

Allowed relationship types:
{", ".join(RELATIONSHIP_TYPES)}

Relationship examples:
- Company OWNS Plant/Unit
- Plant HAS_UNIT Unit
- Unit USES_MATERIAL Material
- Product/Material HAS_COMPONENT Material
- Unit USES_PROCESS Process
- Unit PRODUCES Product
- Unit HAS_EQUIPMENT Equipment
- Equipment CONTROLS Parameter
- Unit HAS_PARAMETER Parameter
- Product/Material HAS_QUALITY_INDEX Parameter
- Parameter HAS_RANGE Condition
- Unit OPERATES_AT Parameter/Condition
- Process HAS_STEP Operation
- Fault/Risk MITIGATED_BY Measure
- Parameter/Condition AFFECTS Process/Product/Equipment
- Unit HAS_RISK Risk
- Unit COMPLIES_WITH Standard

Required JSON shape:
{{
  "nodes": [
    {{"name": "entity name", "label": "entity label", "description": "short description"}}
  ],
  "relationships": [
    {{"source": "source entity name", "type": "relationship type", "target": "target entity name", "evidence": "verbatim evidence from the text", "confidence": 0.0}}
  ]
}}

Rules:
1. Extract only facts clearly stated in this text chunk.
2. Do not infer missing entities, missing relation endpoints, or process links.
3. Entity names must be short canonical terms, not full sentences.
4. Every relationship must have evidence copied from this chunk. Evidence must include enough wording to support both endpoints and the relation.
5. If evidence is a summary rather than a text span, do not output that relationship.
6. Prefer operationally useful petrochemical facts: process units, equipment, materials, products, parameters, conditions, procedures, faults, risks, safeguards, standards.
7. Avoid generic nodes such as "equipment", "process", "risk", "parameter" unless the text names a specific item.
8. Output at most 35 nodes and 45 relationships.
9. If no reliable facts can be extracted, return {{"nodes": [], "relationships": []}}.

Source file: {chunk.source}
Chunk index: {chunk.index}
Document: {chunk.document or clean_document_name(chunk.source)}
Page: {page_label(chunk.page_start, chunk.page_end) or "unknown"}

Text:
{chunk.text}
""".strip()
    return f"""
你是石化领域知识图谱抽取助手。请从给定文本中抽取实体和关系，只输出合法 JSON，不要输出解释。

允许的实体类型：
{", ".join(NODE_TYPES)}

允许的关系类型：
{", ".join(RELATIONSHIP_TYPES)}

关系含义示例：
- Company OWNS Plant/Unit
- Plant HAS_UNIT Unit
- Unit USES_MATERIAL Material
- Unit USES_PROCESS Process
- Unit PRODUCES Product
- Unit HAS_EQUIPMENT Equipment
- Equipment CONTROLS Parameter
- Product USED_FOR Product
- Unit HAS_RISK Risk
- Unit COMPLIES_WITH Standard

输出格式必须是：
{{
  "nodes": [
    {{"name": "实体名称", "label": "实体类型", "description": "简短说明"}}
  ],
  "relationships": [
    {{"source": "源实体名称", "type": "关系类型", "target": "目标实体名称", "evidence": "原文证据"}}
  ]
}}

要求：
1. 只抽取文本中明确出现或强烈表达的事实。
2. 不要编造实体和关系。
3. 实体名称尽量短，例如“催化裂化装置”“分馏塔”“汽油”。
4. evidence 控制在 80 字以内。
5. 最多输出 30 个节点、40 条关系，优先保留最重要的工艺、设备、物料、参数和风险。
6. 如果无法抽取，返回 {{"nodes": [], "relationships": []}}。

来源文件：{chunk.source}
片段编号：{chunk.index}

文本：
{chunk.text}
""".strip()


def completion_kwargs(messages: list[dict[str, str]], json_mode: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if DEEPSEEK_EXTRA_BODY:
        kwargs["extra_body"] = DEEPSEEK_EXTRA_BODY
    return kwargs


def repair_json_with_llm(client: OpenAI, content: str, error: Exception) -> dict[str, Any]:
    repair_prompt = f"""
下面是一段本应为 JSON 的内容，但它无法被 json.loads 解析。
请只返回修复后的合法 JSON，不要解释。
必须保留这个结构：
{{"nodes": [], "relationships": []}}

解析错误：
{error}

原始内容：
{content}
""".strip()
    response = client.chat.completions.create(
        **completion_kwargs(
            [
                {"role": "system", "content": "你只输出合法 JSON。"},
                {"role": "user", "content": repair_prompt},
            ],
            json_mode=True,
        )
    )
    return normalize_extraction(extract_json(response.choices[0].message.content or "{}"))


def extract_chunk(client: OpenAI, chunk: Chunk) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "你只输出合法 JSON，不要输出 Markdown。"},
        {"role": "user", "content": build_prompt(chunk)},
    ]

    last_error: Exception | None = None
    last_content = ""
    for attempt in range(1, 3):
        response = client.chat.completions.create(**completion_kwargs(messages, json_mode=True))
        last_content = response.choices[0].message.content or "{}"
        try:
            extraction = normalize_extraction(extract_json(last_content))
            return validate_extraction(
                extraction,
                chunk.text,
                chunk.source,
                chunk.index,
                ALLOWED_LABELS,
                ALLOWED_REL_TYPES,
            )
        except Exception as exc:
            last_error = exc
            print(f"  ! invalid JSON on attempt {attempt}: {exc}")
            messages.append(
                {
                    "role": "user",
                    "content": "上一次输出不是合法 JSON。请重新抽取，并只输出合法 JSON。",
                }
            )

    try:
        print("  ! trying JSON repair")
        extraction = repair_json_with_llm(client, last_content, last_error or ValueError("invalid JSON"))
        return validate_extraction(
            extraction,
            chunk.text,
            chunk.source,
            chunk.index,
            ALLOWED_LABELS,
            ALLOWED_REL_TYPES,
        )
    except Exception as exc:
        print(f"  ! skip chunk {chunk.index}, JSON repair failed: {exc}")
        return {
            "nodes": [],
            "relationships": [],
            "_error": str(exc),
            "_raw_response_preview": last_content[:1000],
        }


def safe_identifier(value: str, allowed: set[str], fallback: str | None = None) -> str | None:
    if value in allowed:
        return value
    return fallback


def create_constraints(driver) -> None:
    for label in NODE_TYPES:
        driver.execute_query(
            f"""
            MATCH (n:{label})
            WHERE n.canonical_key IS NULL AND n.name IS NOT NULL
            SET n.canonical_key = n.name
            """,
            database_=NEO4J_DATABASE,
        )
        try:
            driver.execute_query(
                f"CREATE CONSTRAINT deepseek_{label}_canonical_key IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.canonical_key IS UNIQUE",
                database_=NEO4J_DATABASE,
            )
        except Exception as exc:
            print(f"  ! unique constraint skipped for {label}: {exc}")
            driver.execute_query(
                f"CREATE INDEX deepseek_{label}_canonical_key_index IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.canonical_key)",
                database_=NEO4J_DATABASE,
            )


def write_node(driver, node: dict[str, Any], source: str, chunk: Chunk | None = None) -> None:
    name = str(node.get("name", "")).strip()
    if not name:
        return
    label = safe_identifier(str(node.get("label", "")).strip(), ALLOWED_LABELS)
    if not label:
        return
    canonical_key = str(node.get("canonical_key") or name).strip()
    description = str(node.get("description", "")).strip()
    aliases = [str(alias).strip() for alias in node.get("aliases", []) if str(alias).strip()]
    support_count = int(node.get("support_count") or 1)
    document = chunk.document if chunk else clean_document_name(source)
    page_start = chunk.page_start if chunk else None
    page_end = chunk.page_end if chunk else None
    driver.execute_query(
        f"""
        MERGE (n:{label} {{canonical_key: $canonical_key}})
        ON CREATE SET n.name = $name,
                      n.created_source = $source
        SET n.name = coalesce(n.name, $name),
            n.description = CASE
                WHEN n.description IS NULL OR n.description = '' THEN $description
                ELSE n.description
            END,
            n.aliases = reduce(
                alias_list = [],
                alias IN coalesce(n.aliases, []) + $aliases |
                CASE WHEN alias IS NULL OR alias = '' OR alias IN alias_list
                    THEN alias_list
                    ELSE alias_list + alias
                END
            ),
            n.support_count = coalesce(n.support_count, 0) + $support_count,
            n.source = coalesce(n.source, $source),
            n.document = coalesce(n.document, $document),
            n.first_page = coalesce(n.first_page, $page_start),
            n.deepseek_kg = true
        """,
        name=name,
        canonical_key=canonical_key,
        description=description,
        aliases=aliases,
        support_count=support_count,
        source=source,
        document=document,
        page_start=page_start,
        database_=NEO4J_DATABASE,
    )


def write_relationship(driver, rel: dict[str, Any], source: str, chunk: Chunk | None = None) -> None:
    source_name = str(rel.get("source", "")).strip()
    target_name = str(rel.get("target", "")).strip()
    if not source_name or not target_name:
        return
    rel_type = safe_identifier(str(rel.get("type", "")).strip(), ALLOWED_REL_TYPES)
    if not rel_type:
        return
    source_key = str(rel.get("source_key") or source_name).strip()
    target_key = str(rel.get("target_key") or target_name).strip()
    source_label = safe_identifier(str(rel.get("source_label", "")).strip(), ALLOWED_LABELS)
    target_label = safe_identifier(str(rel.get("target_label", "")).strip(), ALLOWED_LABELS)
    if not source_label or not target_label:
        return
    evidence = str(rel.get("evidence", "")).strip()
    confidence = float(rel.get("confidence") or 0.0)
    quality_flags = [str(flag) for flag in rel.get("quality_flags", [])]
    support_count = int(rel.get("support_count") or 1)
    document = chunk.document if chunk else clean_document_name(source)
    page_start = chunk.page_start if chunk else None
    page_end = chunk.page_end if chunk else None
    page = page_label(page_start, page_end)
    chunk_index = chunk.index if chunk else None
    ref = source_ref(document, page_start, page_end, chunk_index)
    driver.execute_query(
        f"""
        MATCH (source:{source_label} {{canonical_key: $source_key}})
        MATCH (target:{target_label} {{canonical_key: $target_key}})
        MERGE (source)-[r:{rel_type}]->(target)
        ON CREATE SET r.created_source = $source
        WITH r, (r.evidence IS NULL OR r.confidence IS NULL OR $confidence >= r.confidence) AS use_new_evidence
        SET r.evidence = CASE WHEN use_new_evidence THEN $evidence ELSE r.evidence END,
            r.source = coalesce(r.source, $source),
            r.confidence = CASE WHEN use_new_evidence THEN $confidence ELSE r.confidence END,
            r.quality_flags = $quality_flags,
            r.support_count = coalesce(r.support_count, 0) + $support_count,
            r.document = CASE WHEN use_new_evidence THEN $document ELSE r.document END,
            r.page_start = CASE WHEN use_new_evidence THEN $page_start ELSE r.page_start END,
            r.page_end = CASE WHEN use_new_evidence THEN $page_end ELSE r.page_end END,
            r.page = CASE WHEN use_new_evidence THEN $page ELSE r.page END,
            r.chunk_index = CASE WHEN use_new_evidence THEN $chunk_index ELSE r.chunk_index END,
            r.source_ref = CASE WHEN use_new_evidence THEN $source_ref ELSE r.source_ref END,
            r.documents = CASE
                WHEN $document IS NULL OR $document = '' OR $document IN coalesce(r.documents, [])
                    THEN coalesce(r.documents, [])
                    ELSE coalesce(r.documents, []) + $document
            END,
            r.pages = CASE
                WHEN $page IS NULL OR $page = '' OR $page IN coalesce(r.pages, [])
                    THEN coalesce(r.pages, [])
                    ELSE coalesce(r.pages, []) + $page
            END,
            r.source_refs = CASE
                WHEN $source_ref IS NULL OR $source_ref = '' OR $source_ref IN coalesce(r.source_refs, [])
                    THEN coalesce(r.source_refs, [])
                    ELSE coalesce(r.source_refs, []) + $source_ref
            END,
            r.deepseek_kg = true
        """,
        source_key=source_key,
        target_key=target_key,
        evidence=evidence,
        confidence=confidence,
        quality_flags=quality_flags,
        support_count=support_count,
        source=source,
        document=document,
        page_start=page_start,
        page_end=page_end,
        page=page,
        chunk_index=chunk_index,
        source_ref=ref,
        database_=NEO4J_DATABASE,
    )


def write_extraction(driver, extraction: dict[str, Any], source: str, chunk: Chunk | None = None) -> tuple[int, int]:
    nodes = extraction.get("nodes") or []
    relationships = extraction.get("relationships") or []
    written_nodes = 0
    written_relationships = 0
    for node in nodes:
        if isinstance(node, dict):
            write_node(driver, node, source, chunk=chunk)
            written_nodes += 1
    for rel in relationships:
        if isinstance(rel, dict):
            write_relationship(driver, rel, source, chunk=chunk)
            written_relationships += 1
    return written_nodes, written_relationships


def write_output_json(extractions: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(extractions, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_extractions(output_path: Path | None) -> list[dict[str, Any]]:
    if not output_path or not output_path.exists():
        return []
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def build(input_path: Path, limit: int | None = None, dry_run: bool = False, output_path: Path | None = None) -> None:
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-your-"):
        raise RuntimeError("Please set DEEPSEEK_API_KEY in .env")

    files = iter_input_files(input_path)
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No supported files found: {input_path}")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    total_nodes = 0
    total_relationships = 0
    all_extractions = load_existing_extractions(output_path)
    processed_chunks = {
        (str(item.get("source")), int(item.get("chunk")))
        for item in all_extractions
        if isinstance(item, dict) and item.get("source") and item.get("chunk")
    }
    if processed_chunks:
        print(f"Resume enabled: {len(processed_chunks)} chunks already exist in {output_path}")

    file_chunks: list[tuple[Path, list[Chunk]]] = []
    total_chunks = 0
    initially_done = 0
    print("Counting chunks...")
    for file_path in files:
        text = read_document(file_path)
        chunks = split_text(text, source=file_path.name)
        file_chunks.append((file_path, chunks))
        total_chunks += len(chunks)
        initially_done += sum(1 for chunk in chunks if (chunk.source, chunk.index) in processed_chunks)

    progress = ProgressTracker(total_chunks, label="DeepSeek KG", initial=initially_done)
    progress.render(current="ready", force=True)

    driver = None
    if not dry_run:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        create_constraints(driver)

    try:
        for file_path, chunks in file_chunks:
            print(f"\nProcessing: {file_path}")
            for chunk in chunks:
                if (chunk.source, chunk.index) in processed_chunks:
                    progress.render(current=f"skipped {chunk.source} chunk {chunk.index}", force=True)
                    continue
                progress.render(current=f"{chunk.source} chunk {chunk.index}/{len(chunks)}", force=True)
                extraction = extract_chunk(client, chunk)
                all_extractions.append(
                    {
                        "source": file_path.name,
                        "chunk": chunk.index,
                        "document": chunk.document,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "page": page_label(chunk.page_start, chunk.page_end),
                        "text": chunk.text,
                        "extraction": extraction,
                    }
                )

                if dry_run:
                    node_count = len(extraction.get("nodes") or [])
                    rel_count = len(extraction.get("relationships") or [])
                else:
                    node_count, rel_count = write_extraction(driver, extraction, source=file_path.name, chunk=chunk)

                total_nodes += node_count
                total_relationships += rel_count

                if output_path:
                    write_output_json(all_extractions, output_path)
                progress.advance(current=f"{chunk.source} chunk {chunk.index}/{len(chunks)}")
    finally:
        if driver is not None:
            driver.close()

    progress.finish()

    if output_path:
        write_output_json(all_extractions, output_path)
        print(f"Saved extraction JSON: {output_path}")

    print("\nDeepSeek KG extraction finished.")
    print(f"Files: {len(files)}")
    print(f"Extracted nodes: {total_nodes}")
    print(f"Extracted relationships: {total_relationships}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Neo4j knowledge graph from TXT/PDF files with DeepSeek.")
    parser.add_argument("--input", required=True, help="A TXT/PDF file or a folder containing TXT/PDF files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of files to process.")
    parser.add_argument("--dry-run", action="store_true", help="Extract to JSON without writing to Neo4j.")
    parser.add_argument(
        "--output",
        default="output/deepseek_kg_extraction.json",
        help="Where to save extraction JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(Path(args.input), limit=args.limit, dry_run=args.dry_run, output_path=Path(args.output))


if __name__ == "__main__":
    main()
