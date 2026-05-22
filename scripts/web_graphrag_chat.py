import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever

try:
    from local_embeddings import build_embedder, embedding_dimensions
except ImportError:
    from scripts.local_embeddings import build_embedder, embedding_dimensions


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_EXTRA_BODY = (
    {"thinking": {"type": "disabled"}}
    if DEEPSEEK_MODEL.startswith("deepseek-v4")
    else None
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", DEEPSEEK_MODEL if LLM_PROVIDER == "deepseek" else "gpt-5")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunk_vector_index")
LOCAL_EMBEDDING_DIMENSIONS = embedding_dimensions()

CACHE_PATH = Path("output/local_graph_cache.json")
HOST = "127.0.0.1"
PORT = 7860

GRAPH = {"nodes": [], "relationships": []}
GRAPH_SOURCE = "empty"


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>石化 GraphRAG 问答</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #6b7c93;
      --line: #d9e2ec;
      --brand: #0b7285;
      --brand-dark: #07505f;
      --assistant: #ffffff;
      --user: #dff6f0;
      --fact: #f8fafc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    .app {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      background: #102a43;
      color: #f0f4f8;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    aside h1 {
      font-size: 22px;
      margin: 0;
      line-height: 1.25;
    }
    .status {
      background: rgba(255,255,255,.08);
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 8px;
      padding: 14px;
      font-size: 13px;
      line-height: 1.6;
    }
    .status strong { color: #a7f3d0; }
    .side-actions {
      display: grid;
      gap: 10px;
    }
    button {
      border: 0;
      border-radius: 8px;
      padding: 10px 13px;
      font: inherit;
      cursor: pointer;
      background: var(--brand);
      color: white;
    }
    button:hover { background: var(--brand-dark); }
    button.secondary {
      background: rgba(255,255,255,.12);
      color: white;
    }
    button.secondary:hover { background: rgba(255,255,255,.18); }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      height: 100vh;
    }
    .topbar {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 14px 22px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .topbar h2 {
      margin: 0;
      font-size: 18px;
    }
    .topbar span {
      color: var(--muted);
      font-size: 13px;
    }
    .chat {
      overflow: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .message {
      max-width: 860px;
      padding: 14px 16px;
      border-radius: 8px;
      line-height: 1.65;
      white-space: pre-wrap;
      box-shadow: 0 1px 2px rgba(16,42,67,.08);
    }
    .message.user {
      align-self: flex-end;
      background: var(--user);
    }
    .message.assistant {
      align-self: flex-start;
      background: var(--assistant);
      border: 1px solid var(--line);
    }
    .facts {
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }
    .facts::before {
      content: "引用资料";
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
    }
    .fact {
      background: var(--fact);
      border-left: 4px solid var(--brand);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 13px;
      color: #334e68;
    }
    form {
      background: var(--panel);
      border-top: 1px solid var(--line);
      padding: 16px 22px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
    }
    textarea {
      resize: none;
      height: 52px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      font: inherit;
      line-height: 1.45;
      outline: none;
    }
    textarea:focus {
      border-color: var(--brand);
      box-shadow: 0 0 0 3px rgba(11,114,133,.14);
    }
    @media (max-width: 860px) {
      .app { grid-template-columns: 1fr; }
      aside { display: none; }
      main { height: 100vh; }
      form { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>石化 GraphRAG<br />问答 Demo</h1>
      <div class="status" id="status">正在读取图谱状态...</div>
      <div class="side-actions">
        <button id="refresh" type="button">从 Neo4j 刷新图谱</button>
        <button id="examples" class="secondary" type="button">填入示例问题</button>
      </div>
      <div class="status">
        <strong>工作方式</strong><br />
        后台自动检索 Neo4j 知识图谱和文档 Chunk，把相关上下文交给模型回答；页面只展示简洁答案和引用资料。
      </div>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h2>知识图谱问答</h2>
          <span>Neo4j 图谱 + VectorRetriever + DeepSeek</span>
        </div>
      </div>
      <section class="chat" id="chat">
        <div class="message assistant">你好，我会根据本地知识库辅助回答，并在答案下方列出涉及的资料和页码。</div>
      </section>
      <form id="form">
        <textarea id="question" placeholder="输入你的问题，例如：催化裂化装置有哪些关键设备和风险？"></textarea>
        <button type="submit">发送</button>
      </form>
    </main>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const form = document.getElementById("form");
    const question = document.getElementById("question");
    const statusBox = document.getElementById("status");

    function addMessage(role, text, citations = []) {
      const box = document.createElement("div");
      box.className = `message ${role}`;
      box.textContent = text;
      if (citations.length) {
        const list = document.createElement("div");
        list.className = "facts";
        citations.forEach(f => {
          const item = document.createElement("div");
          item.className = "fact";
          item.textContent = f;
          list.appendChild(item);
        });
        box.appendChild(list);
      }
      chat.appendChild(box);
      chat.scrollTop = chat.scrollHeight;
    }

    async function loadStatus() {
      const res = await fetch("/api/status");
      const data = await res.json();
      statusBox.innerHTML = `<strong>${data.source}</strong><br />节点：${data.nodes}<br />关系：${data.relationships}<br />LLM：${data.llm_status}`;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;
      addMessage("user", text);
      question.value = "";
      addMessage("assistant", "正在检索知识库...");
      const pending = chat.lastChild;
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text })
      });
      const data = await res.json();
      pending.remove();
      addMessage("assistant", data.answer, data.citations || []);
    });

    document.getElementById("refresh").addEventListener("click", async () => {
      statusBox.textContent = "正在从 Neo4j 刷新...";
      const res = await fetch("/api/refresh", { method: "POST" });
      const data = await res.json();
      if (!data.ok) addMessage("assistant", data.error || "刷新失败");
      await loadStatus();
    });

    document.getElementById("examples").addEventListener("click", () => {
      question.value = "催化裂化装置有哪些关键设备、产品、风险和控制参数？";
      question.focus();
    });

    question.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    loadStatus();
  </script>
</body>
</html>
"""


def load_graph_from_neo4j() -> dict:
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
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
              AND coalesce(r.confidence, 1.0) >= 0.70
            RETURN elementId(r) AS id,
                   a.name AS source,
                   type(r) AS type,
                   b.name AS target,
                   properties(r) AS properties
            ORDER BY coalesce(r.support_count, 1) DESC, coalesce(r.confidence, 1.0) DESC
            """,
            database_=NEO4J_DATABASE,
        )

    return {
        "nodes": [dict(record) for record in node_records],
        "relationships": [dict(record) for record in rel_records],
    }


def save_cache(graph: dict) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def refresh_graph() -> tuple[bool, str]:
    global GRAPH, GRAPH_SOURCE
    try:
        GRAPH = load_graph_from_neo4j()
        save_cache(GRAPH)
        GRAPH_SOURCE = "Neo4j 已加载到本地内存"
        return True, ""
    except Exception as exc:
        cached = load_cache()
        if cached:
            GRAPH = cached
            GRAPH_SOURCE = "本地缓存"
            return True, f"Neo4j 暂不可用，已使用本地缓存：{exc}"
        GRAPH = {"nodes": [], "relationships": []}
        GRAPH_SOURCE = "未加载"
        return False, str(exc)


def is_llm_enabled() -> bool:
    if LLM_PROVIDER == "deepseek":
        return bool(DEEPSEEK_API_KEY) and not DEEPSEEK_API_KEY.startswith("sk-your-")
    return bool(OPENAI_API_KEY) and not OPENAI_API_KEY.startswith("sk-your-")


def llm_status() -> str:
    if not is_llm_enabled():
        return "鏈惎鐢紝浣跨敤鏈湴妯℃澘鍥炵瓟"
    if LLM_PROVIDER == "deepseek":
        return f"DeepSeek / {DEEPSEEK_MODEL}"
    return f"OpenAI / {LLM_MODEL}"


def tokenize(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9_]+", lowered))
    tokens.update(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    for node in GRAPH["nodes"]:
        name = str(node.get("name") or "")
        if name and name in text:
            tokens.add(name)
    return {token for token in tokens if token}


def relation_to_fact(rel: dict) -> str:
    evidence = rel.get("properties", {}).get("evidence")
    fact = f"{rel['source']} --{rel['type']}--> {rel['target']}"
    if evidence:
        fact += f"。证据：{evidence}"
    return fact


def parse_source_page(source: str) -> tuple[str, str | None]:
    match = re.search(r"(.+)_page_(\d+)\.txt$", source)
    if match:
        name = clean_source_name(match.group(1))
        return name, str(int(match.group(2)))
    return clean_source_name(source), None


def item_document_page(item: dict) -> tuple[str, str | None]:
    document = str(item.get("document") or "").strip()
    page = str(item.get("page") or "").strip()
    if document:
        return document, page or None
    return parse_source_page(str(item.get("source") or ""))


def props_document_page(props: dict) -> tuple[str, str | None]:
    document = str(props.get("document") or "").strip()
    page = str(props.get("page") or "").strip()
    if document:
        return document, page or None
    return parse_source_page(str(props.get("source") or ""))


def clean_source_name(source: str) -> str:
    name = re.sub(r"\.(txt|md|pdf)$", "", source, flags=re.IGNORECASE)
    name = name.replace("_", " ")
    name = re.split(r"\b(?:z-library|1lib|z-lib)\b", name, maxsplit=1, flags=re.IGNORECASE)[0]
    name = re.sub(r"\(\s*\d+\s*\)$", "", name).strip()
    name = re.sub(r"\s+", " ", name).strip()
    return name


def citation_key(citation: dict) -> tuple[str, str]:
    return (str(citation.get("source") or ""), str(citation.get("page") or ""))


def retrieve_vector_context(question: str, limit: int = 4) -> list[dict]:
    try:
        embedder = build_embedder()

        def formatter(record):
            node = record.get("node")
            score = record.get("score")
            source = node.get("source") if node else ""
            index = node.get("index") if node else ""
            text = node.get("text") if node else ""
            document = node.get("document") if node else ""
            page = node.get("page") if node else ""
            page_start = node.get("page_start") if node else None
            page_end = node.get("page_end") if node else None
            source_ref = node.get("source_ref") if node else ""
            content = json.dumps(
                {
                    "source": source,
                    "document": document,
                    "page": page,
                    "page_start": page_start,
                    "page_end": page_end,
                    "source_ref": source_ref,
                    "index": index,
                    "score": score,
                    "text": text,
                },
                ensure_ascii=False,
            )
            from neo4j_graphrag.types import RetrieverResultItem

            return RetrieverResultItem(
                content=content,
                metadata={"score": score, "source": source, "index": index},
            )

        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
            retriever = VectorRetriever(
                driver,
                VECTOR_INDEX_NAME,
                embedder=embedder,
                result_formatter=formatter,
                neo4j_database=NEO4J_DATABASE,
            )
            result = retriever.search(query_text=question, top_k=limit)
            items = []
            for item in result.items:
                try:
                    items.append(json.loads(str(item.content)))
                except json.JSONDecodeError:
                    items.append({"source": "", "index": "", "score": None, "text": str(item.content)})
            return items
    except Exception as exc:
        return [{"error": f"VectorRetriever 检索失败：{exc}"}]


def retrieve_facts(question: str, limit: int = 12) -> list[dict]:
    tokens = tokenize(question)
    matched_names = {
        str(node.get("name"))
        for node in GRAPH["nodes"]
        if node.get("name") and str(node.get("name")) in question
    }

    scored = []
    for rel in GRAPH["relationships"]:
        source = str(rel.get("source") or "")
        target = str(rel.get("target") or "")
        rel_type = str(rel.get("type") or "")
        properties = rel.get("properties", {})
        evidence = str(properties.get("evidence") or "")
        haystack = f"{source} {target} {rel_type} {evidence}".lower()

        score = 0
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
            scored.append((score, rel))

    if not scored:
        scored = [(1, rel) for rel in GRAPH["relationships"][:limit]]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [rel for _, rel in scored[:limit]]


def build_citations(vector_items: list[dict], relationships: list[dict]) -> list[dict]:
    citations = []
    for item in vector_items:
        if item.get("error"):
            continue
        document, page = item_document_page(item)
        citations.append(
            {
                "source": document,
                "page": page,
                "chunk": item.get("index"),
                "type": "文档片段",
            }
        )

    for rel in relationships:
        props = rel.get("properties", {})
        document, page = props_document_page(props)
        if document:
            citations.append(
                {
                    "source": document,
                    "page": page,
                    "type": "图谱关系",
                }
            )

    unique = []
    seen = set()
    for citation in citations:
        key = citation_key(citation)
        if key not in seen and citation.get("source"):
            seen.add(key)
            unique.append(citation)

    page_citations = [citation for citation in unique if citation.get("page")]
    if page_citations:
        return page_citations[:8]
    return unique[:8]


def citation_label(citation: dict) -> str:
    source = citation.get("source") or "未知资料"
    page = citation.get("page")
    if page:
        return f"{source}，第 {page} 页"
    return str(source)


def local_answer(question: str, context: str) -> str:
    if not context.strip():
        return "当前知识库没有检索到足够相关的资料。建议补充资料或换一种更具体的问法。"
    return "我检索到了相关资料，但当前大模型未启用。请检查 `.env` 里的 DeepSeek 配置。"


def llm_answer(question: str, context: str) -> str:
    if not is_llm_enabled():
        return local_answer(question, context)

    try:
        from openai import OpenAI

        if LLM_PROVIDER == "deepseek":
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            model = DEEPSEEK_MODEL
        else:
            client = OpenAI(api_key=OPENAI_API_KEY)
            model = LLM_MODEL

        prompt = f"""
你是石化领域问答助手。你可以使用下面的知识库检索结果辅助判断，但不要把知识库原文、三元组或检索列表完整贴给用户。

回答要求：
1. 直接回答用户问题，语言简洁、专业、自然。
2. 优先依据知识库上下文；如果上下文不足，要明确说“现有资料不足以确认”，不要编造。
3. 可以结合你的通用专业知识做解释，但不能把没有依据的内容说成来自资料。
4. 不要在正文末尾展开“知识库事实”“检索结果”“文档片段”等列表。
5. 不要输出引用列表；引用会由系统单独展示。
6. 如果问题涉及风险、安全、操作条件，要提醒以企业规程和现场数据为准。

知识库上下文：
{context}

用户问题：
{question}
""".strip()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你负责给出简洁、可信、基于资料的中文回答。"},
                {"role": "user", "content": prompt},
            ],
            extra_body=DEEPSEEK_EXTRA_BODY if LLM_PROVIDER == "deepseek" else None,
        )
        return response.choices[0].message.content or local_answer(question, context)
    except Exception as exc:
        return local_answer(question, context) + f"\n\nLLM 调用失败：{exc}"


def ask(question: str) -> dict:
    vector_items = retrieve_vector_context(question)
    relationships = retrieve_facts(question)

    context_parts = []
    for item in vector_items:
        if item.get("error"):
            continue
        document, page = item_document_page(item)
        source_label = f"{document} 第 {page} 页" if page else document
        context_parts.append(f"[文档] {source_label}\n{item.get('text') or ''}")

    for rel in relationships:
        context_parts.append(f"[图谱] {relation_to_fact(rel)}")

    context = "\n\n".join(context_parts[:16])
    citations = build_citations(vector_items, relationships)
    return {
        "answer": llm_answer(question, context),
        "citations": [label for item in citations if (label := citation_label(item))],
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/status":
            self._send_json(
                {
                    "source": GRAPH_SOURCE,
                    "nodes": len(GRAPH["nodes"]),
                    "relationships": len(GRAPH["relationships"]),
                    "llm_enabled": is_llm_enabled(),
                    "llm_status": llm_status(),
                }
            )
            return

        if parsed.path == "/api/ask":
            query = parse_qs(parsed.query).get("q", [""])[0]
            self._send_json(ask(query))
            return

        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/ask":
            payload = self._read_json()
            question = str(payload.get("question", "")).strip()
            if not question:
                self._send_json({"answer": "请输入问题。", "citations": []}, 400)
                return
            self._send_json(ask(question))
            return

        if self.path == "/api/refresh":
            ok, error = refresh_graph()
            self._send_json({"ok": ok, "error": error})
            return

        self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    ok, error = refresh_graph()
    if not ok:
        print(f"Graph load failed: {error}")
    print(f"Graph source: {GRAPH_SOURCE}")
    print(f"Nodes: {len(GRAPH['nodes'])}, Relationships: {len(GRAPH['relationships'])}")
    print(f"Open http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
