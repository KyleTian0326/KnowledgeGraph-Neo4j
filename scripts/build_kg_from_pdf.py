import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

from build_kg_from_text import NODE_TYPES, PATTERNS, RELATIONSHIP_TYPES


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


async def build_graph(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    llm = OpenAILLM(
        model_name=LLM_MODEL,
        model_params={
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    )
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    ) as driver:
        driver.verify_connectivity()
        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            schema={
                "node_types": NODE_TYPES,
                "relationship_types": RELATIONSHIP_TYPES,
                "patterns": PATTERNS,
            },
            from_file=True,
            neo4j_database=NEO4J_DATABASE,
            on_error="IGNORE",
        )

        result = await kg_builder.run_async(
            file_path=str(input_path),
            document_metadata={"source": input_path.name},
        )
        print(result)

    await llm.async_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Neo4j knowledge graph from a PDF file.")
    parser.add_argument("--input", required=True, help="PDF file to ingest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(build_graph(Path(args.input)))


if __name__ == "__main__":
    main()
