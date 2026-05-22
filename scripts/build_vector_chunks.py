import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

from build_kg_with_deepseek import Chunk, iter_input_files, read_document, split_text
from local_embeddings import build_embedder, embedding_dimensions
from source_metadata import clean_document_name, page_label, source_ref


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunk_vector_index")
CHUNK_LABEL = os.getenv("CHUNK_LABEL", "Chunk")
CHUNK_TEXT_PROPERTY = os.getenv("CHUNK_TEXT_PROPERTY", "text")
CHUNK_EMBEDDING_PROPERTY = os.getenv("CHUNK_EMBEDDING_PROPERTY", "embedding")
EMBEDDING_DIMENSIONS = embedding_dimensions()


def create_chunk_constraints(driver) -> None:
    driver.execute_query(
        f"CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:{CHUNK_LABEL}) REQUIRE c.id IS UNIQUE",
        database_=NEO4J_DATABASE,
    )
    driver.execute_query(
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        database_=NEO4J_DATABASE,
    )


def drop_vector_index(driver) -> None:
    driver.execute_query(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS", database_=NEO4J_DATABASE)


def write_chunk(driver, chunk: Chunk, embedding: list[float]) -> None:
    source = chunk.source
    index = chunk.index
    chunk_id = f"{source}::{index}"
    document = chunk.document or clean_document_name(source)
    page = page_label(chunk.page_start, chunk.page_end)
    ref = source_ref(document, chunk.page_start, chunk.page_end, index)
    driver.execute_query(
        f"""
        MERGE (doc:Document {{id: $document}})
        SET doc.name = $document,
            doc.source = $source
        MERGE (chunk:{CHUNK_LABEL} {{id: $chunk_id}})
        SET chunk.{CHUNK_TEXT_PROPERTY} = $text,
            chunk.source = $source,
            chunk.document = $document,
            chunk.index = $index,
            chunk.page_start = $page_start,
            chunk.page_end = $page_end,
            chunk.page = $page,
            chunk.source_ref = $source_ref,
            chunk.vector_kg = true
        WITH doc, chunk
        CALL db.create.setNodeVectorProperty(chunk, $embedding_property, $embedding)
        MERGE (chunk)-[:FROM_DOCUMENT]->(doc)
        """,
        source=source,
        document=document,
        chunk_id=chunk_id,
        text=chunk.text,
        index=index,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        page=page,
        source_ref=ref,
        embedding_property=CHUNK_EMBEDDING_PROPERTY,
        embedding=embedding,
        database_=NEO4J_DATABASE,
    )


def build(input_path: Path, reset: bool = False) -> None:
    files = iter_input_files(input_path)
    if not files:
        raise FileNotFoundError(f"No supported files found: {input_path}")

    embedder = build_embedder()
    total_chunks = 0

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        create_chunk_constraints(driver)

        if reset:
            drop_vector_index(driver)
            driver.execute_query(
                f"MATCH (c:{CHUNK_LABEL}) WHERE c.vector_kg = true DETACH DELETE c",
                database_=NEO4J_DATABASE,
            )

        for file_path in files:
            print(f"\nChunking: {file_path}")
            text = read_document(file_path)
            chunks = split_text(text, source=file_path.name)
            for chunk in chunks:
                embedding = embedder.embed_query(chunk.text)
                write_chunk(driver, chunk, embedding)
                total_chunks += 1
                print(f"- chunk {chunk.index}/{len(chunks)}")

        create_vector_index(
            driver,
            VECTOR_INDEX_NAME,
            label=CHUNK_LABEL,
            embedding_property=CHUNK_EMBEDDING_PROPERTY,
            dimensions=EMBEDDING_DIMENSIONS,
            similarity_fn="cosine",
            fail_if_exists=False,
            neo4j_database=NEO4J_DATABASE,
        )

    print("\nVector chunk build finished.")
    print(f"Files: {len(files)}")
    print(f"Chunks: {total_chunks}")
    print(f"Vector index: {VECTOR_INDEX_NAME}")
    print(f"Embedding dimensions: {EMBEDDING_DIMENSIONS}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Chunk nodes, embeddings, and a Neo4j vector index.")
    parser.add_argument("--input", required=True, help="A TXT/PDF file or folder containing TXT/PDF files.")
    parser.add_argument("--reset", action="store_true", help="Delete existing vector demo chunks before rebuilding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(Path(args.input), reset=args.reset)


if __name__ == "__main__":
    main()
