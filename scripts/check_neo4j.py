import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def main() -> None:
    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    ) as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            "RETURN 'Neo4j connected' AS message, datetime() AS time",
            database_=NEO4J_DATABASE,
        )
        print(records[0]["message"])
        print(records[0]["time"])


if __name__ == "__main__":
    main()
