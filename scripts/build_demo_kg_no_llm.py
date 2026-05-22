import os
from dataclasses import dataclass

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass(frozen=True)
class Node:
    label: str
    name: str
    description: str = ""


@dataclass(frozen=True)
class Relationship:
    source: str
    rel_type: str
    target: str
    evidence: str


NODES = [
    Node("Company", "中石化某炼化企业", "石化企业主体"),
    Node("Unit", "乙烯装置", "以石脑油为原料生产乙烯和丙烯的生产装置"),
    Node("Material", "石脑油", "乙烯装置的主要裂解原料"),
    Node("Process", "蒸汽裂解", "通过高温裂解生产低碳烯烃的工艺过程"),
    Node("Equipment", "裂解炉", "乙烯装置中的关键设备"),
    Node("Parameter", "炉管出口温度", "裂解炉运行控制参数"),
    Node("Parameter", "稀释蒸汽比", "裂解炉运行控制参数"),
    Node("Product", "乙烯", "乙烯装置主要产品"),
    Node("Product", "丙烯", "乙烯装置联产产品"),
    Node("Product", "聚乙烯", "乙烯下游产品"),
    Node("Product", "聚丙烯", "丙烯下游产品"),
    Node("Risk", "高温泄漏风险", "乙烯装置需要重点防范的安全风险"),
    Node("Standard", "安全生产标准", "装置运行需要符合的标准要求"),
]

RELATIONSHIPS = [
    Relationship("中石化某炼化企业", "OWNS", "乙烯装置", "中石化某炼化企业拥有一套乙烯装置。"),
    Relationship("乙烯装置", "USES_MATERIAL", "石脑油", "乙烯装置以石脑油为主要原料。"),
    Relationship("乙烯装置", "USES_PROCESS", "蒸汽裂解", "乙烯装置通过裂解炉进行蒸汽裂解。"),
    Relationship("乙烯装置", "HAS_EQUIPMENT", "裂解炉", "乙烯装置通过裂解炉进行蒸汽裂解。"),
    Relationship("乙烯装置", "PRODUCES", "乙烯", "乙烯装置生产乙烯和丙烯。"),
    Relationship("乙烯装置", "PRODUCES", "丙烯", "乙烯装置生产乙烯和丙烯。"),
    Relationship("裂解炉", "CONTROLS", "炉管出口温度", "裂解炉需要控制炉管出口温度。"),
    Relationship("裂解炉", "CONTROLS", "稀释蒸汽比", "裂解炉需要控制稀释蒸汽比。"),
    Relationship("乙烯", "USED_FOR", "聚乙烯", "乙烯可以用于生产聚乙烯。"),
    Relationship("丙烯", "USED_FOR", "聚丙烯", "丙烯可以用于生产聚丙烯。"),
    Relationship("乙烯装置", "HAS_RISK", "高温泄漏风险", "乙烯装置应重点防范高温泄漏风险。"),
    Relationship("乙烯装置", "COMPLIES_WITH", "安全生产标准", "乙烯装置应符合安全生产标准。"),
]

ALLOWED_LABELS = {node.label for node in NODES}
ALLOWED_REL_TYPES = {rel.rel_type for rel in RELATIONSHIPS}


def validate_identifier(value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported graph identifier: {value}")
    return value


def reset_demo_graph(driver) -> None:
    driver.execute_query(
        """
        MATCH (n)
        WHERE n.demo = true
        DETACH DELETE n
        """,
        database_=NEO4J_DATABASE,
    )


def create_constraints(driver) -> None:
    for label in sorted(ALLOWED_LABELS):
        safe_label = validate_identifier(label, ALLOWED_LABELS)
        driver.execute_query(
            f"CREATE CONSTRAINT demo_{safe_label}_name IF NOT EXISTS "
            f"FOR (n:{safe_label}) REQUIRE n.name IS UNIQUE",
            database_=NEO4J_DATABASE,
        )


def write_nodes(driver) -> None:
    for node in NODES:
        safe_label = validate_identifier(node.label, ALLOWED_LABELS)
        driver.execute_query(
            f"""
            MERGE (n:{safe_label} {{name: $name}})
            SET n.description = $description,
                n.demo = true
            """,
            name=node.name,
            description=node.description,
            database_=NEO4J_DATABASE,
        )


def write_relationships(driver) -> None:
    for rel in RELATIONSHIPS:
        safe_rel_type = validate_identifier(rel.rel_type, ALLOWED_REL_TYPES)
        driver.execute_query(
            f"""
            MATCH (source {{name: $source}})
            MATCH (target {{name: $target}})
            MERGE (source)-[r:{safe_rel_type}]->(target)
            SET r.evidence = $evidence,
                r.demo = true
            """,
            source=rel.source,
            target=rel.target,
            evidence=rel.evidence,
            database_=NEO4J_DATABASE,
        )


def print_summary(driver) -> None:
    node_records, _, _ = driver.execute_query(
        """
        MATCH (n)
        WHERE n.demo = true
        RETURN labels(n)[0] AS label, count(*) AS count
        ORDER BY label
        """,
        database_=NEO4J_DATABASE,
    )
    rel_records, _, _ = driver.execute_query(
        """
        MATCH ()-[r]->()
        WHERE r.demo = true
        RETURN type(r) AS relationship, count(*) AS count
        ORDER BY relationship
        """,
        database_=NEO4J_DATABASE,
    )

    print("Demo knowledge graph created.")
    print("\nNodes:")
    for record in node_records:
        print(f"- {record['label']}: {record['count']}")

    print("\nRelationships:")
    for record in rel_records:
        print(f"- {record['relationship']}: {record['count']}")

    print("\nOpen Neo4j Browser and run:")
    print("MATCH p=(n)-[r]->(m) WHERE n.demo = true RETURN p")


def main() -> None:
    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    ) as driver:
        driver.verify_connectivity()
        create_constraints(driver)
        reset_demo_graph(driver)
        write_nodes(driver)
        write_relationships(driver)
        print_summary(driver)


if __name__ == "__main__":
    main()
