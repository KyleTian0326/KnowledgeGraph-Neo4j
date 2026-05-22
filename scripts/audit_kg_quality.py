import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"Expected a JSON list: {path}")


def iter_extractions(items: list[dict[str, Any]]):
    for item in items:
        extraction = item.get("extraction")
        if isinstance(extraction, dict):
            yield item, extraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit KG extraction quality reports.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="output/deepseek_kg_extraction.json",
        help="Extraction JSON produced by ingest_source.py or ingest_source_pipeline.py.",
    )
    parser.add_argument("--examples", type=int, default=12, help="How many dropped examples to show.")
    args = parser.parse_args()

    path = Path(args.json_path)
    items = load_items(path)
    chunks = 0
    nodes = 0
    relationships = 0
    input_nodes = 0
    input_relationships = 0
    dropped = 0
    dropped_reasons: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    low_confidence: list[dict[str, Any]] = []
    dropped_examples: list[dict[str, Any]] = []
    entity_names: defaultdict[str, set[str]] = defaultdict(set)

    for item, extraction in iter_extractions(items):
        chunks += 1
        node_items = extraction.get("nodes") or []
        relationship_items = extraction.get("relationships") or []
        nodes += len(node_items)
        relationships += len(relationship_items)

        for node in node_items:
            if not isinstance(node, dict):
                continue
            labels[str(node.get("label") or "")] += 1
            key = str(node.get("canonical_key") or node.get("name") or "")
            name = str(node.get("name") or "")
            if key and name:
                entity_names[key].add(name)

        for rel in relationship_items:
            if not isinstance(rel, dict):
                continue
            relation_types[str(rel.get("type") or "")] += 1
            confidence = float(rel.get("confidence") or 0)
            if confidence < 0.72:
                low_confidence.append(
                    {
                        "source": item.get("source"),
                        "chunk": item.get("chunk"),
                        "triple": f"{rel.get('source')} -{rel.get('type')}-> {rel.get('target')}",
                        "confidence": confidence,
                        "flags": rel.get("quality_flags") or [],
                    }
                )

        quality = extraction.get("_quality") or {}
        input_nodes += int(quality.get("input_nodes") or 0)
        input_relationships += int(quality.get("input_relationships") or 0)
        dropped_items = quality.get("dropped") or []
        dropped += int(quality.get("dropped_count") or len(dropped_items))
        for drop in dropped_items:
            if not isinstance(drop, dict):
                continue
            reason = str(drop.get("reason") or "unknown")
            dropped_reasons[reason] += 1
            if len(dropped_examples) < args.examples:
                dropped_examples.append(
                    {
                        "source": item.get("source"),
                        "chunk": item.get("chunk"),
                        "kind": drop.get("kind"),
                        "name": drop.get("name"),
                        "reason": reason,
                    }
                )

    alias_groups = {key: sorted(names) for key, names in entity_names.items() if len(names) > 1}
    print(f"File: {path}")
    print(f"Chunks: {chunks}")
    print(f"Accepted nodes: {nodes} / raw {input_nodes}")
    print(f"Accepted relationships: {relationships} / raw {input_relationships}")
    print(f"Dropped items: {dropped}")
    print("\nTop dropped reasons:")
    for reason, count in dropped_reasons.most_common(12):
        print(f"- {reason}: {count}")

    print("\nEntity labels:")
    for label, count in labels.most_common():
        print(f"- {label}: {count}")

    print("\nRelationship types:")
    for rel_type, count in relation_types.most_common():
        print(f"- {rel_type}: {count}")

    if alias_groups:
        print("\nPotential merged aliases:")
        for key, names in list(alias_groups.items())[: args.examples]:
            print(f"- {key}: {', '.join(names)}")

    if low_confidence:
        print("\nLow-confidence accepted relationships:")
        for item in low_confidence[: args.examples]:
            print(f"- {item['confidence']:.2f} {item['triple']} ({item['source']} #{item['chunk']}) flags={item['flags']}")

    if dropped_examples:
        print("\nDropped examples:")
        for item in dropped_examples:
            print(f"- {item['reason']} | {item['kind']} | {item['name']} ({item['source']} #{item['chunk']})")


if __name__ == "__main__":
    main()
