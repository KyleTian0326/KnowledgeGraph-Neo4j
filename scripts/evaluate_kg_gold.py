import argparse
import json
import re
from pathlib import Path
from typing import Any

from kg_quality import canonical_key, evidence_is_in_text


def normalize_rel(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_").upper()


def normalize_triple(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, dict):
        source = value.get("source")
        rel_type = value.get("type") or value.get("relation")
        target = value.get("target")
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        source, rel_type, target = value[:3]
    else:
        return None
    if not source or not rel_type or not target:
        return None
    return canonical_key(source), normalize_rel(rel_type), canonical_key(target)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        rows.append(row)
    return rows


def load_gold(path: Path) -> dict[str, set[Any]]:
    rows = load_jsonl(path) if path.suffix.lower() == ".jsonl" else load_json(path)
    if isinstance(rows, dict):
        rows = rows.get("items") or rows.get("gold") or []
    gold_entities: set[str] = set()
    gold_triples: set[tuple[str, str, str]] = set()
    for row in rows:
        for entity in row.get("entities") or row.get("expected_entities") or []:
            key = canonical_key(entity.get("name") if isinstance(entity, dict) else entity)
            if key:
                gold_entities.add(key)
        for triple in row.get("triples") or row.get("expected_triples") or []:
            normalized = normalize_triple(triple)
            if normalized:
                gold_triples.add(normalized)
    return {"entities": gold_entities, "triples": gold_triples}


def iter_extracted_items(path: Path):
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected extraction JSON list: {path}")
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("extraction"), dict):
            yield item, item["extraction"]


def load_predictions(path: Path) -> dict[str, Any]:
    entities: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    relation_evidence = []
    dropped_reasons: dict[str, int] = {}
    for item, extraction in iter_extracted_items(path):
        chunk_text = str(item.get("text") or item.get("chunk_text") or "")
        for node in extraction.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            key = canonical_key(node.get("canonical_key") or node.get("name"))
            if key:
                entities.add(key)
        for rel in extraction.get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            triple = normalize_triple(rel)
            if triple:
                triples.add(triple)
            evidence = str(rel.get("evidence") or "")
            relation_evidence.append(
                {
                    "triple": triple,
                    "has_evidence": bool(evidence),
                    "evidence_in_chunk": evidence_is_in_text(evidence, chunk_text) if chunk_text else None,
                    "confidence": rel.get("confidence"),
                }
            )
        quality = extraction.get("_quality") or {}
        for dropped in quality.get("dropped") or []:
            reason = str(dropped.get("reason") or "unknown")
            dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
    return {
        "entities": entities,
        "triples": triples,
        "relation_evidence": relation_evidence,
        "dropped_reasons": dropped_reasons,
    }


def prf(predicted: set[Any], gold: set[Any]) -> dict[str, Any]:
    tp = predicted & gold
    fp = predicted - gold
    fn = gold - predicted
    precision = len(tp) / len(predicted) if predicted else None
    recall = len(tp) / len(gold) if gold else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
        "f1": round(f1, 6) if f1 is not None else None,
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "false_positive_examples": sorted(map(str, list(fp)))[:20],
        "false_negative_examples": sorted(map(str, list(fn)))[:20],
    }


def evidence_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"count": 0, "evidence_support_rate": None}
    has_evidence = sum(1 for item in items if item["has_evidence"])
    known_locations = [item for item in items if item["evidence_in_chunk"] is not None]
    located = sum(1 for item in known_locations if item["evidence_in_chunk"])
    return {
        "count": len(items),
        "has_evidence_rate": round(has_evidence / len(items), 6),
        "evidence_support_rate": round(located / len(known_locations), 6) if known_locations else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate extracted KG against a human gold file.")
    parser.add_argument("--extraction", required=True, help="Extraction JSON from the KG pipeline.")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL with entities and triples.")
    parser.add_argument("--output", default="output/kg_gold_eval.json", help="JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_gold(Path(args.gold))
    predicted = load_predictions(Path(args.extraction))
    report = {
        "extraction": args.extraction,
        "gold": args.gold,
        "entity": prf(predicted["entities"], gold["entities"]),
        "triple": prf(predicted["triples"], gold["triples"]),
        "evidence": evidence_summary(predicted["relation_evidence"]),
        "dropped_reasons": predicted["dropped_reasons"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("KG gold evaluation finished.")
    print(f"Entity F1: {report['entity']['f1']}")
    print(f"Triple F1: {report['triple']['f1']}")
    print(f"Evidence support rate: {report['evidence']['evidence_support_rate']}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
