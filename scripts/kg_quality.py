import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any


MIN_RELATION_CONFIDENCE = float(os.getenv("KG_MIN_REL_CONFIDENCE", "0.70"))
MAX_ENTITY_NAME_LENGTH = int(os.getenv("KG_MAX_ENTITY_NAME_LENGTH", "60"))
MAX_EVIDENCE_LENGTH = int(os.getenv("KG_MAX_EVIDENCE_LENGTH", "180"))


GENERIC_ENTITY_NAMES = {
    "公司",
    "企业",
    "装置",
    "设备",
    "系统",
    "工艺",
    "流程",
    "产品",
    "物料",
    "参数",
    "风险",
    "标准",
    "要求",
    "问题",
    "措施",
    "原因",
    "现象",
    "unit",
    "equipment",
    "process",
    "parameter",
    "risk",
    "standard",
}


LABEL_ALIASES = {
    "corporation": "Company",
    "enterprise": "Company",
    "factory": "Plant",
    "installation": "Plant",
    "facility": "Plant",
    "device": "Equipment",
    "instrument": "Equipment",
    "machine": "Equipment",
    "feed": "Material",
    "raw_material": "Material",
    "intermediate": "Material",
    "finished_product": "Product",
    "technology": "Process",
    "procedure": "Procedure",
    "step": "Operation",
    "control_action": "ControlAction",
    "operating_condition": "Condition",
    "fault": "Fault",
    "hazard": "Risk",
    "safety_measure": "Measure",
    "specification": "Standard",
    "company": "Company",
    "plant": "Plant",
    "unit": "Unit",
    "equipment": "Equipment",
    "material": "Material",
    "product": "Product",
    "process": "Process",
    "parameter": "Parameter",
    "risk": "Risk",
    "standard": "Standard",
    "operation": "Operation",
    "condition": "Condition",
    "measure": "Measure",
    "cause": "Cause",
    "phenomenon": "Phenomenon",
    "document": "Document",
    "企业": "Company",
    "公司": "Company",
    "工厂": "Plant",
    "厂区": "Plant",
    "装置": "Unit",
    "单元": "Unit",
    "设备": "Equipment",
    "仪表": "Equipment",
    "物料": "Material",
    "原料": "Material",
    "产品": "Product",
    "工艺": "Process",
    "流程": "Process",
    "参数": "Parameter",
    "指标": "Parameter",
    "风险": "Risk",
    "危险": "Risk",
    "标准": "Standard",
    "规范": "Standard",
    "操作": "Operation",
    "步骤": "Procedure",
    "条件": "Condition",
    "措施": "Measure",
    "原因": "Cause",
    "现象": "Phenomenon",
    "文档": "Document",
}


RELATION_ALIASES = {
    "own": "OWNS",
    "owns": "OWNS",
    "has_unit": "HAS_UNIT",
    "contains_unit": "HAS_UNIT",
    "uses_material": "USES_MATERIAL",
    "uses_feed": "USES_MATERIAL",
    "has_component": "HAS_COMPONENT",
    "contains_component": "HAS_COMPONENT",
    "composed_of": "HAS_COMPONENT",
    "contains": "HAS_COMPONENT",
    "uses_process": "USES_PROCESS",
    "produces": "PRODUCES",
    "has_equipment": "HAS_EQUIPMENT",
    "contains_equipment": "HAS_EQUIPMENT",
    "controls": "CONTROLS",
    "controls_parameter": "CONTROLS",
    "used_for": "USED_FOR",
    "has_risk": "HAS_RISK",
    "complies_with": "COMPLIES_WITH",
    "has_parameter": "HAS_PARAMETER",
    "has_quality_index": "HAS_QUALITY_INDEX",
    "has_index": "HAS_QUALITY_INDEX",
    "has_range": "HAS_RANGE",
    "range": "HAS_RANGE",
    "operates_at": "OPERATES_AT",
    "has_condition": "HAS_CONDITION",
    "requires": "REQUIRES",
    "has_step": "HAS_STEP",
    "precedes": "PRECEDES",
    "causes": "CAUSES",
    "affects": "AFFECTS",
    "mitigated_by": "MITIGATED_BY",
    "monitors": "MONITORS",
    "located_in": "LOCATED_IN",
    "refers_to": "REFERS_TO",
    "属于": "PART_OF",
    "包含": "HAS_UNIT",
    "使用物料": "USES_MATERIAL",
    "含有": "HAS_COMPONENT",
    "组成为": "HAS_COMPONENT",
    "组成": "HAS_COMPONENT",
    "主要组分": "HAS_COMPONENT",
    "采用工艺": "USES_PROCESS",
    "生产": "PRODUCES",
    "具有设备": "HAS_EQUIPMENT",
    "控制": "CONTROLS",
    "用于": "USED_FOR",
    "存在风险": "HAS_RISK",
    "符合": "COMPLIES_WITH",
    "具有参数": "HAS_PARAMETER",
    "质量指标": "HAS_QUALITY_INDEX",
    "指标": "HAS_QUALITY_INDEX",
    "范围": "HAS_RANGE",
    "运行于": "OPERATES_AT",
    "需要": "REQUIRES",
    "导致": "CAUSES",
    "影响": "AFFECTS",
    "通过措施缓解": "MITIGATED_BY",
    "监测": "MONITORS",
    "位于": "LOCATED_IN",
}


RELATION_SCHEMA: dict[str, dict[str, set[str]]] = {
    "OWNS": {"source": {"Company"}, "target": {"Plant", "Unit"}},
    "HAS_UNIT": {"source": {"Company", "Plant", "Unit"}, "target": {"Unit", "Process"}},
    "USES_MATERIAL": {"source": {"Plant", "Unit", "Process", "Operation"}, "target": {"Material"}},
    "HAS_COMPONENT": {"source": {"Material", "Product"}, "target": {"Material", "Product"}},
    "USES_PROCESS": {"source": {"Plant", "Unit"}, "target": {"Process"}},
    "PRODUCES": {"source": {"Plant", "Unit", "Process"}, "target": {"Product", "Material"}},
    "HAS_EQUIPMENT": {"source": {"Plant", "Unit", "Process"}, "target": {"Equipment"}},
    "CONTROLS": {"source": {"Equipment", "Unit", "Process", "Operation", "ControlAction"}, "target": {"Parameter", "Condition"}},
    "USED_FOR": {"source": {"Equipment", "Material", "Product", "Process", "Operation"}, "target": {"Process", "Product", "Unit", "Purpose"}},
    "HAS_RISK": {"source": {"Plant", "Unit", "Equipment", "Process", "Operation", "Material"}, "target": {"Risk", "Fault"}},
    "COMPLIES_WITH": {"source": {"Company", "Plant", "Unit", "Process", "Operation", "Equipment"}, "target": {"Standard"}},
    "HAS_PARAMETER": {"source": {"Plant", "Unit", "Equipment", "Process", "Operation"}, "target": {"Parameter"}},
    "HAS_QUALITY_INDEX": {"source": {"Material", "Product"}, "target": {"Parameter"}},
    "HAS_RANGE": {"source": {"Parameter", "Condition"}, "target": {"Condition", "Parameter"}},
    "OPERATES_AT": {"source": {"Plant", "Unit", "Equipment", "Process"}, "target": {"Parameter", "Condition"}},
    "HAS_CONDITION": {"source": {"Process", "Operation", "Unit", "Equipment"}, "target": {"Condition"}},
    "REQUIRES": {"source": {"Process", "Operation", "Procedure", "Equipment"}, "target": {"Material", "Parameter", "Condition", "Equipment", "Measure"}},
    "HAS_STEP": {"source": {"Procedure", "Process", "Operation"}, "target": {"Operation", "Procedure"}},
    "PRECEDES": {"source": {"Operation", "Procedure", "Process"}, "target": {"Operation", "Procedure", "Process"}},
    "CAUSES": {"source": {"Cause", "Fault", "Condition", "Parameter", "Operation", "Process"}, "target": {"Fault", "Risk", "Phenomenon", "Condition"}},
    "AFFECTS": {"source": {"Parameter", "Condition", "Material", "Process", "Operation"}, "target": {"Process", "Product", "Equipment", "Parameter", "Condition", "Material", "Risk", "Phenomenon"}},
    "MITIGATED_BY": {"source": {"Risk", "Fault", "Phenomenon"}, "target": {"Measure", "Procedure", "Operation", "ControlAction"}},
    "MONITORS": {"source": {"Equipment", "Instrument", "ControlAction", "Operation"}, "target": {"Parameter", "Condition", "Phenomenon"}},
    "LOCATED_IN": {"source": {"Equipment", "Unit", "Process"}, "target": {"Unit", "Plant"}},
    "PART_OF": {"source": {"Unit", "Equipment", "Process", "Operation"}, "target": {"Unit", "Plant", "Process"}},
    "REFERS_TO": {"source": {"Document", "Standard", "Procedure"}, "target": {"Plant", "Unit", "Equipment", "Process", "Parameter", "Risk"}},
}


def normalize_space(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def clean_name(value: Any) -> str:
    text = normalize_space(str(value or ""))
    text = re.sub(r"^[\s:：,，.;；\-—_()（）\[\]【】\"'“”]+", "", text)
    text = re.sub(r"[\s:：,，.;；\-—_()（）\[\]【】\"'“”]+$", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(的)?(主要|最重要的|重要的)?(组分|成分|质量|控制|工艺|操作)$", "", text)
    return text


def clean_evidence(value: Any) -> str:
    text = normalize_space(str(value or ""))
    text = text.strip("` \t\r\n")
    if len(text) > MAX_EVIDENCE_LENGTH:
        text = text[:MAX_EVIDENCE_LENGTH].rstrip() + "..."
    return text


def normalized_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = re.sub(r"[\s\-_./\\:：,，.;；()（）\[\]【】{}<>《》\"'“”‘’]+", "", value)
    return value


def canonical_key(name: Any) -> str:
    return normalized_key(clean_name(name))


def is_bad_entity_name(name: str) -> bool:
    key = normalized_key(name)
    if not key:
        return True
    if name in GENERIC_ENTITY_NAMES or key in {normalized_key(item) for item in GENERIC_ENTITY_NAMES}:
        return True
    if len(name) > MAX_ENTITY_NAME_LENGTH:
        return True
    if len(key) < 2:
        return True
    if re.fullmatch(r"[\d.]+", key):
        return True
    return False


def map_label(label: Any, allowed_labels: set[str]) -> str | None:
    raw = clean_name(label)
    if raw in allowed_labels:
        return raw
    alias_key = normalized_key(raw).replace(" ", "_")
    mapped = LABEL_ALIASES.get(alias_key) or LABEL_ALIASES.get(raw.lower()) or LABEL_ALIASES.get(raw)
    if mapped in allowed_labels:
        return mapped
    return None


def map_relation_type(value: Any, allowed_rel_types: set[str]) -> str | None:
    raw = normalize_space(str(value or ""))
    upper = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
    if upper in allowed_rel_types:
        return upper
    alias_key = normalized_key(raw).replace(" ", "_")
    mapped = RELATION_ALIASES.get(alias_key) or RELATION_ALIASES.get(raw.lower()) or RELATION_ALIASES.get(raw)
    if mapped in allowed_rel_types:
        return mapped
    return None


def text_contains_name(text: str, name: str) -> bool:
    key = canonical_key(name)
    if not key:
        return False
    return key in normalized_key(text)


def evidence_is_in_text(evidence: str, chunk_text: str) -> bool:
    evidence_key = normalized_key(evidence)
    if not evidence_key:
        return False
    chunk_key = normalized_key(chunk_text)
    if evidence_key in chunk_key:
        return True

    if len(evidence_key) < 12:
        return False

    segments = [
        segment
        for segment in re.split(r"[\n。！？!?；;]", chunk_text)
        if len(normalized_key(segment)) >= 8
    ]
    for segment in segments:
        ratio = SequenceMatcher(None, evidence_key, normalized_key(segment)).ratio()
        if ratio >= 0.84:
            return True
    return False


def schema_labels_for(rel_type: str, role: str) -> set[str]:
    schema = RELATION_SCHEMA.get(rel_type, {})
    return schema.get(role, set())


def infer_label(rel_type: str, role: str, allowed_labels: set[str]) -> str:
    candidates = sorted(schema_labels_for(rel_type, role) & allowed_labels)
    return candidates[0] if candidates else "Unit"


def relationship_schema_penalty(rel_type: str, source_label: str, target_label: str) -> tuple[float, list[str]]:
    flags: list[str] = []
    schema = RELATION_SCHEMA.get(rel_type)
    if not schema:
        return 0.0, flags
    source_allowed = schema.get("source", set())
    target_allowed = schema.get("target", set())
    penalty = 0.0
    if source_allowed and source_label not in source_allowed:
        penalty += 0.14
        flags.append("source_schema_mismatch")
    if target_allowed and target_label not in target_allowed:
        penalty += 0.14
        flags.append("target_schema_mismatch")
    return penalty, flags


def score_relationship(
    source_name: str,
    target_name: str,
    rel_type: str,
    evidence: str,
    chunk_text: str,
    source_label: str,
    target_label: str,
) -> tuple[float, list[str]]:
    flags: list[str] = []
    score = 0.20

    if evidence_is_in_text(evidence, chunk_text):
        score += 0.35
    else:
        flags.append("evidence_not_found")

    source_in_evidence = text_contains_name(evidence, source_name)
    target_in_evidence = text_contains_name(evidence, target_name)
    source_in_chunk = text_contains_name(chunk_text, source_name)
    target_in_chunk = text_contains_name(chunk_text, target_name)

    if source_in_evidence:
        score += 0.15
    elif source_in_chunk:
        score += 0.05
    else:
        flags.append("source_not_found")

    if target_in_evidence:
        score += 0.15
    elif target_in_chunk:
        score += 0.05
    else:
        flags.append("target_not_found")

    if rel_type in RELATION_SCHEMA:
        score += 0.10

    if 8 <= len(evidence) <= MAX_EVIDENCE_LENGTH:
        score += 0.05

    penalty, schema_flags = relationship_schema_penalty(rel_type, source_label, target_label)
    score -= penalty
    flags.extend(schema_flags)
    return max(0.0, min(score, 1.0)), flags


def _merge_node(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    if not existing.get("description") and incoming.get("description"):
        existing["description"] = incoming["description"]
    aliases = set(existing.get("aliases") or [])
    aliases.update(incoming.get("aliases") or [])
    existing["aliases"] = sorted(alias for alias in aliases if alias and alias != existing.get("name"))
    existing["support_count"] = int(existing.get("support_count") or 1) + int(incoming.get("support_count") or 1)
    return existing


def validate_extraction(
    extraction: dict[str, Any],
    chunk_text: str,
    source: str,
    chunk_index: int,
    allowed_labels: set[str],
    allowed_rel_types: set[str],
) -> dict[str, Any]:
    raw_nodes = extraction.get("nodes") if isinstance(extraction.get("nodes"), list) else []
    raw_relationships = (
        extraction.get("relationships") if isinstance(extraction.get("relationships"), list) else []
    )
    dropped: list[dict[str, str]] = []
    nodes_by_key: dict[str, dict[str, Any]] = {}

    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        name = clean_name(item.get("name"))
        label = map_label(item.get("label"), allowed_labels)
        key = canonical_key(name)
        if not label:
            dropped.append({"kind": "node", "name": name, "reason": "unknown_label"})
            continue
        if is_bad_entity_name(name):
            dropped.append({"kind": "node", "name": name, "reason": "bad_name"})
            continue
        if not text_contains_name(chunk_text, name):
            dropped.append({"kind": "node", "name": name, "reason": "name_not_found_in_chunk"})
            continue
        node = {
            "name": name,
            "label": label,
            "canonical_key": key,
            "description": normalize_space(str(item.get("description") or ""))[:240],
            "aliases": [clean_name(alias) for alias in item.get("aliases", []) if clean_name(alias)]
            if isinstance(item.get("aliases"), list)
            else [],
            "support_count": 1,
        }
        if key in nodes_by_key:
            nodes_by_key[key] = _merge_node(nodes_by_key[key], node)
        else:
            nodes_by_key[key] = node

    relationships_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    for item in raw_relationships:
        if not isinstance(item, dict):
            continue
        source_name = clean_name(item.get("source"))
        target_name = clean_name(item.get("target"))
        rel_type = map_relation_type(item.get("type"), allowed_rel_types)
        evidence = clean_evidence(item.get("evidence"))
        source_key = canonical_key(source_name)
        target_key = canonical_key(target_name)

        if not rel_type:
            dropped.append({"kind": "relationship", "name": f"{source_name}->{target_name}", "reason": "unknown_type"})
            continue
        if is_bad_entity_name(source_name) or is_bad_entity_name(target_name):
            dropped.append({"kind": "relationship", "name": f"{source_name}->{target_name}", "reason": "bad_endpoint"})
            continue
        if not text_contains_name(chunk_text, source_name) or not text_contains_name(chunk_text, target_name):
            dropped.append(
                {
                    "kind": "relationship",
                    "name": f"{source_name}->{target_name}",
                    "reason": "endpoint_not_found_in_chunk",
                }
            )
            continue
        if source_key == target_key:
            dropped.append({"kind": "relationship", "name": source_name, "reason": "self_loop"})
            continue
        if not evidence:
            dropped.append({"kind": "relationship", "name": f"{source_name}->{target_name}", "reason": "missing_evidence"})
            continue
        if not evidence_is_in_text(evidence, chunk_text):
            dropped.append({"kind": "relationship", "name": f"{source_name}->{target_name}", "reason": "evidence_not_found"})
            continue

        source_node = nodes_by_key.get(source_key)
        target_node = nodes_by_key.get(target_key)
        source_label = source_node["label"] if source_node else infer_label(rel_type, "source", allowed_labels)
        target_label = target_node["label"] if target_node else infer_label(rel_type, "target", allowed_labels)
        confidence, flags = score_relationship(
            source_name,
            target_name,
            rel_type,
            evidence,
            chunk_text,
            source_label,
            target_label,
        )
        if "source_schema_mismatch" in flags and "target_schema_mismatch" in flags:
            dropped.append(
                {
                    "kind": "relationship",
                    "name": f"{source_name}-{rel_type}->{target_name}",
                    "reason": "double_schema_mismatch",
                }
            )
            continue
        if confidence < MIN_RELATION_CONFIDENCE:
            dropped.append(
                {
                    "kind": "relationship",
                    "name": f"{source_name}-{rel_type}->{target_name}",
                    "reason": f"low_confidence:{confidence:.2f}",
                }
            )
            continue

        if source_key not in nodes_by_key:
            nodes_by_key[source_key] = {
                "name": source_name,
                "label": source_label,
                "canonical_key": source_key,
                "description": "",
                "aliases": [],
                "support_count": 1,
            }
        if target_key not in nodes_by_key:
            nodes_by_key[target_key] = {
                "name": target_name,
                "label": target_label,
                "canonical_key": target_key,
                "description": "",
                "aliases": [],
                "support_count": 1,
            }

        rel_key = (source_key, rel_type, target_key)
        relationship = {
            "source": nodes_by_key[source_key]["name"],
            "source_key": source_key,
            "source_label": nodes_by_key[source_key]["label"],
            "type": rel_type,
            "target": nodes_by_key[target_key]["name"],
            "target_key": target_key,
            "target_label": nodes_by_key[target_key]["label"],
            "evidence": evidence,
            "confidence": round(confidence, 3),
            "quality_flags": flags,
            "support_count": 1,
        }
        if rel_key in relationships_by_key:
            existing = relationships_by_key[rel_key]
            existing["support_count"] = int(existing.get("support_count") or 1) + 1
            if relationship["confidence"] > existing.get("confidence", 0):
                existing.update(relationship)
        else:
            relationships_by_key[rel_key] = relationship

    for relationship in rule_based_relationships(chunk_text, nodes_by_key, allowed_labels):
        rel_key = (relationship["source_key"], relationship["type"], relationship["target_key"])
        if rel_key in relationships_by_key:
            existing = relationships_by_key[rel_key]
            existing["support_count"] = int(existing.get("support_count") or 1) + 1
            continue
        relationships_by_key[rel_key] = relationship

    nodes = sorted(nodes_by_key.values(), key=lambda item: (item["label"], item["name"]))
    relationships = list(relationships_by_key.values())
    return {
        "nodes": nodes,
        "relationships": relationships,
        "_quality": {
            "source": source,
            "chunk": chunk_index,
            "input_nodes": len(raw_nodes),
            "accepted_nodes": len(nodes),
            "input_relationships": len(raw_relationships),
            "accepted_relationships": len(relationships),
            "dropped_count": len(dropped),
            "dropped": dropped[:30],
            "min_relation_confidence": MIN_RELATION_CONFIDENCE,
        },
    }


def ensure_node(
    nodes_by_key: dict[str, dict[str, Any]],
    name: str,
    label: str,
    allowed_labels: set[str],
) -> dict[str, Any] | None:
    name = clean_name(name)
    if label not in allowed_labels or is_bad_entity_name(name):
        return None
    key = canonical_key(name)
    node = nodes_by_key.get(key)
    if node:
        return node
    node = {
        "name": name,
        "label": label,
        "canonical_key": key,
        "description": "",
        "aliases": [],
        "support_count": 1,
    }
    nodes_by_key[key] = node
    return node


def make_rule_relationship(
    nodes_by_key: dict[str, dict[str, Any]],
    source: str,
    rel_type: str,
    target: str,
    evidence: str,
    source_label: str,
    target_label: str,
    allowed_labels: set[str],
) -> dict[str, Any] | None:
    source_node = ensure_node(nodes_by_key, source, source_label, allowed_labels)
    target_node = ensure_node(nodes_by_key, target, target_label, allowed_labels)
    if not source_node or not target_node:
        return None
    if source_node["canonical_key"] == target_node["canonical_key"]:
        return None
    return {
        "source": source_node["name"],
        "source_key": source_node["canonical_key"],
        "source_label": source_node["label"],
        "type": rel_type,
        "target": target_node["name"],
        "target_key": target_node["canonical_key"],
        "target_label": target_node["label"],
        "evidence": clean_evidence(evidence),
        "confidence": 0.92,
        "quality_flags": ["rule_based"],
        "support_count": 1,
    }


def rule_based_relationships(
    chunk_text: str,
    nodes_by_key: dict[str, dict[str, Any]],
    allowed_labels: set[str],
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    sentences = [sentence.strip() for sentence in re.split(r"[\n。！？!?；;]", chunk_text) if sentence.strip()]

    for sentence in sentences:
        if len(sentence) > 220:
            continue

        component_match = re.search(
            r"(?P<source>[\u4e00-\u9fffA-Za-z0-9（）()]{2,30}?)(?:的)?(?:主要组分|组分|主要成分|成分)是(?P<targets>[^。；;\n]{2,120})",
            sentence,
        )
        if component_match:
            source = component_match.group("source")
            for target in re.split(r"[、,，和及与]", component_match.group("targets")):
                target = re.sub(r"(还含有|其中|少量|大量|约占|占).*", "", target).strip()
                target = re.sub(r"(以下组分|以上组分|组分|成分)$", "", target).strip()
                if target:
                    rel = make_rule_relationship(
                        nodes_by_key,
                        source,
                        "HAS_COMPONENT",
                        target,
                        sentence,
                        "Product",
                        "Material",
                        allowed_labels,
                    )
                    if rel:
                        relationships.append(rel)

        quality_match = re.search(
            r"(?P<source>[\u4e00-\u9fffA-Za-z0-9（）()]{2,30}?)(?:最重要的)?(?:质量指标|控制指标|指标)是(?P<target>[\u4e00-\u9fffA-Za-z0-9（）()/+]{2,40})",
            sentence,
        )
        if quality_match:
            rel = make_rule_relationship(
                nodes_by_key,
                quality_match.group("source"),
                "HAS_QUALITY_INDEX",
                quality_match.group("target"),
                sentence,
                "Product",
                "Parameter",
                allowed_labels,
            )
            if rel:
                relationships.append(rel)

    return relationships
