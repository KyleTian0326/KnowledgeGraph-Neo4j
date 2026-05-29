import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


DISPLAY_METRICS = [
    "page_recall@5",
    "citation_accuracy@5",
    "entity_recall@5",
    "triple_recall@5",
    "relation_precision@5",
    "noise_edge_rate@5",
    "judge_answer_supported_rate",
    "judge_citation_accuracy",
    "judge_missing_evidence_rate",
    "judge_hallucination_rate",
]

METRIC_LABELS = {
    "page_recall@5": "正确页码召回",
    "citation_accuracy@5": "引用页准确率",
    "entity_recall@5": "实体召回",
    "triple_recall@5": "三元组召回",
    "relation_precision@5": "关系准确率",
    "noise_edge_rate@5": "噪声边比例",
    "judge_answer_supported_rate": "答案被证据支撑",
    "judge_citation_accuracy": "裁判引用准确",
    "judge_missing_evidence_rate": "缺证据率",
    "judge_hallucination_rate": "幻觉率",
}

LOWER_IS_BETTER = {
    "noise_edge_rate@5",
    "noise_edge_rate@10",
    "judge_missing_evidence_rate",
    "judge_hallucination_rate",
    "missing_evidence_rate",
    "hallucination_rate",
}


def run_mode(
    python_exe: str,
    dataset: str,
    mode: str,
    ks: str,
    graph_ks: str,
    min_rel_confidence: float,
    judge_answer: bool,
    run_answer: bool,
    output_dir: Path,
) -> dict:
    output_path = output_dir / f"retrieval_eval_{mode}.json"
    command = [
        python_exe,
        "scripts/evaluate_graphrag_retrieval.py",
        "--dataset",
        dataset,
        "--ks",
        ks,
        "--graph-ks",
        graph_ks,
        "--mode",
        mode,
        "--min-rel-confidence",
        str(min_rel_confidence),
        "--output",
        str(output_path),
    ]
    if judge_answer:
        command.append("--judge-answer")
    elif run_answer:
        command.append("--run-answer")
    subprocess.run(command, check=True)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    return {"mode": mode, "output": str(output_path), "summary": report.get("summary", {})}


def flatten_summary(summary: dict) -> dict:
    flattened = {}
    for key, value in summary.items():
        if isinstance(value, dict) and "mean" in value:
            flattened[key] = value["mean"]
    return flattened


def metric_direction(metric: str) -> str:
    return "lower" if metric in LOWER_IS_BETTER else "higher"


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def delta_value(base: Any, current: Any) -> float | None:
    if base is None or current is None:
        return None
    try:
        return round(float(current) - float(base), 6)
    except (TypeError, ValueError):
        return None


def verdict_for_delta(metric: str, delta: float | None, tolerance: float = 1e-9) -> str:
    if delta is None:
        return "无法比较"
    if abs(delta) <= tolerance:
        return "持平"
    improved = delta < 0 if metric_direction(metric) == "lower" else delta > 0
    return "提升" if improved else "下降"


def pick_display_metrics(compact: dict[str, dict], requested: str | None) -> list[str]:
    if requested:
        metrics = [metric.strip() for metric in requested.split(",") if metric.strip()]
    else:
        metrics = DISPLAY_METRICS
    available = {metric for values in compact.values() for metric in values}
    selected = [metric for metric in metrics if metric in available]
    if selected:
        return selected
    return sorted(available)


def build_metric_rows(compact: dict[str, dict], metrics: list[str]) -> list[dict[str, Any]]:
    rows = []
    for metric in metrics:
        vector = compact.get("vector", {}).get(metric)
        graph = compact.get("graph", {}).get(metric)
        hybrid = compact.get("hybrid", {}).get(metric)
        delta_vector = delta_value(vector, hybrid)
        delta_graph = delta_value(graph, hybrid)
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "direction": metric_direction(metric),
                "vector": vector,
                "graph": graph,
                "hybrid": hybrid,
                "hybrid_minus_vector": delta_vector,
                "hybrid_minus_graph": delta_graph,
                "vs_vector": verdict_for_delta(metric, delta_vector),
                "vs_graph": verdict_for_delta(metric, delta_graph),
            }
        )
    return rows


def score_delta(metric: str, base: Any, current: Any) -> float | None:
    delta = delta_value(base, current)
    if delta is None:
        return None
    return -delta if metric_direction(metric) == "lower" else delta


def build_decision(compact: dict[str, dict]) -> dict[str, Any]:
    vector = compact.get("vector", {})
    hybrid = compact.get("hybrid", {})
    if not vector or not hybrid:
        return {
            "status": "unknown",
            "summary": "缺少 vector 或 hybrid 结果，无法判断 GraphRAG 是否带来增益。",
            "signals": {},
        }

    primary_metrics = [
        "judge_answer_supported_rate",
        "judge_citation_accuracy",
        "citation_accuracy@5",
        "triple_recall@5",
        "relation_precision@5",
    ]
    risk_metrics = ["judge_hallucination_rate", "judge_missing_evidence_rate", "noise_edge_rate@5"]
    gains = {
        metric: score_delta(metric, vector.get(metric), hybrid.get(metric))
        for metric in primary_metrics
        if vector.get(metric) is not None and hybrid.get(metric) is not None
    }
    risks = {
        metric: score_delta(metric, vector.get(metric), hybrid.get(metric))
        for metric in risk_metrics
        if vector.get(metric) is not None and hybrid.get(metric) is not None
    }
    positive = sum(1 for value in gains.values() if value is not None and value > 1e-9)
    negative = sum(1 for value in gains.values() if value is not None and value < -1e-9)
    risk_worse = sum(1 for value in risks.values() if value is not None and value < -1e-9)

    if positive > 0 and negative == 0 and risk_worse == 0:
        status = "helpful"
        summary = "hybrid 相比 vector 有正向增益，且没有观察到幻觉/缺证据风险上升。"
    elif positive > 0 and risk_worse == 0:
        status = "mixed_helpful"
        summary = "hybrid 有部分指标提升，但也有部分回答或引用指标下降，需要看题目明细。"
    elif positive == 0 and risk_worse == 0:
        status = "neutral"
        summary = "hybrid 没有明显优于 vector，但也没有观察到额外风险。"
    else:
        status = "risky"
        summary = "hybrid 出现风险指标变差，当前图谱可能给回答引入噪声。"

    return {
        "status": status,
        "summary": summary,
        "signals": {
            "gains_vs_vector": gains,
            "risks_vs_vector": risks,
            "positive_gain_count": positive,
            "negative_gain_count": negative,
            "risk_worse_count": risk_worse,
        },
    }


def markdown_table(rows: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    lines = [
        "# GraphRAG 三模式对照结果",
        "",
        f"结论：{decision.get('summary', '')}",
        "",
        "| 指标 | 趋势 | vector-only | graph-only | hybrid | hybrid 对 vector | hybrid 对 graph |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        direction = "越低越好" if row["direction"] == "lower" else "越高越好"
        lines.append(
            "| {label}<br>`{metric}` | {direction} | {vector} | {graph} | {hybrid} | {vs_vector} ({delta_vector}) | {vs_graph} ({delta_graph}) |".format(
                label=row["label"],
                metric=row["metric"],
                direction=direction,
                vector=format_value(row["vector"]),
                graph=format_value(row["graph"]),
                hybrid=format_value(row["hybrid"]),
                vs_vector=row["vs_vector"],
                delta_vector=format_value(row["hybrid_minus_vector"]),
                vs_graph=row["vs_graph"],
                delta_graph=format_value(row["hybrid_minus_graph"]),
            )
        )
    lines.extend(
        [
            "",
            "判断规则：hybrid 如果在答案支撑、引用准确、三元组召回等指标上高于 vector-only，且幻觉率、缺证据率、噪声边比例没有上升，就说明知识图谱对最终回答有正向帮助。",
            "",
        ]
    )
    return "\n".join(lines)


def print_table(rows: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    print("\nMode comparison table:")
    print("metric                         vector    graph     hybrid    vs vector")
    print("-" * 78)
    for row in rows:
        metric = row["metric"][:28]
        print(
            f"{metric:<30}"
            f"{format_value(row['vector']):>9}"
            f"{format_value(row['graph']):>9}"
            f"{format_value(row['hybrid']):>10}"
            f"    {row['vs_vector']}"
        )
    print(f"\nDecision: {decision.get('summary', '')}")


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "label",
                "direction",
                "vector",
                "graph",
                "hybrid",
                "hybrid_minus_vector",
                "hybrid_minus_graph",
                "vs_vector",
                "vs_graph",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vector-only, graph-only, and hybrid GraphRAG evaluation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ks", default="3,5,10")
    parser.add_argument("--graph-ks", default="5,10")
    parser.add_argument("--modes", default="vector,graph,hybrid")
    parser.add_argument("--min-rel-confidence", type=float, default=0.70)
    parser.add_argument("--run-answer", action="store_true")
    parser.add_argument("--judge-answer", action="store_true")
    parser.add_argument("--python", default=r".\.venv\Scripts\python.exe")
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None, help="Optional Markdown comparison table path. Defaults to output stem with .md.")
    parser.add_argument("--csv", default=None, help="Optional CSV comparison table path. Defaults to output stem with .csv.")
    parser.add_argument("--display-metrics", default=None, help="Comma-separated metrics to show in the comparison table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path("output") / f"graphrag_mode_compare_{run_id}.json"
    output_dir = output_path.parent / f"{output_path.stem}_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    results = [
        run_mode(
            python_exe=args.python,
            dataset=args.dataset,
            mode=mode,
            ks=args.ks,
            graph_ks=args.graph_ks,
            min_rel_confidence=args.min_rel_confidence,
            judge_answer=args.judge_answer,
            run_answer=args.run_answer,
            output_dir=output_dir,
        )
        for mode in modes
    ]
    compact = {result["mode"]: flatten_summary(result["summary"]) for result in results}
    display_metrics = pick_display_metrics(compact, args.display_metrics)
    metric_rows = build_metric_rows(compact, display_metrics)
    decision = build_decision(compact)
    report = {
        "dataset": args.dataset,
        "ks": args.ks,
        "graph_ks": args.graph_ks,
        "min_rel_confidence": args.min_rel_confidence,
        "results": results,
        "compact": compact,
        "comparison": metric_rows,
        "decision": decision,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown) if args.markdown else output_path.with_suffix(".md")
    markdown_path.write_text(markdown_table(metric_rows, decision), encoding="utf-8")
    csv_path = Path(args.csv) if args.csv else output_path.with_suffix(".csv")
    write_csv(metric_rows, csv_path)
    print("GraphRAG mode comparison finished.")
    print_table(metric_rows, decision)
    print(f"Report: {output_path}")
    print(f"Markdown: {markdown_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
