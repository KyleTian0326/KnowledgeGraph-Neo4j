import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


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
    report = {
        "dataset": args.dataset,
        "ks": args.ks,
        "graph_ks": args.graph_ks,
        "min_rel_confidence": args.min_rel_confidence,
        "results": results,
        "compact": compact,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("GraphRAG mode comparison finished.")
    for mode, metrics in compact.items():
        interesting = {
            key: metrics.get(key)
            for key in (
                "page_recall@5",
                "triple_recall@10",
                "relation_precision@10",
                "citation_accuracy@5",
                "judge_answer_supported_rate",
                "judge_hallucination_rate",
            )
            if key in metrics
        }
        print(f"- {mode}: {interesting}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
