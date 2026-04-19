from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUTPUT_DIR = ROOT / "outputs" / "paper_plot_bundle"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_source_json(filename: str, *, include_output_dir: bool = False) -> tuple[Any, str]:
    candidates = [REPORTS / filename, ROOT / filename]
    if include_output_dir:
        candidates.append(OUTPUT_DIR / filename)
    for path in candidates:
        data = _load_json(path)
        if data is not None:
            return data, str(path.relative_to(ROOT))
    return None, ""


def _find_optional_json(filename: str, preferred_dir: Path | None = None) -> tuple[Any, str]:
    candidates: list[Path] = []
    if preferred_dir is not None:
        candidates.append(preferred_dir / filename)
    candidates.extend([REPORTS / filename, ROOT / filename, OUTPUT_DIR / filename])
    seen: set[str] = set()
    for c in candidates:
        k = str(c.resolve())
        if k in seen:
            continue
        seen.add(k)
        data = _load_json(c)
        if data is not None:
            return data, str(c.relative_to(ROOT))
    for path in ROOT.glob(f"outputs/**/{filename}"):
        data = _load_json(path)
        if data is not None:
            return data, str(path.relative_to(ROOT))
    return None, ""


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _collect_enriched_runs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in ROOT.glob("outputs/**/*.json"):
        if path.name not in {"training_result.json", "metrics.json"}:
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        if "training_curve" not in data and "representative_eval_trace" not in data:
            continue
        out.append({"path": str(path.relative_to(ROOT)), "data": data})
    return out


def _collect_outer_loop_round_traces() -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for path in ROOT.glob("outputs/**/outer_loop_round_trace.json"):
        data = _load_json(path)
        if isinstance(data, list):
            traces.append({"path": str(path.relative_to(ROOT)), "rounds": data})
    return traces


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notes_on_missing_sections: list[str] = []

    final_validation, final_validation_path = _resolve_source_json("final_multiseed_validation_20260412.json")
    iterative_repair, iterative_repair_path = _resolve_source_json("iterative_uncertain_repair_20260412.json")
    recognizer_metrics, recognizer_metrics_path = _resolve_source_json("recognizer_metrics_latest.json", include_output_dir=True)

    if not isinstance(final_validation, dict):
        notes_on_missing_sections.append("Missing or unreadable reports/final_multiseed_validation_20260412.json")
        final_validation = {}
        final_validation_path = ""
    if not isinstance(iterative_repair, dict):
        notes_on_missing_sections.append("Missing or unreadable reports/iterative_uncertain_repair_20260412.json")
        iterative_repair = {}
        iterative_repair_path = ""
    if not isinstance(recognizer_metrics, dict):
        notes_on_missing_sections.append("Missing or unreadable reports/recognizer_metrics_latest.json")
        recognizer_metrics = {}
        recognizer_metrics_path = ""

    enriched_runs = _collect_enriched_runs()
    if not enriched_runs:
        notes_on_missing_sections.append("No enriched per-run JSON files with training_curve/representative_eval_trace were found under outputs/.")

    outer_loop_round_traces = _collect_outer_loop_round_traces()
    if not outer_loop_round_traces:
        notes_on_missing_sections.append("No outer_loop_round_trace.json files were found under outputs/.")

    # final benchmark per-run CSV
    final_runs = final_validation.get("runs", []) if isinstance(final_validation.get("runs"), list) else []
    final_per_run_rows: list[dict[str, Any]] = []
    for row in final_runs:
        if not isinstance(row, dict):
            continue
        final_per_run_rows.append(
            {
                "split": row.get("split", ""),
                "mode": row.get("mode", ""),
                "seed": row.get("seed", ""),
                "selection_score": _safe_float(row.get("selection_score", 0.0)),
                "min_recovery_ratio": _safe_float(row.get("min_recovery_ratio", 0.0)),
                "constraint_violation_rate_eval": _safe_float(row.get("constraint_violation_rate_eval", 0.0)),
                "invalid_action_rate_eval": _safe_float(row.get("invalid_action_rate_eval", 0.0)),
                "lipschitz_mean": _safe_float(row.get("lipschitz_mean", 0.0)),
                "wait_hold_usage_eval": _safe_float(row.get("wait_hold_usage_eval", 0.0)),
                "completed": bool(row.get("completed", True)),
                "failed": bool(row.get("failed", False)),
                "artifact_json": row.get("artifact_json", ""),
            }
        )

    _write_csv(
        OUTPUT_DIR / "final_benchmark_per_run.csv",
        final_per_run_rows,
        [
            "split",
            "mode",
            "seed",
            "selection_score",
            "min_recovery_ratio",
            "constraint_violation_rate_eval",
            "invalid_action_rate_eval",
            "lipschitz_mean",
            "wait_hold_usage_eval",
            "completed",
            "failed",
            "artifact_json",
        ],
    )

    # final benchmark summary CSV
    final_summary_rows: list[dict[str, Any]] = []
    stats = final_validation.get("stats", {}) if isinstance(final_validation.get("stats"), dict) else {}
    for split_name, split_stats in stats.items():
        if not isinstance(split_stats, dict):
            continue
        for mode, mode_stats in split_stats.items():
            if not isinstance(mode_stats, dict):
                continue
            final_summary_rows.append(
                {
                    "split": split_name,
                    "mode": mode,
                    "selection_score_mean": _safe_float(mode_stats.get("selection_score_mean", 0.0)),
                    "selection_score_std": _safe_float(mode_stats.get("selection_score_std", 0.0)),
                    "min_recovery_ratio_mean": _safe_float(mode_stats.get("min_recovery_ratio_mean", 0.0)),
                    "constraint_violation_rate_eval_mean": _safe_float(mode_stats.get("constraint_violation_rate_eval_mean", 0.0)),
                    "invalid_action_rate_eval_mean": _safe_float(mode_stats.get("invalid_action_rate_eval_mean", 0.0)),
                    "lipschitz_mean_mean": _safe_float(mode_stats.get("lipschitz_mean_mean", 0.0)),
                    "wait_hold_usage_eval_mean": _safe_float(mode_stats.get("wait_hold_usage_eval_mean", 0.0)),
                }
            )

    _write_csv(
        OUTPUT_DIR / "final_benchmark_summary.csv",
        final_summary_rows,
        [
            "split",
            "mode",
            "selection_score_mean",
            "selection_score_std",
            "min_recovery_ratio_mean",
            "constraint_violation_rate_eval_mean",
            "invalid_action_rate_eval_mean",
            "lipschitz_mean_mean",
            "wait_hold_usage_eval_mean",
        ],
    )

    # iterative repair summary CSV
    iter_rows: list[dict[str, Any]] = []
    for item in iterative_repair.get("iterations", []) if isinstance(iterative_repair.get("iterations"), list) else []:
        if not isinstance(item, dict):
            continue
        uavg = item.get("uncertain_avg", {}) if isinstance(item.get("uncertain_avg"), dict) else {}
        for mode in ("baseline_rl", "single_shot_llm", "full_outer_loop"):
            mode_avg = uavg.get(mode, {}) if isinstance(uavg.get(mode), dict) else None
            if mode_avg is None:
                continue
            iter_rows.append(
                {
                    "iteration": item.get("iteration", ""),
                    "mode": mode,
                    "uncertain_selection_score": _safe_float(mode_avg.get("selection_score", 0.0)),
                    "uncertain_min_recovery_ratio": _safe_float(mode_avg.get("min_recovery_ratio", 0.0)),
                    "uncertain_constraint_violation_rate_eval": _safe_float(mode_avg.get("constraint_violation_rate_eval", 0.0)),
                    "uncertain_invalid_action_rate_eval": _safe_float(mode_avg.get("invalid_action_rate_eval", 0.0)),
                    "uncertain_lipschitz_mean": _safe_float(mode_avg.get("lipschitz_mean", 0.0)),
                    "uncertain_wait_hold_usage_eval": _safe_float(mode_avg.get("wait_hold_usage_eval", 0.0)),
                    "uncertain_run_count": len(item.get("uncertain_runs", [])) if isinstance(item.get("uncertain_runs"), list) else 0,
                    "eval_run_count": len(item.get("eval_runs", [])) if isinstance(item.get("eval_runs"), list) else 0,
                }
            )

    _write_csv(
        OUTPUT_DIR / "iterative_repair_summary.csv",
        iter_rows,
        [
            "iteration",
            "mode",
            "uncertain_selection_score",
            "uncertain_min_recovery_ratio",
            "uncertain_constraint_violation_rate_eval",
            "uncertain_invalid_action_rate_eval",
            "uncertain_lipschitz_mean",
            "uncertain_wait_hold_usage_eval",
            "uncertain_run_count",
            "eval_run_count",
        ],
    )

    # recognition summary CSV
    recog_rows: list[dict[str, Any]] = []
    datasets = recognizer_metrics.get("datasets", {}) if isinstance(recognizer_metrics.get("datasets"), dict) else {}
    for split_name, split_data in datasets.items():
        if not isinstance(split_data, dict):
            continue
        for mode in ("rule", "llm", "hybrid"):
            mode_data = split_data.get(mode, {}) if isinstance(split_data.get(mode), dict) else {}
            recog_rows.append(
                {
                    "split": split_name,
                    "mode": mode,
                    "accuracy": _safe_float(mode_data.get("accuracy", 0.0)),
                    "macro_f1": _safe_float(mode_data.get("macro_f1", 0.0)),
                    "hybrid_used_llm_count": int(mode_data.get("hybrid_used_llm_count", 0)) if mode == "hybrid" else 0,
                    "hybrid_used_llm_ratio": _safe_float(mode_data.get("hybrid_used_llm_ratio", 0.0)) if mode == "hybrid" else 0.0,
                }
            )

    _write_csv(
        OUTPUT_DIR / "recognition_summary.csv",
        recog_rows,
        ["split", "mode", "accuracy", "macro_f1", "hybrid_used_llm_count", "hybrid_used_llm_ratio"],
    )

    recognizer_preferred_dir = Path(recognizer_metrics_path).parent if recognizer_metrics_path else None
    recognizer_sample_predictions, recognizer_sample_predictions_path = _find_optional_json(
        "recognizer_sample_predictions.json",
        preferred_dir=(ROOT / recognizer_preferred_dir) if recognizer_preferred_dir else None,
    )
    recognizer_confusion_matrices, recognizer_confusion_matrices_path = _find_optional_json(
        "recognizer_confusion_matrices.json",
        preferred_dir=(ROOT / recognizer_preferred_dir) if recognizer_preferred_dir else None,
    )
    recognizer_plot_bundle, recognizer_plot_bundle_path = _find_optional_json(
        "recognizer_plot_bundle.json",
        preferred_dir=(ROOT / recognizer_preferred_dir) if recognizer_preferred_dir else None,
    )
    if recognizer_sample_predictions is None:
        notes_on_missing_sections.append("Missing recognizer_sample_predictions.json")
    if recognizer_confusion_matrices is None:
        notes_on_missing_sections.append("Missing recognizer_confusion_matrices.json")
    if recognizer_plot_bundle is None:
        notes_on_missing_sections.append("Missing recognizer_plot_bundle.json")

    representative_training_curves = []
    representative_eval_traces = []
    for item in enriched_runs:
        data = item["data"]
        if isinstance(data.get("training_curve"), list) and data.get("training_curve"):
            representative_training_curves.append({"source": item["path"], "training_curve": data.get("training_curve")})
        if isinstance(data.get("representative_eval_trace"), list) and data.get("representative_eval_trace"):
            representative_eval_traces.append(
                {
                    "source": item["path"],
                    "representative_eval_trace": data.get("representative_eval_trace"),
                    "representative_eval_trace_meta": data.get("representative_eval_trace_meta", data.get("representative_eval_summary", {})),
                }
            )

    bundle = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "builder_script": "build_paper_plot_bundle.py",
            "source_reports": {
                "final_multiseed_validation": "reports/final_multiseed_validation_20260412.json",
                "iterative_uncertain_repair": "reports/iterative_uncertain_repair_20260412.json",
                "recognizer_metrics_latest": "reports/recognizer_metrics_latest.json",
            },
            "actual_source_paths": {
                "final_multiseed_validation": final_validation_path,
                "iterative_uncertain_repair": iterative_repair_path,
                "recognizer_metrics_latest": recognizer_metrics_path,
            },
            "enriched_run_count": len(enriched_runs),
            "outer_loop_round_trace_count": len(outer_loop_round_traces),
        },
        "final_benchmark": final_validation,
        "iterative_repair": iterative_repair,
        "recognition_support": {
            "recognizer_metrics_latest": recognizer_metrics,
            "recognition_summary_csv": "outputs/paper_plot_bundle/recognition_summary.csv",
            "recognizer_sample_predictions": recognizer_sample_predictions,
            "recognizer_sample_predictions_source": recognizer_sample_predictions_path,
            "recognizer_confusion_matrices": recognizer_confusion_matrices,
            "recognizer_confusion_matrices_source": recognizer_confusion_matrices_path,
            "recognizer_plot_bundle": recognizer_plot_bundle,
            "recognizer_plot_bundle_source": recognizer_plot_bundle_path,
        },
        "representative_training_curves": representative_training_curves,
        "representative_eval_traces": representative_eval_traces,
        "outer_loop_round_traces": outer_loop_round_traces,
        "notes_on_missing_sections": notes_on_missing_sections,
    }

    (OUTPUT_DIR / "paper_plot_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    readme = (
        "# Paper Plot Bundle\n\n"
        "Generated by `build_paper_plot_bundle.py`.\n\n"
        "## Files\n"
        "- `paper_plot_bundle.json`: merged plotting-ready JSON bundle.\n"
        "- `final_benchmark_per_run.csv`: per-run benchmark rows.\n"
        "- `final_benchmark_summary.csv`: split/mode summary rows.\n"
        "- `iterative_repair_summary.csv`: iterative repair aggregate rows.\n"
        "- `recognition_summary.csv`: recognizer metrics by split and mode.\n\n"
        "## Missing data behavior\n"
        "If source artifacts are unavailable, this bundle remains valid and documents gaps under `notes_on_missing_sections`.\n"
    )
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({"output_dir": str(OUTPUT_DIR), "notes_on_missing_sections": notes_on_missing_sections}, indent=2))


if __name__ == "__main__":
    main()
