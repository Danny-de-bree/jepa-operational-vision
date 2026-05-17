from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .degradation import DegradationConfig, degrade_image
from .ijepa_localization import IJepaPatchLocalizer
from .obstacle_dataset import DEFAULT_OBSTACLE_DATASET, load_balanced_obstacle_rows, load_obstacle_image, parse_yolo_boxes
from .prototypes import build_class_prototypes, guess_objects_with_prototypes
from .small_head import guess_objects_with_head, train_small_head


def run_bulk_eval(
    dataset_name: str,
    split: str,
    model_name: str,
    eval_samples: int,
    support_samples: int,
    seed: int,
    output: str,
    degradation_mode: str,
    degradation_ratio: float,
    patch_size: int,
    fill: str,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = resolve_output_path(output, run_id)
    localizer = IJepaPatchLocalizer(model_name=model_name)
    support_rows = load_balanced_obstacle_rows(
        dataset_name,
        split,
        support_samples,
        random_seed=seed + 10_000,
    )
    support_files = {row["file_name"] for row in support_rows}
    eval_rows = load_disjoint_eval_rows(dataset_name, split, eval_samples, seed, support_files)
    prototypes = build_class_prototypes(dataset_name, split, support_rows, localizer)
    head = train_small_head(dataset_name, split, support_rows, localizer)
    degradation = DegradationConfig(
        mode=degradation_mode,
        ratio=degradation_ratio,
        patch_size=patch_size,
        fill=fill,
        seed=seed,
    )

    records = []
    for sample_index, row in enumerate(eval_rows):
        clean_image = load_obstacle_image(dataset_name, row, split)
        image = degrade_image(clean_image, degradation, sample_index=sample_index)
        boxes = parse_yolo_boxes(row)
        prototype_guesses = guess_objects_with_prototypes(image, boxes, localizer, prototypes)
        head_guesses = guess_objects_with_head(image, boxes, localizer, head)
        head_by_object = {guess.object_index: guess for guess in head_guesses}
        prototype_by_object = {guess.object_index: guess for guess in prototype_guesses}

        for object_index, box in enumerate(boxes, start=1):
            prototype_guess = prototype_by_object.get(object_index)
            head_guess = head_by_object.get(object_index)
            records.append(
                {
                    "sample": sample_index,
                    "run_id": run_id,
                    "support_samples": support_samples,
                    "eval_samples": eval_samples,
                    "seed": seed,
                    "degradation_mode": degradation.mode,
                    "degradation_ratio": degradation.ratio,
                    "degradation_patch_size": degradation.patch_size,
                    "degradation_fill": degradation.fill,
                    "file_name": row["file_name"],
                    "object": object_index,
                    "yolo_label": box.class_name,
                    "prototype_guess": prototype_guess.ijepa_guess if prototype_guess else None,
                    "prototype_similarity": prototype_guess.similarity if prototype_guess else None,
                    "prototype_agreement": prototype_guess.agreement if prototype_guess else None,
                    "head_guess": head_guess.head_guess if head_guess else None,
                    "head_confidence": head_guess.confidence if head_guess else None,
                    "head_agreement": head_guess.agreement if head_guess else None,
                    "head_train_objects": head.train_objects if head else None,
                    "head_train_accuracy": head.train_accuracy if head else None,
                    "head_parameter_count": head.parameter_count if head else None,
                    "prototype_classes": len(prototypes),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    write_summaries(df, output_path)
    return output_path


def load_disjoint_eval_rows(dataset_name: str, split: str, eval_samples: int, seed: int, support_files: set[str]):
    candidates = load_balanced_obstacle_rows(
        dataset_name,
        split,
        eval_samples + len(support_files) + 100,
        random_seed=seed,
    )
    selected = []
    for row in candidates:
        if row["file_name"] in support_files:
            continue
        selected.append(row)
        if len(selected) >= eval_samples:
            break
    if len(selected) < eval_samples:
        raise ValueError(f"Could only find {len(selected)} disjoint eval rows; requested {eval_samples}.")
    return selected


def write_summaries(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return

    summary = {
        "run_id": str(df["run_id"].iloc[0]) if "run_id" in df else None,
        "support_samples": first_int(df, "support_samples"),
        "eval_samples": first_int(df, "eval_samples"),
        "seed": first_int(df, "seed"),
        "degradation_mode": str(df["degradation_mode"].iloc[0]) if "degradation_mode" in df else "none",
        "degradation_ratio": first_float(df, "degradation_ratio"),
        "degradation_patch_size": first_int(df, "degradation_patch_size"),
        "degradation_fill": str(df["degradation_fill"].iloc[0]) if "degradation_fill" in df else None,
        "objects": int(len(df)),
        "classes": int(df["yolo_label"].nunique()),
        "prototype_accuracy": safe_accuracy(df, "prototype_agreement"),
        "head_accuracy": safe_accuracy(df, "head_agreement"),
        "prototype_macro_accuracy": macro_accuracy(df, "prototype_agreement"),
        "head_macro_accuracy": macro_accuracy(df, "head_agreement"),
        "head_train_objects": first_int(df, "head_train_objects"),
        "head_parameter_count": first_int(df, "head_parameter_count"),
        "prototype_classes": first_int(df, "prototype_classes"),
    }
    summary_path = sibling_path(output_path, "_summary.csv")
    per_class_path = sibling_path(output_path, "_per_class.csv")
    prototype_confusion_path = sibling_path(output_path, "_prototype_confusion.csv")
    head_confusion_path = sibling_path(output_path, "_head_confusion.csv")
    report_path = sibling_path(output_path, "_report.json")

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    per_class = (
        df.groupby("yolo_label")
        .agg(
            objects=("yolo_label", "size"),
            prototype_accuracy=("prototype_agreement", safe_mean),
            head_accuracy=("head_agreement", safe_mean),
            avg_prototype_similarity=("prototype_similarity", "mean"),
            avg_head_confidence=("head_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["head_accuracy", "objects"], ascending=[True, False])
    )
    per_class.to_csv(per_class_path, index=False)

    prototype_confusion = confusion_frame(df["yolo_label"], df["prototype_guess"])
    head_confusion = confusion_frame(df["yolo_label"], df["head_guess"])
    prototype_confusion.to_csv(prototype_confusion_path)
    head_confusion.to_csv(head_confusion_path)
    report = {
        "summary": summary,
        "per_class": per_class.replace({np.nan: None}).to_dict(orient="records"),
        "top_prototype_confusions": top_confusions(df, "prototype_guess"),
        "top_head_confusions": top_confusions(df, "head_guess"),
        "files": {
            "objects": str(output_path),
            "summary": str(summary_path),
            "per_class": str(per_class_path),
            "prototype_confusion": str(prototype_confusion_path),
            "head_confusion": str(head_confusion_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def sibling_path(output_path: Path, suffix: str) -> Path:
    names = {
        "_summary.csv": "summary.csv",
        "_per_class.csv": "per_class.csv",
        "_prototype_confusion.csv": "prototype_confusion.csv",
        "_head_confusion.csv": "head_confusion.csv",
        "_report.json": "report.json",
    }
    return output_path.with_name(names.get(suffix, f"{output_path.stem}{suffix}"))


def resolve_output_path(output: str, run_id: str) -> Path:
    path = Path(output)
    if output == "auto":
        return Path("outputs") / f"run_{run_id}" / "objects.csv"
    if "{timestamp}" in output or "{run_id}" in output:
        resolved = Path(output.format(timestamp=run_id, run_id=run_id))
        if resolved.suffix:
            return resolved
        return resolved / "objects.csv"
    if path.suffix == "":
        return path / "objects.csv"
    if path.exists():
        return path.with_name(f"{path.stem}_{run_id}{path.suffix}")
    return path


def safe_mean(values) -> float:
    clean = pd.Series(values).dropna()
    if clean.empty:
        return np.nan
    return float(clean.astype(bool).mean())


def safe_accuracy(df: pd.DataFrame, column: str) -> float | None:
    clean = df[column].dropna()
    if clean.empty:
        return None
    return float(clean.astype(bool).mean())


def macro_accuracy(df: pd.DataFrame, column: str) -> float | None:
    clean = df.dropna(subset=[column])
    if clean.empty:
        return None
    return float(clean.groupby("yolo_label")[column].apply(safe_mean).mean())


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    clean = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()
    labels = sorted(set(clean["true"]).union(set(clean["pred"])))
    matrix = confusion_matrix(clean["true"], clean["pred"], labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def top_confusions(df: pd.DataFrame, prediction_column: str, limit: int = 10) -> list[dict]:
    clean = df.dropna(subset=[prediction_column])
    wrong = clean[clean["yolo_label"] != clean[prediction_column]]
    if wrong.empty:
        return []
    counts = (
        wrong.groupby(["yolo_label", prediction_column])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(limit)
    )
    return [
        {"yolo_label": row["yolo_label"], "predicted": row[prediction_column], "count": int(row["count"])}
        for _, row in counts.iterrows()
    ]


def first_int(df: pd.DataFrame, column: str) -> int | None:
    clean = df[column].dropna()
    if clean.empty:
        return None
    return int(clean.iloc[0])


def first_float(df: pd.DataFrame, column: str) -> float | None:
    clean = df[column].dropna()
    if clean.empty:
        return None
    return float(clean.iloc[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bulk I-JEPA prototype/head evaluation.")
    parser.add_argument("--dataset-name", default=DEFAULT_OBSTACLE_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="facebook/ijepa_vith14_1k")
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument(
        "--support-samples",
        "--reference-samples",
        dest="support_samples",
        type=int,
        default=80,
        help="Images used to build class prototypes and train the tiny classifier.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--degradation-mode",
        choices=["none", "pixel", "patch", "both"],
        default="none",
        help="Apply degradation to eval images only.",
    )
    parser.add_argument("--degradation-ratio", type=float, default=0.0)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--fill", choices=["black", "gray", "noise"], default="gray")
    parser.add_argument(
        "--output",
        default="auto",
        help="Output CSV path or run directory. Supports {timestamp} or {run_id}. Default: outputs/run_<timestamp>/objects.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run_bulk_eval(**vars(args))
    print(f"Saved bulk evaluation: {output}")
    for suffix in [
        "_summary.csv",
        "_per_class.csv",
        "_prototype_confusion.csv",
        "_head_confusion.csv",
        "_report.json",
    ]:
        print(f"Saved: {sibling_path(output, suffix)}")


if __name__ == "__main__":
    main()
