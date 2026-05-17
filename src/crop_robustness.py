from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .bulk_eval import confusion_frame, load_disjoint_eval_rows, macro_accuracy, safe_accuracy, top_confusions
from .degradation import DegradationConfig, degrade_image
from .ijepa_localization import IJepaPatchLocalizer
from .obstacle_dataset import DEFAULT_OBSTACLE_DATASET, load_balanced_obstacle_rows, load_obstacle_image, parse_yolo_boxes
from .prototypes import build_class_prototypes, safe_crop
from .small_head import train_small_head


def run_crop_robustness(
    dataset_name: str,
    split: str,
    model_name: str,
    eval_samples: int,
    support_samples: int,
    seed: int,
    ratios: str,
    degradation_mode: str,
    patch_size: int,
    fill: str,
    output: str,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = resolve_output_path(output, run_id)
    ratio_values = parse_ratios(ratios)

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
    prototype_names = list(prototypes)
    prototype_matrix = np.vstack([prototypes[name] for name in prototype_names]) if prototype_names else None
    head = train_small_head(dataset_name, split, support_rows, localizer)

    records = []
    for sample_index, row in enumerate(eval_rows):
        image = load_obstacle_image(dataset_name, row, split)
        boxes = parse_yolo_boxes(row)
        for object_index, box in enumerate(boxes, start=1):
            crop = safe_crop(image, box)
            if crop is None:
                continue
            for ratio in ratio_values:
                config = DegradationConfig(
                    mode="none" if ratio <= 0 else degradation_mode,
                    ratio=ratio,
                    patch_size=patch_size,
                    fill=fill,
                    seed=seed,
                )
                variant_index = stable_variant_index(sample_index, object_index, ratio)
                degraded_crop = degrade_image(crop, config, sample_index=variant_index)
                embedding = localizer.embed_image(degraded_crop)

                prototype_guess, prototype_similarity = guess_with_prototypes(
                    embedding,
                    prototype_names,
                    prototype_matrix,
                )
                head_guess, head_confidence = guess_with_head(embedding, head)
                condition = "clean" if ratio <= 0 else f"{degradation_mode}_{int(ratio * 100)}"
                records.append(
                    {
                        "sample": sample_index,
                        "run_id": run_id,
                        "experiment_type": "crop_robustness",
                        "condition": condition,
                        "support_samples": support_samples,
                        "eval_samples": eval_samples,
                        "seed": seed,
                        "degradation_scope": "crop",
                        "degradation_mode": "none" if ratio <= 0 else degradation_mode,
                        "degradation_ratio": ratio,
                        "degradation_patch_size": patch_size,
                        "degradation_fill": fill,
                        "file_name": row["file_name"],
                        "object": object_index,
                        "yolo_label": box.class_name,
                        "crop_width": crop.size[0],
                        "crop_height": crop.size[1],
                        "prototype_guess": prototype_guess,
                        "prototype_similarity": prototype_similarity,
                        "prototype_agreement": prototype_guess == box.class_name if prototype_guess else None,
                        "head_guess": head_guess,
                        "head_confidence": head_confidence,
                        "head_agreement": head_guess == box.class_name if head_guess else None,
                        "head_train_objects": head.train_objects if head else None,
                        "head_train_accuracy": head.train_accuracy if head else None,
                        "head_parameter_count": head.parameter_count if head else None,
                        "prototype_classes": len(prototypes),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    write_crop_summaries(df, output_path)
    return output_path


def guess_with_prototypes(
    embedding: np.ndarray,
    prototype_names: list[str],
    prototype_matrix: np.ndarray | None,
) -> tuple[str | None, float | None]:
    if prototype_matrix is None or not prototype_names:
        return None, None
    similarities = cosine_similarity([embedding], prototype_matrix)[0]
    best_index = int(np.argmax(similarities))
    return prototype_names[best_index], float(similarities[best_index])


def guess_with_head(embedding: np.ndarray, head) -> tuple[str | None, float | None]:
    if head is None:
        return None, None
    probabilities = head.classifier.predict_proba([embedding])[0]
    best_index = int(np.argmax(probabilities))
    return head.classes[best_index], float(probabilities[best_index])


def write_crop_summaries(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return

    summary = {
        "run_id": str(df["run_id"].iloc[0]),
        "experiment_type": "crop_robustness",
        "support_samples": first_value(df, "support_samples"),
        "eval_samples": first_value(df, "eval_samples"),
        "seed": first_value(df, "seed"),
        "degradation_scope": "crop",
        "degradation_mode": ",".join(sorted(df["degradation_mode"].astype(str).unique())),
        "degradation_ratios": ",".join(str(value) for value in sorted(df["degradation_ratio"].unique())),
        "objects": int(df[["file_name", "object"]].drop_duplicates().shape[0]),
        "rows": int(len(df)),
        "classes": int(df["yolo_label"].nunique()),
        "head_train_objects": first_value(df, "head_train_objects"),
        "head_parameter_count": first_value(df, "head_parameter_count"),
        "prototype_classes": first_value(df, "prototype_classes"),
    }

    per_degradation = (
        df.groupby(["condition", "degradation_ratio"])
        .apply(
            lambda group: pd.Series(
                {
                    "rows": int(len(group)),
                    "objects": int(group[["file_name", "object"]].drop_duplicates().shape[0]),
                    "classes": int(group["yolo_label"].nunique()),
                    "prototype_accuracy": safe_accuracy(group, "prototype_agreement"),
                    "head_accuracy": safe_accuracy(group, "head_agreement"),
                    "prototype_macro_accuracy": macro_accuracy(group, "prototype_agreement"),
                    "head_macro_accuracy": macro_accuracy(group, "head_agreement"),
                    "avg_prototype_similarity": group["prototype_similarity"].mean(),
                    "avg_head_confidence": group["head_confidence"].mean(),
                }
            )
        )
        .reset_index()
        .sort_values("degradation_ratio")
    )

    per_class = (
        df.groupby(["condition", "degradation_ratio", "yolo_label"])
        .apply(
            lambda group: pd.Series(
                {
                    "objects": int(group[["file_name", "object"]].drop_duplicates().shape[0]),
                    "prototype_accuracy": safe_accuracy(group, "prototype_agreement"),
                    "head_accuracy": safe_accuracy(group, "head_agreement"),
                    "avg_prototype_similarity": group["prototype_similarity"].mean(),
                    "avg_head_confidence": group["head_confidence"].mean(),
                }
            )
        )
        .reset_index()
        .sort_values(["degradation_ratio", "head_accuracy", "objects"], ascending=[True, True, False])
    )

    summary_path = output_path.with_name("summary.csv")
    per_degradation_path = output_path.with_name("per_degradation.csv")
    per_class_path = output_path.with_name("per_class.csv")
    prototype_confusion_path = output_path.with_name("prototype_confusion.csv")
    head_confusion_path = output_path.with_name("head_confusion.csv")
    report_path = output_path.with_name("report.json")

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    per_degradation.to_csv(per_degradation_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    confusion_frame(df["yolo_label"], df["prototype_guess"]).to_csv(prototype_confusion_path)
    confusion_frame(df["yolo_label"], df["head_guess"]).to_csv(head_confusion_path)

    report = {
        "summary": summary,
        "per_degradation": per_degradation.replace({np.nan: None}).to_dict(orient="records"),
        "top_prototype_confusions": top_confusions(df, "prototype_guess"),
        "top_head_confusions": top_confusions(df, "head_guess"),
        "files": {
            "objects": str(output_path),
            "summary": str(summary_path),
            "per_degradation": str(per_degradation_path),
            "per_class": str(per_class_path),
            "prototype_confusion": str(prototype_confusion_path),
            "head_confusion": str(head_confusion_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_ratios(value: str) -> list[float]:
    ratios = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    for ratio in ratios:
        if ratio < 0 or ratio > 1:
            raise ValueError("Ratios must be in the range [0, 1].")
    return ratios or [0.0]


def stable_variant_index(sample_index: int, object_index: int, ratio: float) -> int:
    return sample_index * 10_000 + object_index * 100 + int(ratio * 100)


def first_value(df: pd.DataFrame, column: str):
    clean = df[column].dropna()
    if clean.empty:
        return None
    value = clean.iloc[0]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def resolve_output_path(output: str, run_id: str) -> Path:
    if output == "auto":
        return Path("outputs") / f"crop_run_{run_id}" / "objects.csv"
    path = Path(output.format(timestamp=run_id, run_id=run_id))
    if path.suffix:
        return path
    return path / "objects.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run crop-level I-JEPA robustness evaluation.")
    parser.add_argument("--dataset-name", default=DEFAULT_OBSTACLE_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="facebook/ijepa_vith14_1k")
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--support-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ratios", default="0,0.25,0.5,0.75")
    parser.add_argument("--degradation-mode", choices=["pixel", "patch", "both"], default="pixel")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--fill", choices=["black", "gray", "noise"], default="gray")
    parser.add_argument(
        "--output",
        default="auto",
        help="Output CSV path or run directory. Supports {timestamp} or {run_id}. Default: outputs/crop_run_<timestamp>/objects.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run_crop_robustness(**vars(args))
    print(f"Saved crop robustness evaluation: {output}")
    for name in [
        "summary.csv",
        "per_degradation.csv",
        "per_class.csv",
        "prototype_confusion.csv",
        "head_confusion.csv",
        "report.json",
    ]:
        print(f"Saved: {output.with_name(name)}")


if __name__ == "__main__":
    main()
