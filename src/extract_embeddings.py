from __future__ import annotations

import argparse
from dataclasses import asdict

from tqdm.auto import tqdm

from .jepa_adapter import VisionEmbeddingAdapter, VisionEncoderConfig
from .utils import detect_label_names, embeddings_to_frame, load_image_split, save_embeddings


def extract_embeddings(
    dataset_name: str,
    split: str,
    image_column: str,
    label_column: str,
    model_name: str,
    max_samples: int,
    output_dir: str,
    batch_size: int = 8,
    device: str | None = None,
) -> dict:
    dataset = load_image_split(dataset_name, split, max_samples)
    if image_column not in dataset.column_names:
        raise ValueError(f"Image column '{image_column}' not found. Available columns: {dataset.column_names}")
    if label_column not in dataset.column_names:
        raise ValueError(f"Label column '{label_column}' not found. Available columns: {dataset.column_names}")

    config = VisionEncoderConfig(model_name=model_name, batch_size=batch_size, device=device)
    adapter = VisionEmbeddingAdapter(config)

    images = [row[image_column] for row in tqdm(dataset, desc="Loading samples")]
    labels = [row[label_column] for row in dataset]
    sample_ids = [str(i) for i in range(len(dataset))]
    embeddings = adapter.embed_images(images)
    label_names = detect_label_names(dataset, label_column)
    df = embeddings_to_frame(embeddings, labels, label_names, sample_ids, model_name, dataset_name, split)
    parquet_path = save_embeddings(df, output_dir)

    return {
        "output": str(parquet_path),
        "samples": len(df),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "config": asdict(config),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract image embeddings from a Hugging Face dataset.")
    parser.add_argument("--dataset-name", default="beans")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="labels")
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/beans")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = extract_embeddings(**vars(args))
    print(f"Saved embeddings: {result['output']}")
    print(f"Samples: {result['samples']}")
    print(f"Embedding dimension: {result['embedding_dim']}")


if __name__ == "__main__":
    main()
