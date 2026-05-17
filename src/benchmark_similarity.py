from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .utils import embedding_matrix, read_embeddings


def benchmark_similarity(embeddings: str, top_k: int = 5) -> dict[str, Any]:
    df = read_embeddings(embeddings)
    matrix = embedding_matrix(df)
    if len(df) < 2:
        raise ValueError("Need at least two embeddings for nearest-neighbor benchmarking.")

    scores = cosine_similarity(matrix)
    np.fill_diagonal(scores, -np.inf)
    k = max(1, min(top_k, len(df) - 1))
    neighbor_indices = np.argsort(-scores, axis=1)[:, :k]
    labels = df["label"].astype(str).to_numpy()
    matches = labels[neighbor_indices] == labels[:, None]
    top_k_accuracy = float(matches.any(axis=1).mean())

    examples = []
    for query_idx in range(min(5, len(df))):
        nearest_idx = int(neighbor_indices[query_idx, 0])
        examples.append(
            {
                "query_sample_id": str(df.iloc[query_idx]["sample_id"]),
                "query_label": str(df.iloc[query_idx].get("label_name", df.iloc[query_idx]["label"])),
                "neighbor_sample_id": str(df.iloc[nearest_idx]["sample_id"]),
                "neighbor_label": str(df.iloc[nearest_idx].get("label_name", df.iloc[nearest_idx]["label"])),
                "cosine_similarity": float(scores[query_idx, nearest_idx]),
                "match": bool(labels[query_idx] == labels[nearest_idx]),
            }
        )

    return {
        "samples": int(len(df)),
        "embedding_dim": int(matrix.shape[1]),
        "top_k": int(k),
        "top_k_accuracy": top_k_accuracy,
        "examples": examples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a nearest-neighbor similarity benchmark.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = benchmark_similarity(args.embeddings, args.top_k)
    print(f"Samples: {result['samples']}")
    print(f"Embedding dimension: {result['embedding_dim']}")
    print(f"Top-{result['top_k']} retrieval accuracy: {result['top_k_accuracy']:.3f}")
    if result["examples"]:
        print("Example nearest neighbor:")
        for key, value in result["examples"][0].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
