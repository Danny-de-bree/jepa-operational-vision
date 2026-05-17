---
title: JEPA Demo
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# JEPA-demo

Streamlit demo for probing frozen I-JEPA visual representations against YOLO-format obstacle labels.

This repository is designed to run locally and as a Docker-based Hugging Face Space.

## Concept

This project explores representation-first operational vision:

- YOLO provides object-level grounding: labels, boxes, and counts.
- I-JEPA provides frozen visual representations: saliency, context, and scene structure.
- A tiny logistic-regression classifier tests whether frozen I-JEPA embeddings are already enough to classify object crops.

The classifier is intentionally small. Its trainable parameter count is approximately:

```text
embedding_dim * num_classes + num_classes
```

For example, with a 1,280-dimensional I-JEPA embedding and 10 classes, the head has about 12,810 trainable parameters while I-JEPA remains frozen.

## Logistic Regression Head

The lightweight classifier is logistic regression, not linear regression.

Linear regression predicts a continuous value:

```text
features -> linear formula -> number
```

Logistic regression uses a linear formula too, but converts class scores into probabilities:

```text
I-JEPA crop embedding
        |
linear scores per class
        |
softmax probabilities
        |
predicted class
```

In this project, that means the large I-JEPA model provides the visual embedding and stays frozen. The tiny logistic-regression head only learns simple linear decision boundaries between YOLO classes in that embedding space.

The goal is not only object detection. The goal is to inspect whether a representation model can support broader scene understanding: isolated objects, group-like scenes, context-heavy surroundings, and multi-region visual structure.

## Defaults

| Component | Value |
| --- | --- |
| Dataset | `Abtinz/Obstacle-Detection-Dataset-YOLO` |
| Model | `facebook/ijepa_vith14_1k` |
| UI | Streamlit |
| Dependency manager | `uv` + `pyproject.toml` + `uv.lock` |
| Torch build | CPU-only |

## Features

- Loads YOLO-format obstacle metadata from a Hugging Face dataset.
- Downloads source images from the dataset repository.
- Runs frozen I-JEPA image and patch embeddings.
- Displays YOLO boxes as benchmark labels.
- Overlays I-JEPA patch saliency on the image.
- Estimates scene structure from connected saliency regions.
- Compares YOLO labels with I-JEPA class prototypes built from reference images.
- Optionally trains a tiny `LogisticRegression` classifier on frozen I-JEPA crop embeddings.
- Analyzes object context using object crop, context crop, and scene embeddings.

## Local Setup

```bash
cd ~/projects/JEPA-demo
uv sync
```

Run the app:

```bash
uv run streamlit run app.py
```

Open the Streamlit URL, usually:

```text
http://localhost:8501
```

The first run downloads the I-JEPA checkpoint and dataset images.

## Bulk Evaluation

For larger offline runs, use:

```bash
uv run python -m src.bulk_eval \
  --eval-samples 50 \
  --support-samples 80
```

`--support-samples` is used for both class prototypes and tiny-classifier training. `--eval-samples` is the number of images evaluated afterward. Support and eval images are kept disjoint by file name.

Outputs:

```text
outputs/bulk_eval.csv
outputs/run_YYYYMMDD_HHMMSS/objects.csv
outputs/run_YYYYMMDD_HHMMSS/summary.csv
outputs/run_YYYYMMDD_HHMMSS/per_class.csv
outputs/run_YYYYMMDD_HHMMSS/prototype_confusion.csv
outputs/run_YYYYMMDD_HHMMSS/head_confusion.csv
outputs/run_YYYYMMDD_HHMMSS/report.json
```

By default, bulk runs create a timestamped run directory under `outputs/`. The main CSV also includes `run_id`, `support_samples`, `eval_samples`, and `seed` columns. The summary files include overall accuracy, macro accuracy, per-class accuracy, confusion matrices, and a compact JSON report with top confusions and file references.

### Degraded Image Evaluation

Bulk eval can degrade eval images while keeping support/training images clean. This is useful for testing how far prototype matching and the tiny classifier survive image corruption.

Random pixel masking:

```bash
uv run python -m src.bulk_eval \
  --eval-samples 50 \
  --support-samples 80 \
  --degradation-mode pixel \
  --degradation-ratio 0.25 \
  --fill gray
```

Random patch masking, closer to masked-image representation learning:

```bash
uv run python -m src.bulk_eval \
  --eval-samples 50 \
  --support-samples 80 \
  --degradation-mode patch \
  --degradation-ratio 0.25 \
  --patch-size 32 \
  --fill gray
```

Both degradations:

```bash
uv run python -m src.bulk_eval \
  --eval-samples 50 \
  --support-samples 80 \
  --degradation-mode both \
  --degradation-ratio 0.25 \
  --patch-size 32 \
  --fill gray
```

Supported degradation modes:

```text
none
pixel
patch
both
```

Supported fill modes:

```text
black
gray
noise
```

## Embedding Space Projection

Export object-crop embeddings to a 2D projection CSV:

```bash
uv run python -m src.embedding_plot \
  --samples 80 \
  --method tsne \
  --output outputs/embedding_projection.csv
```

Then load `outputs/embedding_projection.csv` in the Streamlit app under **Embedding Space Explorer**. Classes that cluster cleanly are better separated in frozen I-JEPA embedding space; overlapping classes are likely to be confused by prototypes or a tiny linear classifier.

To project the exact objects from a bulk run:

```bash
uv run python -m src.embedding_plot \
  --from-bulk outputs/run_YYYYMMDD_HHMMSS/objects.csv \
  --method tsne \
  --output outputs/run_YYYYMMDD_HHMMSS/embedding_projection.csv
```

This preserves bulk-run columns such as `prototype_guess`, `head_guess`, and agreement flags in the projection CSV.

To compare clean vs degraded embeddings in one shared projection space:

```bash
uv run python -m src.embedding_plot \
  --from-bulk outputs/run_CLEAN/objects.csv \
  --compare-bulk outputs/run_DEGRADED/objects.csv \
  --method tsne \
  --output outputs/compare_clean_degraded_embedding_projection.csv
```

The standalone embedding page will show condition markers, optional density contours, and centroid arrows from `baseline` to `compare` per class.

You can also run the embedding explorer as a standalone Streamlit page:

```bash
uv run streamlit run pages/embedding_space.py
```

## Crop-Level Robustness Experiment

The next useful experiment is to isolate object identity from full-scene context. Instead of degrading the whole image first, use the YOLO boxes to extract clean object crops and then degrade the crops directly.

Pipeline:

```text
YOLO bounding boxes
        |
        v
clean object crops
        |
        +--> 0% pixel masking
        +--> 25% pixel masking
        +--> 50% pixel masking
        `--> 75% pixel masking
        |
        v
frozen I-JEPA embeddings
        |
        v
tiny LogisticRegression head
```

The intended evaluation:

- Extract object crops from clean images using the YOLO-format bounding boxes.
- Generate degraded crop variants at 0%, 25%, 50%, and 75% pixel masking.
- Embed all crop variants with the frozen I-JEPA encoder.
- Train the tiny logistic-regression head only on clean crops.
- Test the same head on degraded crops, especially the 75% masked crops.
- Export the embeddings to a t-SNE or UMAP projection for visual inspection.

This gives a cleaner robustness question:

```text
If the object is already localized, how much visual identity survives in I-JEPA embedding space?
```

Expected scenarios:

1. Strong visual concepts stay clustered.
   Classes such as cars, trees, dustbins, people, or trucks may remain close to their clean class clusters even under heavy crop degradation. That suggests the frozen representation contains a robust visual concept for those classes.

2. Weak or detail-heavy concepts collapse into attractors.
   Classes such as guard rails, electrical boxes, pedestrian crossings, or benches may drift toward broader visual attractors. That suggests the class depends more on fine details, exact geometry, texture, or surrounding context.

This is different from asking whether YOLO can still draw the box. YOLO gives the `where`. The crop-level test asks whether I-JEPA plus a tiny head can still recover the `what` once the location is already known.

Run the crop-level batch experiment:

```bash
uv run python -m src.crop_robustness \
  --eval-samples 50 \
  --support-samples 80 \
  --ratios 0,0.25,0.5,0.75 \
  --degradation-mode pixel \
  --fill gray
```

Outputs:

```text
outputs/crop_run_YYYYMMDD_HHMMSS/objects.csv
outputs/crop_run_YYYYMMDD_HHMMSS/summary.csv
outputs/crop_run_YYYYMMDD_HHMMSS/per_degradation.csv
outputs/crop_run_YYYYMMDD_HHMMSS/per_class.csv
outputs/crop_run_YYYYMMDD_HHMMSS/prototype_confusion.csv
outputs/crop_run_YYYYMMDD_HHMMSS/head_confusion.csv
outputs/crop_run_YYYYMMDD_HHMMSS/report.json
```

Create a t-SNE overlay for the crop variants:

```bash
uv run python -m src.embedding_plot \
  --from-bulk outputs/crop_run_YYYYMMDD_HHMMSS/objects.csv \
  --method tsne \
  --output outputs/crop_run_YYYYMMDD_HHMMSS/embedding_projection.csv
```

Then open:

```bash
uv run streamlit run pages/embedding_space.py
```

Load the generated `embedding_projection.csv`. Marker shape shows the overlay condition (`clean`, `pixel_25`, `pixel_50`, `pixel_75`). Color can be switched between YOLO label, head guess, prototype guess, and agreement flags.

The embedding dashboard also includes a **YOLO + JEPA Alignment Report**:

- condition-level accuracy for clean vs degraded crops
- strong classes that keep their representation under degradation
- weak/detail-heavy classes that collapse under degradation
- head and prototype confusion attractors
- filters for head mistakes and prototype mistakes

## Hugging Face Spaces

Use a Docker Space.

The included `Dockerfile`:

- installs `uv`
- runs `uv sync --locked`
- exposes port `7860`
- starts Streamlit on `0.0.0.0:7860`

## Method

```text
YOLO dataset metadata
        |
        v
download image
        |
        v
frozen I-JEPA
        |
        +--> patch embeddings -> saliency overlay + scene structure
        |
        +--> object crop embeddings -> prototype match
        |
        +--> object/context/scene embeddings -> context pattern
        |
        `--> optional LogisticRegression head -> class prediction
```

## UI Signals

- `Scene structure agreement`: whether I-JEPA saliency structure matches YOLO single-vs-multiple structure.
- `Prototype label agreement`: whether nearest class prototype matches the YOLO object label.
- `Small head agreement`: whether the optional lightweight classifier matches the YOLO object label.
- `Context pattern`: embedding-based estimate of object surroundings.
- `Context strength`: low/medium/high similarity signal between object, context, and scene embeddings.

Signal terms:

- `isolated / object-dominant`: one object is the main visual focus.
- `group / crowd context`: the object appears in a broader group-like situation.
- `near other objects / scene-embedded`: the object and surrounding scene are strongly related.
- `multi-region`: several separate salient visual regions.
- `distributed/group pattern`: saliency is spread across a broader clustered scene.

## Notes

I-JEPA is not used as a trained detector in this demo. YOLO labels are the benchmark reference. I-JEPA is used as a frozen representation model for saliency, class-prototype matching, context analysis, and lightweight supervised heads.

Prototype quality depends on support coverage. Rare or visually specific classes may require more support samples before prototype matching becomes stable.

## Project Layout

```text
.
|-- app.py
|-- Dockerfile
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- docs/
|   `-- blog_outline.md
`-- src/
    |-- __init__.py
    |-- benchmark_similarity.py
    |-- bulk_eval.py
    |-- context_analysis.py
    |-- extract_embeddings.py
    |-- embedding_plot.py
    |-- ijepa_localization.py
    |-- jepa_adapter.py
    |-- obstacle_dataset.py
    |-- prototypes.py
    |-- small_head.py
    |-- utils.py
    `-- visualization.py
```
