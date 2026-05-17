from __future__ import annotations

import traceback

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.context_analysis import analyze_object_contexts
from src.ijepa_localization import IJepaPatchLocalizer, iou
from src.obstacle_dataset import (
    DEFAULT_OBSTACLE_DATASET,
    load_balanced_obstacle_rows,
    load_obstacle_image,
    load_obstacle_rows,
    parse_yolo_boxes,
)
from src.prototypes import build_class_prototypes, guess_objects_with_prototypes
from src.small_head import guess_objects_with_head, train_small_head
from src.visualization import draw_yolo_with_heatmap


DEFAULT_SPLIT = "train"
DEFAULT_MODEL = "facebook/ijepa_vith14_1k"


@st.cache_resource(show_spinner=False)
def get_localizer(model_name: str) -> IJepaPatchLocalizer:
    return IJepaPatchLocalizer(model_name=model_name)


@st.cache_data(show_spinner=False)
def get_rows(
    dataset_name: str,
    split: str,
    max_samples: int,
    min_objects: int,
    sample_mode: str,
    random_seed: int,
):
    if sample_mode == "balanced single/multiple":
        return load_balanced_obstacle_rows(dataset_name, split, max_samples, random_seed=random_seed)
    return load_obstacle_rows(
        dataset_name,
        split,
        max_samples,
        min_objects=min_objects,
        random_seed=random_seed,
    )


def run_localization_probe(
    dataset_name: str,
    split: str,
    model_name: str,
    max_samples: int,
    min_objects: int,
    sample_mode: str,
    saliency_threshold: float,
    analyze_context: bool,
    use_prototypes: bool,
    prototype_samples: int,
    train_head: bool,
    head_train_samples: int,
    random_seed: int,
):
    rows = get_rows(dataset_name, split, max_samples, min_objects, sample_mode, random_seed)
    localizer = get_localizer(model_name)
    prototypes = {}
    if use_prototypes:
        prototype_rows = load_balanced_obstacle_rows(
            dataset_name,
            split,
            prototype_samples,
            random_seed=random_seed + 10_000,
        )
        prototypes = build_class_prototypes(dataset_name, split, prototype_rows, localizer)
    trained_head = None
    if train_head:
        head_rows = load_balanced_obstacle_rows(
            dataset_name,
            split,
            head_train_samples,
            random_seed=random_seed + 20_000,
        )
        trained_head = train_small_head(dataset_name, split, head_rows, localizer)

    results = []
    overlays = []
    image_embeddings = []
    yolo_labels = []

    progress = st.progress(0)
    for index, row in enumerate(rows):
        image = load_obstacle_image(dataset_name, row, split)
        yolo_boxes = parse_yolo_boxes(row)
        localization = localizer.localize(image)
        yolo_xyxy = [box.to_xyxy(*image.size) for box in yolo_boxes]
        context_results = analyze_object_contexts(image, yolo_boxes, localizer) if analyze_context else []
        prototype_guesses = (
            guess_objects_with_prototypes(image, yolo_boxes, localizer, prototypes)
            if use_prototypes
            else []
        )
        head_guesses = (
            guess_objects_with_head(image, yolo_boxes, localizer, trained_head)
            if train_head
            else []
        )
        candidate_boxes = getattr(localization, "candidate_boxes_xyxy", None) or [localization.box_xyxy]
        all_ijepa_boxes = candidate_boxes
        best_iou = max((iou(candidate, box) for candidate in all_ijepa_boxes for box in yolo_xyxy), default=0.0)
        structure_guess = describe_object_structure(len(candidate_boxes))
        yolo_structure = describe_yolo_structure(len(yolo_boxes))
        structure_agrees = structure_matches_yolo(len(candidate_boxes), len(yolo_boxes))
        class_names = sorted({box.class_name for box in yolo_boxes})
        yolo_label = ", ".join(class_names) if class_names else "none"
        image_embeddings.append(localization.image_embedding)
        yolo_labels.append(yolo_label)

        results.append(
            {
                "sample": index,
                "file_name": row["file_name"],
                "yolo_objects": yolo_label,
                "ijepa_object_guess": "pending",
                "num_yolo_objects": len(yolo_boxes),
                "context_patterns": ", ".join(
                    sorted({result.context_pattern for result in context_results})
                )
                if context_results
                else "not analyzed",
                "prototype_agreement": round(
                    sum(guess.agreement for guess in prototype_guesses) / len(prototype_guesses),
                    4,
                )
                if prototype_guesses
                else None,
                "small_head_agreement": round(
                    sum(guess.agreement for guess in head_guesses) / len(head_guesses),
                    4,
                )
                if head_guesses
                else None,
                "ijepa_salient_regions": len(candidate_boxes),
                "ijepa_structure_guess": structure_guess,
                "yolo_structure": yolo_structure,
                "structure_agrees": structure_agrees,
                "best_iou_vs_yolo": round(best_iou, 4),
                "ijepa_score": round(localization.score, 4),
            }
        )
        overlays.append(
            {
                "sample": index,
                "image": draw_yolo_with_heatmap(
                    image,
                    yolo_boxes,
                    localization.heatmap,
                    saliency_threshold=saliency_threshold,
                ),
                "heatmap": localization.heatmap,
                "objects": yolo_label,
                "best_iou": best_iou,
                "candidate_boxes": candidate_boxes,
                "num_yolo_boxes": len(yolo_boxes),
                "structure_guess": structure_guess,
                "yolo_structure": yolo_structure,
                "structure_agrees": structure_agrees,
                "context_results": context_results,
                "prototype_guesses": prototype_guesses,
                "head_guesses": head_guesses,
                "head_train_accuracy": trained_head.train_accuracy if trained_head else None,
                "head_train_objects": trained_head.train_objects if trained_head else None,
                "head_parameter_count": trained_head.parameter_count if trained_head else None,
                "prototype_classes": len(prototypes),
                "representation_report": build_representation_report(
                    yolo_label,
                    structure_guess,
                    context_results,
                    head_guesses,
                    prototype_guesses,
                    trained_head.parameter_count if trained_head else None,
                ),
            }
        )
        progress.progress((index + 1) / len(rows))

    if len(image_embeddings) > 1:
        similarities = cosine_similarity(image_embeddings)
        for index in range(len(results)):
            similarities[index, index] = -1
            neighbor = int(similarities[index].argmax())
            results[index]["ijepa_object_guess"] = yolo_labels[neighbor]
            results[index]["nearest_labeled_sample"] = neighbor
            results[index]["object_guess_similarity"] = round(float(similarities[index, neighbor]), 4)
    else:
        results[0]["ijepa_object_guess"] = "needs at least 2 samples"
        results[0]["nearest_labeled_sample"] = None
        results[0]["object_guess_similarity"] = None

    return pd.DataFrame(results), overlays


def describe_object_structure(salient_regions: int) -> str:
    if salient_regions <= 1:
        return "single focus"
    if salient_regions <= 3:
        return "multi-region"
    return "distributed/group pattern"


def describe_yolo_structure(yolo_objects: int) -> str:
    if yolo_objects <= 1:
        return "single object"
    if yolo_objects <= 3:
        return "multiple objects"
    return "group of objects"


def structure_matches_yolo(salient_regions: int, yolo_objects: int) -> bool:
    yolo_is_multi = yolo_objects > 1
    ijepa_is_multi = salient_regions > 1
    return yolo_is_multi == ijepa_is_multi


def build_representation_report(
    yolo_objects: str,
    scene_structure: str,
    context_results,
    head_guesses,
    prototype_guesses,
    head_parameter_count: int | None,
) -> str:
    head_summary = "not trained"
    if head_guesses:
        labels = sorted({guess.head_guess for guess in head_guesses})
        confidence = sum(guess.confidence for guess in head_guesses) / len(head_guesses)
        head_summary = f"{', '.join(labels)} (avg confidence {confidence:.2f})"

    prototype_summary = "not run"
    if prototype_guesses:
        labels = sorted({guess.ijepa_guess for guess in prototype_guesses})
        prototype_summary = ", ".join(labels)

    context_summary = "not analyzed"
    if context_results:
        patterns = sorted({result.context_pattern for result in context_results})
        strengths = sorted({result.context_strength for result in context_results})
        context_summary = f"{', '.join(patterns)} | strength: {', '.join(strengths)}"

    parameter_text = f"{head_parameter_count:,} trainable params" if head_parameter_count else "no trainable head"
    return (
        f"YOLO reference: {yolo_objects}. "
        f"Tiny classifier: {head_summary}. "
        f"Prototype match: {prototype_summary}. "
        f"Scene structure: {scene_structure}. "
        f"Context: {context_summary}. "
        f"Head size: {parameter_text}."
    )


st.set_page_config(page_title="JEPA-demo", layout="wide")

st.title("JEPA-demo")
st.caption(
    "I-JEPA patch-representation probe for obstacle localization, compared against YOLO-format "
    "dataset labels as the benchmark."
)

st.markdown(
    """
    **Representation-first operational vision:** YOLO provides object-level grounding, while
    frozen I-JEPA provides scene/context representations. A tiny logistic-regression head can
    test whether those representations are already enough to say what an object likely is.
    """
)

with st.expander("What is what?", expanded=True):
    st.markdown(
        """
        - **YOLO**: benchmark labels and boxes from the dataset.
        - **I-JEPA saliency**: orange overlay showing where frozen I-JEPA has strong patch-level representation activity.
        - **Prototype match**: compares an object crop with average I-JEPA embeddings per YOLO class.
        - **Tiny classifier**: logistic regression trained on frozen I-JEPA crop embeddings.
        - **Isolated object**: one clear focus point, relatively separate from the scene.
        - **Group-like scene**: several similar objects or people forming a cluster.
        - **Context-heavy surroundings**: the area around the object contributes strongly to the scene meaning.
        - **Multi-region visual structure**: several separate visually important regions, not necessarily one group.
        """
    )

with st.sidebar:
    st.header("Experiment")
    dataset_name = st.text_input("Dataset", value=DEFAULT_OBSTACLE_DATASET)
    split = DEFAULT_SPLIT
    model_name = st.text_input("I-JEPA model", value=DEFAULT_MODEL)
    max_samples = st.slider("Max samples", min_value=1, max_value=25, value=3, step=1)
    random_seed = st.number_input("Sample seed", min_value=0, value=7, step=1)
    sample_mode = st.selectbox(
        "Sample mode",
        ["balanced single/multiple", "minimum object filter"],
    )
    min_objects = 1
    saliency_threshold = st.slider(
        "Saliency overlay threshold",
        min_value=0.4,
        max_value=0.95,
        value=0.7,
        step=0.05,
    )
    analyze_context = st.checkbox("Analyze object context", value=True)
    use_prototypes = st.checkbox("Match with class prototypes", value=True)
    prototype_samples = st.slider("Prototype reference images", min_value=4, max_value=80, value=12, step=4)
    train_head = st.checkbox("Train lightweight classifier", value=False)
    head_train_samples = st.slider(
        "Classifier training images",
        min_value=8,
        max_value=160,
        value=40,
        step=8,
    )
    run = st.button("Run", type="primary")

with st.expander("How to read the signals", expanded=False):
    st.markdown(
        """
        - **Green boxes**: YOLO-format benchmark labels from the dataset.
        - **Orange overlay**: I-JEPA patch saliency, showing where representation activity is strongest.
        - **Object guess**: nearest labeled example in I-JEPA embedding space.
        - **Context pattern**: rough signal for isolated objects, nearby objects, or group/crowd context.
        - **Context strength**: how strongly object/context/scene embeddings relate.
        - **Scene structure**: single focus, multi-region, or distributed/group pattern.
        """
    )

if run:
    try:
        with st.status("Running I-JEPA localization probe...", expanded=True):
            results, overlays = run_localization_probe(
                dataset_name=dataset_name,
                split=split,
                model_name=model_name,
                max_samples=max_samples,
                min_objects=min_objects,
                sample_mode=sample_mode,
                saliency_threshold=saliency_threshold,
                analyze_context=analyze_context,
                use_prototypes=use_prototypes,
                prototype_samples=prototype_samples,
                train_head=train_head,
                head_train_samples=head_train_samples,
                random_seed=int(random_seed),
            )
            st.session_state["results"] = results
            st.session_state["overlays"] = overlays
            st.success("Probe completed.")
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.code(traceback.format_exc(limit=8))

results = st.session_state.get("results")
overlays = st.session_state.get("overlays", [])

if results is not None and not results.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", len(results))
    col2.metric("YOLO objects", int(results["num_yolo_objects"].sum()))
    prototype_classes = overlays[0].get("prototype_classes") if overlays else None
    col3.metric("Prototype classes", prototype_classes if prototype_classes else "off")
    head_objects = overlays[0].get("head_train_objects") if overlays else None
    col4.metric("Classifier train crops", head_objects if head_objects else "off")
    if overlays and overlays[0].get("head_parameter_count"):
        st.caption(
            f"Tiny classifier size: {overlays[0]['head_parameter_count']:,} trainable parameters "
            "on top of frozen I-JEPA."
        )

    st.subheader("Run details")
    st.dataframe(results, width="stretch")

    st.subheader("Representation reports")
    for item in overlays:
        st.markdown(f"**Sample {item['sample']} - objects: {item['objects']}**")
        st.info(item["representation_report"])
        image_col, detail_col = st.columns([2, 1])
        with image_col:
            st.image(
                item["image"],
                caption=f"Green: YOLO benchmark labels. Orange: I-JEPA saliency. Best IoU proxy: {item['best_iou']:.3f}",
                width="stretch",
            )
        with detail_col:
            st.metric("YOLO boxes", item["num_yolo_boxes"])
            st.metric("I-JEPA salient regions", len(item["candidate_boxes"]))
            st.metric("Scene structure match", "yes" if item["structure_agrees"] else "no")
            st.write("YOLO structure")
            st.info(item["yolo_structure"])
            st.write("I-JEPA structure guess")
            st.info(item["structure_guess"])
            if item["context_results"]:
                st.write("Object context")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "object": result.object_index,
                                "class": result.class_name,
                                "pattern": result.context_pattern,
                                "strength": result.context_strength,
                                "scene_structure": item["structure_guess"],
                                "object_context_similarity": round(
                                    result.object_context_similarity, 3
                                ),
                                "scene_context_similarity": round(result.scene_context_similarity, 3),
                            }
                            for result in item["context_results"]
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            if item["prototype_guesses"]:
                st.write("YOLO vs I-JEPA prototype")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "object": guess.object_index,
                                "yolo_label": guess.yolo_label,
                                "ijepa_guess": guess.ijepa_guess,
                                "agreement": "yes" if guess.agreement else "no",
                                "similarity": round(guess.similarity, 3),
                            }
                            for guess in item["prototype_guesses"]
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            if item["head_guesses"]:
                st.write("YOLO vs small head")
                if item["head_train_accuracy"] is not None:
                    st.caption(
                        f"Head train accuracy: {item['head_train_accuracy']:.0%} "
                        f"on {item['head_train_objects']} object crops. "
                        f"Trainable parameters: {item['head_parameter_count']:,}"
                    )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "object": guess.object_index,
                                "yolo_label": guess.yolo_label,
                                "head_guess": guess.head_guess,
                                "agreement": "yes" if guess.agreement else "no",
                                "confidence": round(guess.confidence, 3),
                            }
                            for guess in item["head_guesses"]
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
else:
    st.info("Run a small sample first. The first I-JEPA model download can take a while.")

st.divider()
st.subheader("Embedding Space Explorer")
projection_file = st.text_input("Embedding projection CSV", value="")
if projection_file:
    try:
        projection_df = pd.read_csv(projection_file)
        fig = px.scatter(
            projection_df,
            x="x",
            y="y",
            color="label",
            hover_data=["file_name", "object"],
            title="I-JEPA object-crop embedding projection",
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.warning(f"Could not load projection file: {exc}")
