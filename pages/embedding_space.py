from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def format_percent(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def accuracy_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    if "condition" not in df.columns:
        return pd.DataFrame()

    records = []
    for condition, group in df.groupby("condition"):
        record = {"condition": condition, "objects": len(group)}
        if "degradation_ratio" in group.columns:
            record["degradation_ratio"] = float(group["degradation_ratio"].dropna().iloc[0])
        if "head_agreement" in group.columns:
            clean = group["head_agreement"].dropna()
            record["head_accuracy"] = float(clean.astype(bool).mean()) if not clean.empty else None
        if "prototype_agreement" in group.columns:
            clean = group["prototype_agreement"].dropna()
            record["prototype_accuracy"] = float(clean.astype(bool).mean()) if not clean.empty else None
        if "head_confidence" in group.columns:
            record["avg_head_confidence"] = float(group["head_confidence"].mean())
        if "prototype_similarity" in group.columns:
            record["avg_prototype_similarity"] = float(group["prototype_similarity"].mean())
        records.append(record)

    result = pd.DataFrame(records)
    if "degradation_ratio" in result.columns:
        result = result.sort_values("degradation_ratio")
    return result


def class_stability(df: pd.DataFrame) -> pd.DataFrame:
    if not {"condition", "label", "head_agreement"}.issubset(df.columns):
        return pd.DataFrame()
    if "degradation_ratio" not in df.columns:
        return pd.DataFrame()

    per_class = (
        df.dropna(subset=["head_agreement"])
        .groupby(["label", "condition", "degradation_ratio"])["head_agreement"]
        .apply(lambda values: values.astype(bool).mean())
        .reset_index(name="head_accuracy")
    )
    clean = per_class[per_class["degradation_ratio"] == 0].set_index("label")
    worst = per_class.sort_values("degradation_ratio").groupby("label").tail(1).set_index("label")
    labels = sorted(set(clean.index).intersection(worst.index))
    rows = []
    for label in labels:
        clean_acc = float(clean.loc[label, "head_accuracy"])
        worst_acc = float(worst.loc[label, "head_accuracy"])
        rows.append(
            {
                "label": label,
                "clean_head_accuracy": clean_acc,
                "worst_condition": worst.loc[label, "condition"],
                "worst_head_accuracy": worst_acc,
                "drop": clean_acc - worst_acc,
            }
        )
    return pd.DataFrame(rows).sort_values(["worst_head_accuracy", "drop"], ascending=[False, True])


def top_attractors(df: pd.DataFrame, prediction_column: str = "head_guess") -> pd.DataFrame:
    required = {"label", prediction_column}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    wrong = df.dropna(subset=[prediction_column])
    wrong = wrong[wrong["label"] != wrong[prediction_column]]
    if wrong.empty:
        return pd.DataFrame()

    group_columns = [prediction_column]
    if "condition" in wrong.columns:
        group_columns = ["condition", prediction_column]

    return (
        wrong.groupby(group_columns)
        .size()
        .reset_index(name="wrong_predictions")
        .sort_values("wrong_predictions", ascending=False)
        .head(12)
    )


def render_alignment_report(df: pd.DataFrame) -> None:
    if "condition" not in df.columns:
        return

    with st.expander("YOLO + JEPA Alignment Report", expanded=True):
        st.markdown(
            """
            **Where** comes from the YOLO boxes. **What** comes from frozen I-JEPA crop embeddings,
            tested with prototype matching and the tiny logistic-regression head. The table shows
            how stable that object identity remains as the crop is degraded.
            """
        )

        condition_metrics = accuracy_by_condition(df)
        if not condition_metrics.empty:
            display_metrics = condition_metrics.copy()
            for column in [
                "head_accuracy",
                "prototype_accuracy",
                "avg_head_confidence",
                "avg_prototype_similarity",
            ]:
                if column in display_metrics.columns:
                    display_metrics[column] = display_metrics[column].map(format_percent)
            st.write("Condition stability")
            st.dataframe(display_metrics, width="stretch", hide_index=True)

        stability = class_stability(df)
        if not stability.empty:
            strong = stability.sort_values(["worst_head_accuracy", "drop"], ascending=[False, True]).head(8).copy()
            weak = stability.sort_values(["worst_head_accuracy", "drop"], ascending=[True, False]).head(8).copy()
            for table in [strong, weak]:
                for column in ["clean_head_accuracy", "worst_head_accuracy", "drop"]:
                    table[column] = table[column].map(format_percent)
            left, right = st.columns(2)
            with left:
                st.write("Strong representation under degradation")
                st.dataframe(strong, width="stretch", hide_index=True)
            with right:
                st.write("Weak / detail-heavy representation")
                st.dataframe(weak, width="stretch", hide_index=True)

        attractors = top_attractors(df, "head_guess")
        if not attractors.empty:
            st.write("Head confusion attractors")
            st.dataframe(attractors, width="stretch", hide_index=True)

        if "prototype_guess" in df.columns:
            prototype_attractors = top_attractors(df, "prototype_guess")
            if not prototype_attractors.empty:
                st.write("Prototype confusion attractors")
                st.dataframe(prototype_attractors, width="stretch", hide_index=True)


def build_projection_figure(df: pd.DataFrame, show_density: bool, color_by: str):
    hover_data = [
        column
        for column in [
            "condition",
            "file_name",
            "object",
            "yolo_label",
            "prototype_guess",
            "prototype_agreement",
            "head_guess",
            "head_agreement",
            "degradation_mode",
            "degradation_ratio",
            "degradation_scope",
        ]
        if column in df.columns
    ]
    color_column = color_by if color_by in df.columns else "label"
    if show_density:
        fig = px.density_contour(
            df,
            x="x",
            y="y",
            color="condition" if "condition" in df.columns else "label",
            title="I-JEPA embedding density and object points",
            height=760,
        )
        scatter = px.scatter(
            df,
            x="x",
            y="y",
            color=color_column,
            symbol="condition" if "condition" in df.columns else None,
            hover_data=hover_data,
        )
        for trace in scatter.data:
            fig.add_trace(trace)
        return fig
    return px.scatter(
        df,
        x="x",
        y="y",
        color=color_column,
        symbol="condition" if "condition" in df.columns else None,
        hover_data=hover_data,
        title="I-JEPA object-crop embedding projection",
        height=760,
    )


def add_centroid_arrows(fig, df: pd.DataFrame) -> None:
    conditions = list(df["condition"].dropna().unique())
    baseline_name = "baseline" if "baseline" in conditions else "clean" if "clean" in conditions else None
    if baseline_name is None:
        return
    centroids = df.groupby(["condition", "label"])[["x", "y"]].mean().reset_index()
    baseline = centroids[centroids["condition"] == baseline_name].set_index("label")
    compare_conditions = [condition for condition in conditions if condition != baseline_name]
    for condition in compare_conditions:
        compare = centroids[centroids["condition"] == condition].set_index("label")
        for label in sorted(set(baseline.index).intersection(compare.index)):
            start = baseline.loc[label]
            end = compare.loc[label]
            fig.add_annotation(
                x=float(end["x"]),
                y=float(end["y"]),
                ax=float(start["x"]),
                ay=float(start["y"]),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                opacity=0.45,
            )


st.set_page_config(page_title="Embedding Space", layout="wide")

st.title("Embedding Space")
st.caption("Visualize 2D projections of frozen I-JEPA object-crop embeddings.")

default_path = "outputs/embedding_projection.csv"
projection_file = st.text_input("Projection CSV", value=default_path)

if projection_file:
    path = Path(projection_file)
    if not path.exists():
        st.warning(f"File not found: {path}")
        st.stop()

    df = pd.read_csv(path)
    required = {"x", "y", "label", "file_name", "object"}
    missing = required.difference(df.columns)
    if missing:
        st.error(f"Projection CSV is missing columns: {sorted(missing)}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Objects", len(df))
    col2.metric("Classes", df["label"].nunique())
    col3.metric("Source", path.name)

    if "condition" in df.columns:
        st.caption("Overlay: marker shape shows the crop/image condition, for example clean vs pixel_75.")

    render_alignment_report(df)

    selected_labels = st.multiselect(
        "Labels",
        options=sorted(df["label"].unique()),
        default=sorted(df["label"].unique()),
    )
    filtered = df[df["label"].isin(selected_labels)]

    color_options = [
        column
        for column in [
            "label",
            "condition",
            "head_guess",
            "head_agreement",
            "prototype_guess",
            "prototype_agreement",
        ]
        if column in filtered.columns
    ]
    controls = st.columns(4)
    color_by = controls[0].selectbox("Color by", options=color_options, index=0)
    if "head_agreement" in filtered.columns or "prototype_agreement" in filtered.columns:
        mistake_filter = controls[1].selectbox(
            "Mistakes",
            options=["all", "head mistakes", "prototype mistakes"],
            index=0,
        )
        if mistake_filter == "head mistakes" and "head_agreement" in filtered.columns:
            filtered = filtered[filtered["head_agreement"] == False]  # noqa: E712
        if mistake_filter == "prototype mistakes" and "prototype_agreement" in filtered.columns:
            filtered = filtered[filtered["prototype_agreement"] == False]  # noqa: E712
    show_density = controls[2].checkbox("Density contours", value="condition" in filtered.columns)
    show_centroids = controls[3].checkbox("Centroid arrows", value="condition" in filtered.columns)

    if filtered.empty:
        st.warning("No points left after filtering.")
        st.stop()

    if "condition" in filtered.columns and "head_agreement" in filtered.columns:
        metrics = filtered.groupby("condition").agg(objects=("label", "size")).reset_index()
        head_accuracy = (
            filtered.groupby("condition")["head_agreement"]
            .apply(lambda values: values.dropna().astype(bool).mean())
            .reset_index(name="head_accuracy")
        )
        metrics = metrics.merge(head_accuracy, on="condition", how="left")
        if "prototype_agreement" in filtered.columns:
            prototype_accuracy = (
                filtered.groupby("condition")["prototype_agreement"]
                .apply(lambda values: values.dropna().astype(bool).mean())
                .reset_index(name="prototype_accuracy")
            )
            metrics = metrics.merge(prototype_accuracy, on="condition", how="left")
        st.dataframe(metrics, width="stretch", hide_index=True)

    fig = build_projection_figure(filtered, show_density=show_density, color_by=color_by)
    fig.update_traces(
        marker={"size": 10, "opacity": 0.85},
        selector={"mode": "markers"},
    )
    if show_centroids and "condition" in filtered.columns:
        add_centroid_arrows(fig, filtered)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Class Counts")
    st.dataframe(
        filtered["label"].value_counts().rename_axis("label").reset_index(name="objects"),
        width="stretch",
        hide_index=True,
    )
