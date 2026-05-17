"""Streamlit UI for Neural Geometry Lab.

Run:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from nglab.core import extract_activations, load_lm
from nglab.datasets import make_addition_dataset, make_month_dataset, make_number_dataset, make_weekday_dataset
from nglab.geometry import circular_concept_metrics, fit_fourier_probes
from nglab.plotting import plot_concept_circle, plot_fourier_scores, plot_predicted_mod_circle

st.set_page_config(page_title="Neural Geometry Lab", layout="wide")
st.title("Neural Geometry Lab")
st.caption("Activation manifolds, cyclic concepts and Fourier probes for Hugging Face causal language models.")

with st.sidebar:
    st.header("Model")
    model_name = st.text_input("HF model id", value="gpt2")
    layer = st.text_input("Layer index", value="-1")
    batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=8)
    device = st.selectbox("Device", ["auto", "cuda", "mps", "cpu"], index=0)
    dtype = st.selectbox("dtype", ["auto", "fp16", "bf16", "fp32"], index=0)
    device_map = st.selectbox("device_map", ["", "auto"], index=0)
    trust_remote_code = st.checkbox("trust_remote_code", value=False)

    st.header("Experiment")
    task = st.selectbox("Task", ["weekdays", "months", "numbers", "addition"])
    contexts = st.number_input("Prompt templates to use (0 = all)", min_value=0, max_value=10, value=3)
    periods_text = st.text_input("Fourier periods", value="2,5,10,20,50,100")
    run = st.button("Run experiment", type="primary")


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def limit_contexts(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or "context" not in df.columns:
        return df
    keep = list(dict.fromkeys(df["context"].tolist()))[:n]
    return df[df["context"].isin(keep)].reset_index(drop=True)

if run:
    try:
        layers = parse_int_list(layer)
        periods = parse_int_list(periods_text)
        with st.spinner("Loading model and extracting activations..."):
            loaded = load_lm(
                model_name,
                device=device,
                dtype=dtype,
                device_map=(device_map or None),
                trust_remote_code=trust_remote_code,
            )
            if task == "weekdays":
                df = limit_contexts(make_weekday_dataset(), int(contexts))
                acts = extract_activations(loaded, df, layers=layers, batch_size=int(batch_size), extraction="target_or_last")
            elif task == "months":
                df = limit_contexts(make_month_dataset(), int(contexts))
                acts = extract_activations(loaded, df, layers=layers, batch_size=int(batch_size), extraction="target_or_last")
            elif task == "numbers":
                df = limit_contexts(make_number_dataset(0, 99), int(contexts))
                acts = extract_activations(loaded, df, layers=layers, batch_size=int(batch_size), extraction="target_or_last")
            else:
                df = limit_contexts(make_addition_dataset(20, 20), int(contexts))
                acts = extract_activations(loaded, df, layers=layers, batch_size=int(batch_size), extraction="last")

        st.subheader("Prompt sample")
        st.dataframe(df.head(20))

        tmp = Path(tempfile.mkdtemp(prefix="nglab_streamlit_"))
        for lyr, x in acts.items():
            st.markdown(f"## Layer `{lyr}`")
            if task in {"weekdays", "months"}:
                period = 7 if task == "weekdays" else 12
                metrics = circular_concept_metrics(x, df["value"].to_numpy(), period=period)
                st.json(metrics)
                img = plot_concept_circle(
                    x, df["value"].to_numpy(), df["label"].tolist(),
                    period=period,
                    title=f"{task} concept circle — layer {lyr}",
                    outpath=tmp / f"{task}_layer_{lyr}.png",
                )
                st.image(str(img))
            else:
                y_col = "value" if task == "numbers" else "sum"
                scores, preds = fit_fourier_probes(x, df[y_col].to_numpy(), periods=periods)
                st.dataframe(scores)
                img = plot_fourier_scores(scores, title=f"{task} Fourier probes — layer {lyr}", outpath=tmp / f"scores_{task}_{lyr}.png")
                st.image(str(img))
                best_period = int(scores.sort_values("r2_mean", ascending=False).iloc[0]["period"])
                img2 = plot_predicted_mod_circle(
                    preds[best_period], df[y_col].to_numpy(), period=best_period,
                    title=f"Best predicted mod circle p={best_period}, layer {lyr}",
                    outpath=tmp / f"mod_{best_period}_{lyr}.png",
                    annotate=False,
                )
                st.image(str(img2))
    except Exception as e:
        st.exception(e)
else:
    st.info("Choose a task and click Run experiment. Start with `gpt2` or `Qwen/Qwen2.5-0.5B` for quick tests; use larger models for stronger signals.")
