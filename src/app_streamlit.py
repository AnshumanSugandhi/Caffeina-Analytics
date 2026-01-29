from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure the project root and src/ are on sys.path so absolute imports work.
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
for p in (ROOT_DIR, BASE_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Prefer absolute imports; fallback to package-relative when installed as a package.
try:
    from pipeline import PipelineConfig, run_pipeline
except ImportError:  # pragma: no cover
    from .pipeline import PipelineConfig, run_pipeline


st.set_page_config(page_title="California Coffee Market Expansion", layout="wide")
st.title("California Coffee Market Expansion")
st.caption("Location ranking + latte price prediction (per PDR).")

st.sidebar.header("Inputs")
demo = st.sidebar.toggle("Use demo dataset", value=True)
uploaded = st.sidebar.file_uploader("Or upload your CSV", type=["csv"], disabled=demo)

st.sidebar.header("Parameters")
top_n = st.sidebar.slider("Top N", min_value=3, max_value=20, value=5, step=1)
k = st.sidebar.slider("Clusters (KMeans)", min_value=2, max_value=12, value=5, step=1)

artifacts_dir = Path("artifacts")
cfg = PipelineConfig(artifacts_dir=artifacts_dir, cluster_k=int(k), top_n=int(top_n))

tmp_input = None
if (not demo) and uploaded is not None:
    tmp_dir = Path("data")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_input = tmp_dir / "uploaded.csv"
    tmp_input.write_bytes(uploaded.getvalue())

run = st.button("Run analysis", type="primary")
if run:
    res = run_pipeline(input_csv=tmp_input, demo=demo, cfg=cfg)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Top ranked locations")
        st.dataframe(res["top"], use_container_width=True)
    with c2:
        st.subheader("Model quality (if label present)")
        if res["latte_mae"] is None:
            st.info("No `latte_price_12oz` column found; price model not trained.")
        else:
            st.metric("Latte model MAE", f"${res['latte_mae']:.2f}")

    st.subheader("Full ranked table")
    st.dataframe(res["ranked"], use_container_width=True, height=520)

    st.download_button(
        "Download ranked CSV",
        data=res["ranked"].to_csv(index=False).encode("utf-8"),
        file_name="ranked_locations.csv",
        mime="text/csv",
    )

