import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
from utils.data import simulate_data, read_csv_safely, build_time_candidates

st.title("📊 DataViz Studio")
st.sidebar.header("Data")
mode = st.sidebar.radio("Choose data", ["Simulate", "Upload CSV"], horizontal=True, key="viz_mode")
if mode == "Simulate":
    n = st.sidebar.slider("Rows", 200, 5000, 800, step=100, key="viz_rows")
    k = st.sidebar.slider("Readable variables", 3, 20, 8, step=1, key="viz_k")
    df = simulate_data(n=n, k=k, with_time=True, readable_names=True)
else:
    up = st.sidebar.file_uploader("CSV file", type=["csv"], key="viz_file")
    if up is None: st.stop()
    try:
        df = read_csv_safely(up)
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); st.stop()

st.markdown("### Correlation heatmap")
num_for_corr = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(num_for_corr) >= 2:
    corr = df[num_for_corr].corr()
    figc = px.imshow(corr, text_auto=False, color_continuous_scale="RdBu_r", origin="lower",
                     labels=dict(color="corr")); figc.update_layout(height=500)
    st.plotly_chart(figc, use_container_width=True)
else:
    st.info("Not enough numeric columns for a correlation heatmap.")

st.markdown("### Scatter matrix")
pick_scatter = st.multiselect("Select up to 6 columns", df.columns.tolist(), default=num_for_corr[:4], key="sm_cols")
if pick_scatter:
    pick_scatter = pick_scatter[:6]
    try:
        figsm = px.scatter_matrix(df[pick_scatter].dropna(), dimensions=pick_scatter, opacity=0.6)
        figsm.update_layout(height=600)
        st.plotly_chart(figsm, use_container_width=True)
    except Exception as e:
        st.info(f"Could not draw scatter matrix: {e}")

st.markdown("### Time-series explorer")
df_ts, time_candidates = build_time_candidates(df)
if time_candidates:
    tcol = st.selectbox("Time column", time_candidates, index=0, key="ts_timecol")
    tsd = df_ts.dropna(subset=[tcol]).sort_values(tcol)
    if not num_for_corr:
        st.info("No numeric series to plot.")
    else:
        y1 = st.selectbox("Series 1", num_for_corr, index=0, key="ts_y1")
        y2 = st.selectbox("Series 2 (optional)", ["(none)"] + num_for_corr, index=0, key="ts_y2")
        figts = go.Figure()
        figts.add_trace(go.Scatter(x=tsd[tcol], y=tsd[y1], mode="lines", name=y1))
        if y2 != "(none)":
            figts.add_trace(go.Scatter(x=tsd[tcol], y=tsd[y2], mode="lines", name=y2, yaxis="y2"))
            figts.update_layout(yaxis2=dict(overlaying="y", side="right"))
        figts.update_layout(xaxis_title=str(tcol), yaxis_title=y1, height=450)
        st.plotly_chart(figts, use_container_width=True)
else:
    st.info("No time-like column found. Add a datetime column or an integer 'year' column (e.g., 1990..2025).")
