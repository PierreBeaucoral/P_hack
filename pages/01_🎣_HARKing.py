import time, itertools
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from utils.data import simulate_data, read_csv_safely
from utils.harking import run_harking, TRANSFORMS, bh_fdr
import statsmodels.api as sm

st.title("🎣 HARKing")
st.sidebar.header("Data")
mode = st.sidebar.radio("Choose data", ["Simulate", "Upload CSV"], horizontal=True, key="hark_mode")
if mode == "Simulate":
    n = st.sidebar.slider("Rows", 200, 5000, 800, step=100, key="hark_rows")
    k = st.sidebar.slider("Readable variables", 3, 20, 8, step=1, key="hark_k")
    df = simulate_data(n=n, k=k, with_time=True, readable_names=True)
else:
    up = st.sidebar.file_uploader("CSV file", type=["csv"], key="hark_file")
    if up is None: st.stop()
    try:
        df = read_csv_safely(up)
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); st.stop()

max_rows = st.sidebar.slider("Row cap (sampling)", 1_000, 100_000, 10_000, step=1_000, key="hark_cap")
if len(df) > max_rows:
    df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    st.sidebar.caption(f"Sampled to {max_rows} rows for speed.")

with st.expander("Data preview", expanded=False):
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(num_cols) < 2: st.error("Need at least 2 numeric columns."); st.stop()

# Controls
st.subheader("Search space")
default_x_blacklist = {"target_trendy","time","month"}
Y_pool = st.multiselect("Eligible Y variables", num_cols, default=num_cols, key="y_pool")
X_pool = st.multiselect("Eligible X variables", num_cols,
                        default=[c for c in num_cols if c not in default_x_blacklist], key="x_pool")
if not Y_pool or not X_pool: st.warning("Select at least one variable in each pool."); st.stop()

alpha   = st.slider("Significance threshold α", 0.001, 0.10, 0.05, step=0.001, key="alpha")
max_lag = st.slider("Max lag to try", 0, 12, 2, step=1, key="max_lag")
min_r2  = st.slider("Minimum R² (ignored if R² is NaN)", 0.00, 0.99, 0.30, 0.01, key="min_r2")
do_bins = st.checkbox("Try binning (quartiles/deciles)", True, key="do_bins")

all_tf = list(TRANSFORMS.keys())
tf_choices = st.multiselect("Transforms to consider", all_tf,
                            default=[t for t in ["identity","zscore","log1p|signed","diff","detrend_linear"] if t in all_tf],
                            key="tf_choices")
max_results = st.slider("Keep top K results", 5, 100, 20, step=5, key="max_results")
good_needed = st.slider("Early stop after K ‘good’ hits (p<0.01 & R²≥min)", 0, 200, 20, step=5, key="good_needed")
max_models  = st.slider("Hard cap on models", 1_000, 300_000, 30_000, step=1_000, key="max_models")

# Estimate workload
n_filters = 1 + (min(3, len(X_pool)) * 2 * 4 if do_bins else 0)  # rough: 4 slices per var for q/deciles
n_lags = 2*max_lag + 1
n_tx = max(len(tf_choices), 1)
est_total = len(Y_pool) * len(X_pool) * n_tx * n_tx * n_lags * n_filters
st.info(f"Will try approx **{est_total:,}** specs (Y={len(Y_pool)}, X={len(X_pool)}, "
        f"trans={n_tx}×{n_tx}, lags={n_lags}, filters~{n_filters}).")

# Run
run_button = st.button("Run HARKing search", type="primary")
if not run_button: st.stop()

prog = st.progress(0); status = st.empty()
st.session_state["hark_counter"] = 0
with st.spinner("Exploring specs…"):
    resdf = run_harking(df, Y_pool, X_pool, tf_choices, max_lag, do_bins, min_r2)
# progress display (best-effort from cached hook)
status.text(f"Estimated tested specs: ~{st.session_state.get('hark_counter',0):,}")
prog.progress(100)

if resdf.empty:
    st.warning("No models met the criteria. Lower R², change pools, or widen transforms/lags/filters.")
    st.stop()

resdf = resdf.sort_values("p").reset_index(drop=True)
resdf["rank"] = np.arange(1, len(resdf)+1)
resdf["p_FDR<=α"] = bh_fdr(resdf["p"].values, alpha=alpha)

top = resdf.head(max_results)
st.subheader("Top ‘discoveries’")
st.dataframe(top[["rank","x","y","tfx","tfy","lag","filter","p","R2","n","b1","p_FDR<=α"]],
             use_container_width=True, hide_index=True)

best = top.iloc[0]
def plot_spec(row):
    x = df[row.x].values; y = df[row.y].values
    from utils.harking import TRANSFORMS, align
    x = TRANSFORMS[row.tfx](x) if row.tfx in TRANSFORMS else x
    y = TRANSFORMS[row.tfy](y) if row.tfy in TRANSFORMS else y
    if row.lag > 0: x, y = x[:-int(row.lag)], y[int(row.lag):]
    elif row.lag < 0: x, y = x[-int(row.lag):], y[:int(row.lag)]
    x, y = align(x, y)
    X = sm.add_constant(x, has_constant="add"); mod = sm.OLS(y, X).fit()
    r2 = mod.rsquared; yhat = mod.predict(X)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data", opacity=0.65))
    fig.add_trace(go.Scatter(x=x, y=yhat, mode="lines", name="OLS fit"))
    r2_text = f"{r2:.2f}" if np.isfinite(r2) else "NaN"
    p_text  = f"{mod.pvalues[1]:.3g}" if len(mod.pvalues)>1 and np.isfinite(mod.pvalues[1]) else "NaN"
    fig.update_layout(
        xaxis_title=f"{row.x} [{row.tfx}]",
        yaxis_title=f"{row.y} [{row.tfy}]",
        title=f"Best spec: p={p_text}, R²={r2_text}, n={int(mod.nobs)}, lag={int(row.lag)}",
        height=460, margin=dict(l=10,r=10,t=60,b=10)
    )
    return fig

st.plotly_chart(plot_spec(best), use_container_width=True)
st.download_button("Download top-K (CSV)", data=top.to_csv(index=False).encode("utf-8"),
                   file_name="harking_topK.csv", mime="text/csv")
