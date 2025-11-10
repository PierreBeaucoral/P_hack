# Multivariate HARKing — OLS with a main regressor + controls
# Tries many control sets, lags & transforms (main X / Y), ranks by p(main X), filters by model R².
# Includes guards: finite R² only, min obs, zero-variance checks, cap on combinations, early-stop.
# Requires: streamlit, pandas, numpy, plotly, statsmodels, scipy

import itertools, math, time, random
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import probplot

# ---------- Page ----------
st.set_page_config(page_title="🎣 Multivariate HARKing (Main X + Controls)", page_icon="🧪", layout="wide")
st.title("🎣 Multivariate HARKing — Main X + Controls")

# ---------- Helpers ----------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

def is_num(s):
    return pd.api.types.is_numeric_dtype(s)

def align_cols(df, cols):
    """Drop rows with NaNs in any of `cols` and return cleaned df."""
    return df.dropna(subset=cols)

def detrend_linear(s):
    s = np.asarray(s, float)
    idx = np.arange(len(s))
    ok = np.isfinite(s)
    if ok.sum() < 3:
        return s
    b = np.polyfit(idx[ok], s[ok], 1)
    trend = b[0]*idx + b[1]
    return s - trend

TRANSFORMS_Y = {
    "identity":     lambda s: s,
    "zscore":       lambda s: (np.asarray(s, float) - np.nanmean(s)) / (np.nanstd(s) + 1e-12),
    "log1p|signed": lambda s: np.sign(s) * np.log1p(np.abs(s)),
    "diff":         lambda s: pd.Series(s).diff().values,
    "pct_change":   lambda s: pd.Series(s).pct_change().replace([np.inf, -np.inf], np.nan).values,
    "rollmean_3":   lambda s: pd.Series(s).rolling(3, min_periods=1).mean().values,
    "rollmean_6":   lambda s: pd.Series(s).rolling(6, min_periods=1).mean().values,
    "sqrt|signed":  lambda s: np.sign(s) * np.sqrt(np.abs(s)),
    "square":       lambda s: np.square(s),
    "rank":         lambda s: pd.Series(s).rank(pct=True).values,
    "detrend_linear": detrend_linear,
}

# For controls we keep it simple & fast: zscore/identity only.
TRANSFORMS_CTRL = {
    "identity": lambda s: s,
    "zscore":   lambda s: (np.asarray(s, float) - np.nanmean(s)) / (np.nanstd(s) + 1e-12),
}

def add_lag(arr, lag):
    if lag == 0: 
        return arr
    if lag > 0:
        return np.asarray(arr[:-lag])
    else:
        L = -lag
        return np.asarray(arr[L:])

def finite_var(a):
    return np.isfinite(a).sum() >= 3 and np.nanstd(a) > 0

def regress_mainX(y, x_main, X_ctrl):
    """OLS: y ~ const + x_main + X_ctrl ; return model, p(main), R², n, coef(main)."""
    # Build design
    if X_ctrl is None or X_ctrl.size == 0:
        X = sm.add_constant(x_main, has_constant="add")
        names = ["const", "main"]
    else:
        X = np.column_stack([np.ones_like(x_main), x_main, X_ctrl])
        names = ["const", "main"] + [f"c{i+1}" for i in range(X_ctrl.shape[1])]
    try:
        model = sm.OLS(y, X, missing='drop').fit()
        # main regressor is column index 1
        p_main = float(model.pvalues[1]) if len(model.params) > 1 else np.nan
        r2 = float(model.rsquared)
        n = int(model.nobs)
        b_main = float(model.params[1]) if len(model.params) > 1 else np.nan
        if not np.isfinite(r2):
            return None, np.nan, np.nan, 0, np.nan
        return model, p_main, r2, n, b_main
    except Exception:
        return None, np.nan, np.nan, 0, np.nan

def fwll_partial_plot(y, x_main, X_ctrl):
    """Frisch–Waugh–Lovell partial regression: residualize y and x_main on controls, plot residuals."""
    if X_ctrl is None or X_ctrl.size == 0:
        # Nothing to partial out → just simple scatter
        return x_main, y, None
    Xc = sm.add_constant(X_ctrl, has_constant="add")
    try:
        # residualize y
        yhat_c = sm.OLS(y, Xc).fit().predict(Xc)
        ry = y - yhat_c
        # residualize main x
        xhat_c = sm.OLS(x_main, Xc).fit().predict(Xc)
        rx = x_main - xhat_c
        return rx, ry, Xc
    except Exception:
        return x_main, y, None

def sample_control_sets(candidates, min_k, max_k, max_sets, mandatory=set()):
    """
    Create a list of unique control sets (tuples). Respects mandatory ⊆ candidates.
    Randomly samples if total explodes.
    """
    cands = [c for c in candidates if c not in mandatory]
    pools = []
    for k in range(min_k, max_k+1):
        pools.extend(list(itertools.combinations(cands, k)))
    # Attach mandatory to each
    pools = [tuple(sorted(list(mandatory) + list(p)))) for p in pools] if mandatory else pools
    pools = list(dict.fromkeys(pools))  # unique
    if len(pools) > max_sets:
        rng.shuffle(pools)
        pools = pools[:max_sets]
    return pools

# ---------- Data intake ----------
st.sidebar.header("Data (Multivariate HARKing)")
mode = st.sidebar.radio("Choose data source", ["Reuse global df if provided", "Upload CSV"], key="mx_data_mode")

df = None
if "df" in st.session_state and isinstance(st.session_state["df"], pd.DataFrame) and mode == "Reuse global df if provided":
    df = st.session_state["df"].copy()
else:
    up = st.sidebar.file_uploader("CSV file", type=["csv"], key="mx_up")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

if df is None:
    st.info("No data yet. Either upload a CSV here **or** navigate to another page that sets `st.session_state['df']`.")
    st.stop()

num_cols = [c for c in df.columns if is_num(df[c])]
if len(num_cols) < 3:
    st.error("Need at least three numeric columns (Y, main X, and ≥1 potential control).")
    st.stop()

with st.expander("Preview", expanded=False):
    st.dataframe(df.head(12), use_container_width=True)

# ---------- Controls ----------
st.markdown("### Search configuration")

cY, cX, cC = st.columns([1,1,2])
y_col = cY.selectbox("Outcome (Y)", num_cols, index=0, key="mx_y")
x_main = cX.selectbox("Main regressor (X_main)", [c for c in num_cols if c != y_col], index=0, key="mx_main")

control_candidates = [c for c in num_cols if c not in {y_col, x_main}]
ctrl_default = control_candidates[:6]  # limit default
ctrl_pool = cC.multiselect("Candidate controls (pool)", control_candidates, default=ctrl_default, key="mx_ctrl_pool")

# Transform & lag options
cT1, cT2, cL = st.columns([1,1,1])
tY_choices = list(TRANSFORMS_Y.keys())
tX_choices = [k for k in TRANSFORMS_Y.keys()]  # allow same transforms for main X
tY = cT1.multiselect("Y transforms", tY_choices, default=["identity","zscore","detrend_linear"], key="mx_tY")
tX = cT2.multiselect("X_main transforms", tX_choices, default=["identity","zscore","detrend_linear"], key="mx_tX")
max_lag = cL.slider("Max lag to try (on X_main only)", 0, 12, 2, step=1, key="mx_lag")

# Control set size & sampling caps
cK1, cK2, cK3 = st.columns([1,1,1])
min_ctrl = cK1.slider("Min #controls", 0, min(5, len(ctrl_pool)), 0, step=1, key="mx_minC")
max_ctrl = cK2.slider("Max #controls", 0, min(8, len(ctrl_pool)), min(3, len(ctrl_pool)), step=1, key="mx_maxC")
max_ctrl_sets = cK3.slider("Max control sets to try", 10, 5000, 500, step=10, key="mx_maxsets")

# Model filtering & stopping
cF1, cF2, cF3 = st.columns([1,1,1])
alpha = cF1.slider("Significance threshold α (for display)", 0.001, 0.10, 0.05, step=0.001, key="mx_alpha")
min_r2 = cF2.slider("Minimum model R²", 0.00, 0.99, 0.30, 0.01, key="mx_minr2")
good_stop = cF3.slider("Early stop after K good hits (p<0.01 & R²≥min)", 0, 100, 25, step=5, key="mx_goodstop")

max_models = st.slider("Hard cap on models", 1000, 300000, 40000, step=1000, key="mx_cap")
top_k = st.slider("Keep top K results", 5, 200, 30, step=5, key="mx_topK")

# Which transform on controls (simple/fast)
ctrl_transform = st.selectbox("Transform for controls (applied to each control)", ["identity","zscore"], index=1, key="mx_ctrl_tx")

st.markdown("---")

# ---------- Build candidate sets ----------
if max_ctrl < min_ctrl:
    st.warning("Max #controls must be ≥ Min #controls.")
    st.stop()

# Enumerate (or sample) control sets
control_sets = sample_control_sets(ctrl_pool, min_ctrl, max_ctrl, max_ctrl_sets)

# Estimate total workload
nY = len(tY)
nX = len(tX)
nL = 2*max_lag + 1
est_total = nY * nX * nL * max(1, len(control_sets))
st.info(f"About to try ~ **{est_total:,}** models "
        f"(Y_transforms={nY}, X_main_transforms={nX}, lags={nL}, control_sets≈{len(control_sets)}).")

# ---------- Run search ----------
results = []
prog = st.progress(0)
status = st.empty()
loop_done = 0
good_hits = 0
last_t = time.time()

def apply_transform(arr, name, TRANS):
    return TRANS[name](arr) if name in TRANS else arr

try:
    with st.spinner("🌪️ Exploring multivariate specs…"):
        total = max(1, est_total)
        # Build base arrays
        y_raw = df[y_col].values
        x_raw = df[x_main].values

        for ty in tY:
            y_tf = apply_transform(y_raw, ty, TRANSFORMS_Y)
            for tx in tX:
                x_tf0 = apply_transform(x_raw, tx, TRANSFORMS_Y)  # reuse pool for X_main
                for lag in range(-max_lag, max_lag+1):
                    # lag on main X only
                    if lag != 0:
                        if lag > 0:
                            y_use = y_tf[lag:]
                            x_main_use = x_tf0[:-lag]
                        else:
                            L = -lag
                            y_use = y_tf[:-L]
                            x_main_use = x_tf0[L:]
                    else:
                        y_use = y_tf
                        x_main_use = x_tf0

                    # sanity checks
                    if not finite_var(y_use) or not finite_var(x_main_use):
                        loop_done += len(control_sets)
                        if time.time() - last_t > 0.05:
                            prog.progress(min(1.0, loop_done / total))
                            status.text(f"Tried ~{loop_done:,}/{est_total:,} specs …")
                            last_t = time.time()
                        continue

                    for Cset in control_sets:
                        cols = [y_col, x_main] + list(Cset)
                        D = df.loc[:, cols].copy()

                        # align to the indices that survived lag
                        # simplest: rebuild slices using the current y/x lengths
                        n_eff = min(len(y_use), len(x_main_use))
                        if n_eff < 3:
                            loop_done += 1
                            continue
                        # take last n_eff rows to align
                        y_arr = np.asarray(y_use[-n_eff:], float)
                        x_arr = np.asarray(x_main_use[-n_eff:], float)

                        X_ctrl = None
                        if len(Cset) > 0:
                            ctrl_mat = []
                            for c in Cset:
                                v = D[c].values[-n_eff:]
                                v = TRANSFORMS_CTRL[ctrl_transform](v)
                                ctrl_mat.append(v)
                            X_ctrl = np.column_stack(ctrl_mat)

                            # finite & variance checks on controls
                            if not all(finite_var(col) for col in ctrl_mat):
                                loop_done += 1
                                continue

                        # Run model
                        model, p_main, r2, nobs, b_main = regress_mainX(y_arr, x_arr, X_ctrl)
                        if (not np.isfinite(p_main)) or (not np.isfinite(r2)) or nobs < 3:
                            loop_done += 1
                            continue
                        if r2 < min_r2:
                            loop_done += 1
                            continue

                        results.append({
                            "y": y_col,
                            "x_main": x_main,
                            "lag": lag,
                            "tY": ty,
                            "tX": tx,
                            "controls": ",".join(Cset) if Cset else "(none)",
                            "p_main": p_main,
                            "R2": r2,
                            "n": nobs,
                            "beta_main": b_main
                        })

                        # early stop: good hit => p<0.01 & R²≥min
                        if p_main < 0.01 and r2 >= min_r2:
                            good_hits += 1

                        loop_done += 1
                        # progress
                        if time.time() - last_t > 0.05:
                            prog.progress(min(1.0, loop_done / total))
                            status.text(f"Tried ~{loop_done:,}/{est_total:,} specs …")
                            last_t = time.time()

                        if (good_stop and good_hits >= good_stop) or (loop_done >= max_models):
                            raise SystemExit
except SystemExit:
    pass

# ---------- Results ----------
resdf = pd.DataFrame(results)
if resdf.empty:
    st.warning("No models met the criteria. Try lowering `Minimum model R²`, reducing lag/transform complexity, or widening control sets.")
    st.stop()

resdf.sort_values(["p_main", "R2"], ascending=[True, False], inplace=True)
resdf["rank"] = np.arange(1, len(resdf)+1)

st.subheader("🎣 Top ‘discoveries’ for the MAIN regressor (sorted by p-value of main X)")
st.dataframe(
    resdf.head(top_k)[["rank","y","x_main","tY","tX","lag","controls","p_main","R2","n","beta_main"]],
    use_container_width=True
)

# ---------- Best spec details + Partial regression plot ----------
best = resdf.iloc[0]
st.markdown("### 📌 Best spec — details")

st.write(
    f"- **Y**: `{best.y}` with transform **{best.tY}**"
    f"\n- **Main X**: `{best.x_main}` with transform **{best.tX}**, **lag={int(best.lag)}**"
    f"\n- **Controls**: {best.controls}"
    f"\n- **p(main)**: {best.p_main:.3g}, **R²**: {best.R2:.3f}, **n**: {int(best.n)}, **β(main)**: {best.beta_main:.3g}"
)

# Rebuild arrays for best spec to visualize partial effect
y_raw = df[best.y].values
x_raw = df[best.x_main].values
y_tf = TRANSFORMS_Y[best.tY](y_raw) if best.tY in TRANSFORMS_Y else y_raw
x_tf0 = TRANSFORMS_Y[best.tX](x_raw) if best.tX in TRANSFORMS_Y else x_raw

lag = int(best.lag)
if lag != 0:
    if lag > 0:
        y_use = y_tf[lag:]
        x_use = x_tf0[:-lag]
    else:
        L = -lag
        y_use = y_tf[:-L]
        x_use = x_tf0[L:]
else:
    y_use = y_tf
    x_use = x_tf0

# controls
Cset = [] if best.controls == "(none)" else best.controls.split(",")
X_ctrl = None
if len(Cset) > 0:
    ctrl_mat = []
    for c in Cset:
        v = df[c].values
        # align to n_eff
        n_eff = min(len(y_use), len(x_use), len(v))
        v = v[-n_eff:]
        y_plot = np.asarray(y_use[-n_eff:], float)
        x_plot = np.asarray(x_use[-n_eff:], float)
        # control transform
        # (same choice as in the search; we can't know which was used unless we store it — assume zscore default if chosen)
        # We'll respect current ctrl_transform selection for consistency:
        v = TRANSFORMS_CTRL[ctrl_transform](v)
        ctrl_mat.append(v)
    X_ctrl = np.column_stack(ctrl_mat)
else:
    n_eff = min(len(y_use), len(x_use))
    y_plot = np.asarray(y_use[-n_eff:], float)
    x_plot = np.asarray(x_use[-n_eff:], float)

# ensure y_plot/x_plot exist if controls present
if len(Cset) > 0:
    n_eff = min(len(y_use), len(x_use), *(len(col) for col in ctrl_mat)) if len(ctrl_mat)>0 else min(len(y_use), len(x_use))
    y_plot = np.asarray(y_use[-n_eff:], float)
    x_plot = np.asarray(x_use[-n_eff:], float)
    X_ctrl = np.column_stack([col[-n_eff:] for col in ctrl_mat]) if len(ctrl_mat)>0 else None

rx, ry, Xc = fwll_partial_plot(y_plot, x_plot, X_ctrl)

fig = go.Figure()
fig.add_trace(go.Scatter(x=rx, y=ry, mode="markers", name="Partial residuals", opacity=0.65))
# add partial OLS line
X_line = sm.add_constant(rx, has_constant="add")
try:
    mod_line = sm.OLS(ry, X_line).fit()
    yh = mod_line.predict(X_line)
    fig.add_trace(go.Scatter(x=rx, y=yh, mode="lines", name="Partial OLS fit"))
    title_extra = f" (slope={mod_line.params[1]:.3g})"
except Exception:
    title_extra = ""

fig.update_layout(
    template="plotly_white",
    xaxis_title=f"Residualized {best.x_main} [{best.tX}]",
    yaxis_title=f"Residualized {best.y} [{best.tY}]",
    title=f"Partial regression plot — effect of MAIN regressor after controls{title_extra}"
)
st.plotly_chart(fig, use_container_width=True)

# Download
st.download_button(
    "💾 Download top-K multivariate HARKing (CSV)",
    data=resdf.head(top_k).to_csv(index=False).encode("utf-8"),
    file_name="multivariate_harking_topK.csv",
    mime="text/csv",
    key="mx_dl"
)

with st.expander("ℹ️ Notes & guardrails", expanded=False):
    st.markdown("""
- We only keep models with **finite R²**, **n ≥ 3**, and non-zero variance in Y and X_main (after lag/transform).
- **Controls** use a single transform option (identity/zscore) to keep the search tractable.
- **Lags** apply to the **main regressor only**. (You can extend to lag controls if you wish.)
- The **partial regression plot** shows the marginal relation of X_main with Y after removing linear effects of controls (Frisch–Waugh–Lovell).
- Use **Early stop** and **Max control sets** to keep runtime reasonable.
""")
