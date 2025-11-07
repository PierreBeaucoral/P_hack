# app.py — HARKing Generator + Regression & Viz Lab
# Guards: skip specs with zero variance / too few obs; keep R² even if NaN (per request).

import itertools, time, random
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import probplot
import statsmodels.api as sm

# ----------------------- Page -----------------------
st.set_page_config(page_title="HARKing + Regression Lab", layout="wide")
st.title("🧪 HARKing Generator + 📊 Regression & Viz Lab")

# ----------------------- About / Help -----------------------
with st.expander("ℹ️ About this app (short version)", expanded=False):
    st.markdown("""
**What is this?**  
A playful but serious web app where you can:
1) **🎣 HARKing** — brute-force many OLS specs *Y ~ X* with transforms, lags, and bin-filters; enforce a **minimum R²**; rank by p-value.
2) **📐 Regression Lab** — run regular OLS with dummies/standardization, view coefficients with CIs, VIF, and diagnostics.
3) **📊 DataViz** — quick correlation heatmaps, scatter matrices, and a time-series explorer.

**Note on R²:** specs with zero variance or too few observations are skipped. R² can be NaN (kept, per your request).
    """)

# ----------------------- Utils -----------------------

RANDOM_SEED = 42

# ---- Helpers to detect/convert time columns (supports integer "year") ----
def build_time_candidates(df: pd.DataFrame):
    """
    Return (df_with_converted_cols, time_col_names).
    - Keeps native datetime64 columns.
    - Converts columns whose name contains 'date'/'time' or 'year' and look like years (1800–2100)
      into datetime via YYYY-01-01, exposing them as <col>__as_date.
    - Also tries general to_datetime parsing for string-like date columns.
    """
    df = df.copy()
    time_cols = []

    # keep native datetimes
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            time_cols.append(c)

    # detect/convert year/date-ish columns
    for c in df.columns:
        name = str(c).lower()

        # fast path for explicit 'year' columns
        if "year" in name and (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c])):
            ser_num = pd.to_numeric(df[c], errors="coerce")
            # consider it a year column if ≥80% look like years 1800–2100
            if (ser_num.between(1800, 2100)).mean() >= 0.8:
                cname = f"{c}__as_date"
                # map year -> January 1st of that year
                df[cname] = pd.to_datetime(ser_num.astype("Int64").astype(str), format="%Y", errors="coerce")
                time_cols.append(cname)
                continue

        # generic date/time names: try parsing
        if ("date" in name) or ("time" in name):
            ser_parsed = pd.to_datetime(df[c], errors="coerce")
            if ser_parsed.notna().mean() >= 0.6:  # if most rows parsed, accept
                cname = f"{c}__as_date"
                df[cname] = ser_parsed
                time_cols.append(cname)

    # de-duplicate while preserving order
    seen = set()
    time_cols_unique = []
    for c in time_cols:
        if c not in seen:
            time_cols_unique.append(c)
            seen.add(c)

    return df, time_cols_unique

rng = np.random.default_rng(RANDOM_SEED)

FRIENDLY_VARS = [
    "household_income","inflation_rate","unemployment_rate","policy_rate",
    "industrial_output","consumer_confidence","housing_prices","exports_value",
    "imports_value","net_trade","exchange_rate","energy_price","co2_emissions",
    "tourism_arrivals","public_spending","private_investment","bank_credit",
    "stock_index","wage_index","productivity_index"
]

def pick_friendly_names(k: int):
    base = FRIENDLY_VARS.copy()
    rng.shuffle(base)
    if k <= len(base):
        return base[:k]
    extra, i = [], 1
    while len(base) + len(extra) < k:
        extra.extend([f"{name}_{i}" for name in FRIENDLY_VARS])
        i += 1
    return base + extra[:k]

def simulate_data(n=800, k=8, with_time=True, readable_names=True):
    names = pick_friendly_names(k) if readable_names else [f"X{i+1}" for i in range(k)]
    X = {names[i]: rng.normal(0, 1, n) for i in range(k)}
    if with_time:
        t = pd.date_range("2010-01-01", periods=n, freq="D")
        X["time"] = np.arange(n)
        X["month"] = pd.to_datetime(t).month
        X["date"] = t
    df = pd.DataFrame(X)
    # Outcomes: a null target, a weak structural target, and a trendy target
    df["target_null"]   = rng.normal(0,1,n)
    x1 = df[names[0]].values
    x2 = df[names[1]].values
    df["target_policy"] = 0.3*x1 - 0.2*(x2 > 0).astype(int) + rng.normal(0,1,n)
    df["target_trendy"] = rng.normal(0,1,n).cumsum() + rng.normal(0,5,n)
    return df

def make_bins(x, q=4):
    s = pd.Series(x)
    try:
        return pd.qcut(s, q=q, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=q)

def align(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]

def regress_simple(x, y):
    # guards: enough obs & non-zero variance
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, 0, np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, 0, np.nan
    X = sm.add_constant(x, has_constant="add")
    try:
        model = sm.OLS(y, X, missing='drop').fit()
        pv = model.pvalues[1] if len(model.params) > 1 else np.nan
        r2 = model.rsquared
        n  = int(model.nobs)
        b1 = model.params[1] if len(model.params) > 1 else np.nan
        # (per request) do NOT drop NaN R²
        return pv, r2, n, b1
    except Exception:
        return np.nan, np.nan, 0, np.nan

def bh_fdr(pvals, alpha=0.05):
    ps = np.array(pvals)
    order = np.argsort(ps)
    ranked = ps[order]
    m = float(len(ps))
    thresh = (np.arange(1, len(ps)+1) / m) * alpha
    passed = ranked <= thresh
    if not passed.any():
        return np.full_like(ps, False, dtype=bool)
    kmax = np.max(np.where(passed))
    cutoff = ranked[kmax]
    return ps <= cutoff

def detrend_linear(s):
    s = np.asarray(s, float)
    idx = np.arange(len(s))
    ok = np.isfinite(s)
    if ok.sum() < 3:
        return s
    b = np.polyfit(idx[ok], s[ok], 1)
    trend = b[0]*idx + b[1]
    return s - trend

TRANSFORMS = {
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

HEADLINE_VERBS = ["predicts","explains","drives","is tightly coupled with","foretells","perfectly mirrors"]
SPICE = ["per aggressive model exploration","after extensive post-hoc analysis",
         "using cutting-edge curve fitting","according to exploratory research","following a thorough p-hack"]

def silly_headline(xn, yn, p, r2, lag, tfx, tfy, filt_desc):
    lagbit = f" (lag {lag})" if lag else ""
    fx = "" if tfx == "identity" else f" [{tfx}]"
    fy = "" if tfy == "identity" else f" [{tfy}]"
    filt = f" {filt_desc}" if filt_desc else ""
    return f"{xn}{fx} {random.choice(HEADLINE_VERBS)} {yn}{fy}{lagbit} (p={p:.3g}, R²={r2:.2f}) — {random.choice(SPICE)}{filt}"

def bogus_abstract(xn, yn, p, r2, n, lag, tfx, tfy, filt_desc):
    lagbit = f" with a temporal lag of {lag} period(s)" if lag else ""
    fx = "" if tfx == "identity" else f" after a {tfx} transform"
    fy = "" if tfy == "identity" else f" on a {tfy} scale"
    filt = f" under a data slice ({filt_desc})" if filt_desc else ""
    return (f"We explore associations between {xn}{fx} and {yn}{fy}{lagbit}{filt}. "
            f"(two-sided p={p:.3g}, R²={r2:.2f}, n={n}). This illustrates post-hoc inference risk; "
            f"use pre-registration, multiple-testing correction, and theory-driven analysis.")

# ----------------------- Data intake -----------------------
st.sidebar.header("Data")
st.sidebar.markdown("""
**Quick start**
1. Keep **Simulate** if you’re just exploring.
2. Adjust **Rows** and **Readable variables**.
3. Jump to **🎣 HARKing** and click through defaults.
4. Sanity-check any hit in **📐 Regression Lab**.
""")
mode = st.sidebar.radio("Choose data", ["Simulate", "Upload CSV"], horizontal=True)
if mode == "Simulate":
    n = st.sidebar.slider("Rows", 200, 5000, 800, step=100, key="rows_sim")
    k = st.sidebar.slider("Readable variables", 3, 20, 8, step=1, key="readable_vars")
    df = simulate_data(n=n, k=k, with_time=True, readable_names=True)
else:
    up = st.sidebar.file_uploader("CSV file", type=["csv"])
    if up is None:
        st.stop()
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

# Optional sampling for speed
max_rows = st.sidebar.slider("Row cap for HARKing (sampling)", 1_000, 100_000, 10_000, step=1_000, key="row_cap")
if len(df) > max_rows:
    df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    st.sidebar.caption(f"Sampled to {max_rows} rows for speed.")

with st.expander("Data preview", expanded=False):
    st.dataframe(df.head(12), use_container_width=True)

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(num_cols) < 2:
    st.error("Need at least two numeric columns in the data.")
    st.stop()

# ----------------------- Tabs -----------------------
tab_home, tab_hark, tab_reg, tab_viz = st.tabs(["🏠 Overview", "🎣 HARKing", "📐 Regression Lab", "📊 DataViz Studio"])

# ====================== TAB 0: OVERVIEW ======================
with tab_home:
    st.markdown("## What you can do here")
    st.markdown("""
**This site is both a sandbox and a lab:**
- **Generate findings on demand (HARKing):** Search thousands of toy regressions with different transforms, lags, and data slices to see how easily “significant” results appear.  
- **Validate like a grown-up (Regression Lab):** Re-estimate relations with explicit predictors, dummies, standardization, and full diagnostics.
- **Understand your data (DataViz Studio):** Use heatmaps, scatter matrices, and time-series overlays for quick EDA.

**Why this matters:**  
Modern empirical work can unintentionally “discover” patterns by trying many specifications. This app **shows the mechanics** and helps you **avoid traps** by checking model quality (R² threshold), multiple-testing (FDR column), and residual diagnostics.
""")

    st.markdown("## How to use it (2-minute tour)")
    st.markdown("""
1. **Pick data** in the left sidebar: keep **Simulate** to play, or **Upload CSV** to use your own data.
2. Go to **🎣 HARKing**:
   - Choose **eligible Y** and **eligible X** pools (defaults are good).
   - Keep a mild **Minimum R²** (e.g., 0.30) and small **lag window** at first.
   - Inspect the **Top discoveries** table and the **Best spec plot**.
3. Go to **📐 Regression Lab**:
   - Recreate a promising relation: select an outcome and predictors, use **dummies** and **standardization** where appropriate.
   - Inspect **coefficients + CIs**, **VIF**, and **diagnostics** (residual plots, QQ, leverage).
4. Use **📊 DataViz** to confirm intuition:
   - Check **correlation heatmap** and **scatter matrix**.
   - Explore **time-series** with overlays (secondary Y-axis available).
""")

    st.markdown("## Methodology & Ethics")
    st.markdown("""
- **Multiple testing:** The HARKing table includes a **Benjamini–Hochberg FDR** flag. Expect many “discoveries” to fail this correction. That’s the lesson.
- **R² handling:** Specs with **zero variance** or **too few observations** are skipped. R² may be **NaN** and is shown as-is.
- **Spurious trends:** Use `detrend_linear`, `diff`, or `pct_change` to reduce spurious trend-on-trend links.
- **Good practice:** Prefer **pre-registration**, **out-of-sample checks**, and **theory-driven** specifications.
""")

# ====================== TAB 1: HARKING ======================
with tab_hark:
    st.subheader("HARKing Generator")

    st.sidebar.header("HARKing knobs")
    alpha    = st.sidebar.slider("Significance threshold α", 0.001, 0.10, 0.05, step=0.001, key="alpha")
    max_lag  = st.sidebar.slider("Max lag to try", 0, 12, 2, step=1, key="max_lag")
    min_r2   = st.sidebar.slider("Minimum R²", 0.00, 0.99, 0.30, 0.01,
                                 help="Only keep specs with R² ≥ this value (NaN R² bypasses this check).",
                                 key="min_r2")
    do_bins  = st.sidebar.checkbox("Try binning (quartiles/deciles)", True, key="do_bins")
    do_interactions = st.sidebar.checkbox("Try simple interactions (X1×X2)", False, key="do_interact")
    # Single definition of K slider to avoid duplicate ID
    max_results = st.slider("Keep top K results", 5, 100, 20, step=5, key="hark_max_results")

    # Early-stop controls
    good_needed = st.slider("Early stop after K ‘good’ hits (p<0.01 & R²≥min)", 0, 200, 20, step=5, key="good_needed")
    max_models  = st.slider("Hard cap on models", 1_000, 300_000, 30_000, step=1_000, key="max_models")

    # Pools
    st.markdown("#### Variable pools (for the HARKing search)")
    default_x_blacklist = {"target_trendy","time","month"}
    pool_cols = [c for c in num_cols]
    Y_pool = st.multiselect("Eligible Y variables", pool_cols, default=pool_cols, key="y_pool")
    X_pool = st.multiselect("Eligible X variables", pool_cols,
                            default=[c for c in pool_cols if c not in default_x_blacklist],
                            key="x_pool")
    if not Y_pool or not X_pool:
        st.warning("Select at least one variable in each pool.")
        st.stop()

    # Transforms
    all_tf = list(TRANSFORMS.keys())
    default_tf = [t for t in ["identity","zscore","log1p|signed","diff","detrend_linear"] if t in all_tf]
    trans_choices = st.multiselect("Transforms to consider", all_tf, default=default_tf, key="tf_choices")

    # Working copy + engineered interactions (limited)
    df_work = df.copy()
    eng_cols = []
    if do_interactions and len(X_pool) >= 2:
        pairs = list(itertools.combinations(X_pool[:min(6,len(X_pool))], 2))
        for a, b in pairs:
            name = f"{a}×{b}"
            df_work[name] = df_work[a]*df_work[b]
            eng_cols.append(name)

    Ycands = [c for c in Y_pool]
    Xcands = [c for c in X_pool] + eng_cols

    results = []
    filters, filt_labels = [None], [""]
    if do_bins:
        for col in X_pool[:min(3, len(X_pool))]:
            bins4 = make_bins(df_work[col], q=4)
            bins10 = make_bins(df_work[col], q=10)
            for series, label in [(bins4, f"{col} in quartile q"), (bins10, f"{col} in decile d")]:
                if hasattr(series, "codes"):
                    codes = pd.Series(series.cat.codes).values
                    uniq = np.unique(codes[codes >= 0])[:4]
                    for code in uniq:
                        mask = (codes == code)
                        filters.append(mask)
                        filt_labels.append(label.replace("q", str(code+1)).replace("d", str(code+1)))

    def try_one(xname, yname, lag, tfx, tfy, filt_mask=None, label_filt=""):
        x = df_work[xname].values
        y = df_work[yname].values
        x = TRANSFORMS[tfx](x) if tfx in TRANSFORMS else x
        y = TRANSFORMS[tfy](y) if tfy in TRANSFORMS else y
        if lag > 0:
            x = x[:-lag]; y = y[lag:]
        elif lag < 0:
            x = x[-lag:]; y = y[:lag]
        if filt_mask is not None:
            x = x[filt_mask]; y = y[filt_mask]
        x, y = align(x, y)
        # hard guards before OLS
        if len(x) < 3 or len(y) < 3:
            return None
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return None
        pv, r2, n, b1 = regress_simple(x, y)
        # keep R² even if NaN; still require finite p-value and R² >= threshold if numeric
        if not np.isfinite(pv):
            return None
        if np.isfinite(r2) and (r2 < min_r2):
            return None
        row = {"x": xname, "y": yname, "lag": lag, "tfx": tfx, "tfy": tfy,
               "p": float(pv), "R2": float(r2), "n": int(n), "b1": float(b1),
               "filter": label_filt}
        results.append(row)
        return row

    n_filters = len(filters)
    n_lags = 2*max_lag + 1
    n_tx = max(len(trans_choices), 1)
    est_total = len(Ycands) * max(0, len(Xcands)) * n_tx * n_tx * n_lags * n_filters
    st.info(f"About to try ~ **{est_total:,}** specs (Y={len(Ycands)}, X={len(Xcands)}, "
            f"trans={n_tx}×{n_tx}, lags={n_lags}, filters={n_filters}).")

    prog = st.progress(0)
    status = st.empty()
    model_counter, good_hits = 0, 0
    last_t = time.time()
    stop_now = False

    try:
        with st.spinner("🌪️ Exploring specs…"):
            total_loops = est_total if est_total > 0 else 1
            loop_done = 0
            for yname in Ycands:
                for xname in Xcands:
                    if xname == yname:
                        loop_done += n_tx*n_tx*n_lags*n_filters
                        prog.progress(min(1.0, loop_done / total_loops))
                        continue
                    for lag in range(-max_lag, max_lag+1):
                        for tfx in trans_choices:
                            for tfy in trans_choices:
                                for F, Flab in zip(filters, filt_labels):
                                    row = try_one(xname, yname, lag, tfx, tfy, F, Flab)
                                    model_counter += 1
                                    loop_done += 1
                                    if time.time() - last_t > 0.05:
                                        prog.progress(min(1.0, loop_done / total_loops))
                                        status.text(f"Tried {model_counter:,} / ~{est_total:,} specs …")
                                        last_t = time.time()
                                    if row and (row["p"] < 0.01) and (np.isfinite(row["R2"]) and row["R2"] >= min_r2):
                                        good_hits += 1
                                    if (good_needed and good_hits >= good_needed) or (model_counter >= max_models):
                                        stop_now = True
                                        raise SystemExit
    except SystemExit:
        pass

    resdf = pd.DataFrame(results)
    if resdf.empty:
        st.warning("No models met the criteria. Lower R², change pools, or widen transforms/lags/filters.")
    else:
        resdf.sort_values("p", inplace=True)
        resdf["rank"] = np.arange(1, len(resdf)+1)
        resdf["p_FDR<=α"] = bh_fdr(resdf["p"].values, alpha=alpha)
        top = resdf.head(max_results)

        st.subheader("🎣 Top ‘discoveries’ (lowest p-values first)")
        st.dataframe(top[["rank","x","y","tfx","tfy","lag","filter","p","R2","n","b1","p_FDR<=α"]],
                     use_container_width=True)

        best = top.iloc[0]
        headline = silly_headline(best.x, best.y, best.p, best.R2, int(best.lag), best.tfx, best.tfy, best["filter"])
        abstract = bogus_abstract(best.x, best.y, best.p, best.R2, int(best.n), int(best.lag), best.tfx, best.tfy, best["filter"])

        st.markdown("## 📰 Headline")
        st.write(headline)
        st.markdown("## 🧾 Abstract (auto-drafted)")
        st.write(abstract)

        def plot_spec(bestrow):
            x = df_work[bestrow.x].values
            y = df_work[bestrow.y].values
            x = TRANSFORMS[bestrow.tfx](x) if bestrow.tfx in TRANSFORMS else x
            y = TRANSFORMS[bestrow.tfy](y) if bestrow.tfy in TRANSFORMS else y
            lag = int(bestrow.lag)
            if lag > 0: x, y = x[:-lag], y[lag:]
            elif lag < 0: x, y = x[-lag:], y[:lag]
            x, y = align(x, y)
            if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
                fig = go.Figure()
                fig.update_layout(template="plotly_white", title="Unable to plot: insufficient variance/obs")
                return fig
            X = sm.add_constant(x, has_constant="add")
            mod = sm.OLS(y, X).fit()
            r2 = mod.rsquared  # may be NaN
            yhat = mod.predict(X)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data", opacity=0.65))
            fig.add_trace(go.Scatter(x=x, y=yhat, mode="lines", name="OLS fit"))
            r2_text = f"{r2:.2f}" if np.isfinite(r2) else "NaN"
            p_text  = f"{mod.pvalues[1]:.3g}" if len(mod.pvalues)>1 and np.isfinite(mod.pvalues[1]) else "NaN"
            fig.update_layout(template="plotly_white",
                              xaxis_title=f"{bestrow.x} [{bestrow.tfx}]",
                              yaxis_title=f"{bestrow.y} [{bestrow.tfy}]",
                              title=f"Best spec: p={p_text}, R²={r2_text}, n={int(mod.nobs)}, lag={lag}")
            return fig

        st.markdown("## 📉 Plot of the ‘best’ spec")
        st.plotly_chart(plot_spec(best), use_container_width=True)

        st.download_button("💾 Download top-K table (CSV)",
                           data=top.to_csv(index=False).encode("utf-8"),
                           file_name="harking_topK.csv", mime="text/csv")

        if stop_now:
            st.warning(f"Stopped early after {model_counter:,} specs "
                       f"(good hits ≥ {good_needed} or cap {max_models:,}).")

        with st.expander("Reproducibility blurb"):
            st.markdown("""This app intentionally **does not correct** for the massive multiple-testing in its main table.
The `p_FDR<=α` column shows which hits would survive a Benjamini–Hochberg FDR control at your α. Most will not. 🚨""")

# =================== TAB 2: REGRESSION LAB ===================
with tab_reg:
    st.subheader("OLS Regression (with diagnostics)")

    c1, c2 = st.columns(2)
    all_cols = [c for c in df.columns]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ycol = c1.selectbox("Outcome (y)", num_cols, index=0)
    x_candidates = [c for c in all_cols if c != ycol]
    x_sel = c2.multiselect("Predictors (X)", x_candidates, default=[c for c in num_cols if c != ycol][:2])

    c3, c4, c5 = st.columns(3)
    add_constant  = c3.checkbox("Add intercept", True)
    standardize_X = c4.checkbox("Standardize X", False, help="z-score numeric predictors")
    dummies       = c5.checkbox("One-hot encode categoricals", True)

    if not x_sel:
        st.info("Select at least one predictor.")
    else:
        D = df[[ycol] + x_sel].copy()
        if dummies:
            cat_like = [c for c in x_sel if (D[c].dtype == "object" or pd.api.types.is_categorical_dtype(D[c]))]
            if cat_like:
                D = pd.get_dummies(D, columns=cat_like, drop_first=True)

        y = D[ycol].astype(float)
        X = D.drop(columns=[ycol])

        if standardize_X:
            for c in X.columns:
                if pd.api.types.is_numeric_dtype(X[c]):
                    mu, sd = X[c].mean(), X[c].std(ddof=0)
                    if sd > 0:
                        X[c] = (X[c] - mu) / sd

        if add_constant:
            X = sm.add_constant(X, has_constant="add")

        M = pd.concat([y, X], axis=1).dropna()
        y_al = M[ycol].values
        X_al = M.drop(columns=[ycol]).values
        X_cols = list(M.drop(columns=[ycol]).columns)

        if M.shape[0] < len(X_cols) + 5:
            st.warning("Not enough observations after cleaning for this many predictors.")
        else:
            model = sm.OLS(y_al, X_al).fit()

            st.markdown("### Model summary")
            st.text(model.summary().as_text())

            # Coef table with CI (robust to ndarray/DataFrame conf_int)
            params = pd.Series(model.params).reset_index(drop=True)
            bse    = pd.Series(model.bse).reset_index(drop=True)
            tvals  = pd.Series(model.tvalues).reset_index(drop=True)
            pvals  = pd.Series(model.pvalues).reset_index(drop=True)
            conf_arr = model.conf_int()
            conf_df  = (conf_arr if isinstance(conf_arr, pd.DataFrame)
                        else pd.DataFrame(conf_arr, columns=["ci_low","ci_high"]))
            conf_df = conf_df.reset_index(drop=True)

            coeft = pd.DataFrame({
                "term": X_cols,
                "coef": params,
                "se":   bse,
                "t":    tvals,
                "p":    pvals,
            })
            coeft = pd.concat([coeft, conf_df[["ci_low","ci_high"]]], axis=1)
            st.markdown("### Coefficients")
            st.dataframe(coeft, use_container_width=True)

            # VIF (skip constant)
            st.markdown("### Multicollinearity (VIF)")
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                vif_df = []
                X_noconst = M.drop(columns=[ycol]).copy()
                if "const" in X_noconst.columns:
                    X_noconst = X_noconst.drop(columns=["const"])
                arr = X_noconst.values
                for i, name in enumerate(X_noconst.columns):
                    vif_df.append({"variable": name, "VIF": variance_inflation_factor(arr, i)})
                st.dataframe(pd.DataFrame(vif_df), use_container_width=True)
            except Exception as e:
                st.info(f"VIF not available: {e}")

            # Predictions & residuals
            yhat = model.predict(X_al)
            resid = y_al - yhat

            # Diagnostics
            st.markdown("### Diagnostics")
            # 1) Residuals vs Fitted
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=yhat, y=resid, mode="markers", name="Residuals", opacity=0.65))
            fig1.add_hline(y=0, line_dash="dot")
            fig1.update_layout(template="plotly_white", xaxis_title="Fitted values", yaxis_title="Residuals",
                               title="Residuals vs Fitted")
            # 2) QQ plot
            osm, osr = probplot(resid, dist="norm")[:2]
            xqq, _ = osm
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=np.sort(xqq), y=np.sort(resid), mode="markers", name="QQ"))
            fig2.add_trace(go.Scatter(x=[np.min(xqq), np.max(xqq)], y=[np.min(resid), np.max(resid)],
                                      mode="lines", name="45°", line=dict(dash="dot")))
            fig2.update_layout(template="plotly_white", xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles",
                               title="QQ Plot")
            # 3) Scale–Location
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=yhat, y=np.sqrt(np.abs(resid)), mode="markers", opacity=0.65))
            fig3.update_layout(template="plotly_white", xaxis_title="Fitted values", yaxis_title="√|Residuals|",
                               title="Scale–Location")
            # 4) Leverage vs Residuals²
            infl = model.get_influence()
            leverage = infl.hat_matrix_diag
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=leverage, y=resid**2, mode="markers", opacity=0.65))
            fig4.update_layout(template="plotly_white", xaxis_title="Leverage (hat diag)",
                               yaxis_title="Residual^2", title="Leverage vs Residuals^2")

            cdiag1, cdiag2 = st.columns(2)
            cdiag1.plotly_chart(fig1, use_container_width=True)
            cdiag1.plotly_chart(fig3, use_container_width=True)
            cdiag2.plotly_chart(fig2, use_container_width=True)
            cdiag2.plotly_chart(fig4, use_container_width=True)

            # Downloads
            pred_df = pd.DataFrame({"y": y_al, "yhat": yhat, "resid": resid})
            st.download_button("💾 Download predictions & residuals (CSV)",
                               data=pred_df.to_csv(index=False).encode("utf-8"),
                               file_name="pred_resid.csv", mime="text/csv")

# ===================== TAB 3: DATAVIZ =====================
with tab_viz:
    st.subheader("Quick Visualizations")

    st.markdown("### Correlation heatmap")
    num_for_corr = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_for_corr) >= 2:
        corr = df[num_for_corr].corr()
        figc = px.imshow(corr, text_auto=False, color_continuous_scale="RdBu_r", origin="lower",
                         labels=dict(color="corr"))
        figc.update_layout(template="plotly_white", height=500)
        st.plotly_chart(figc, use_container_width=True)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

    st.markdown("### Scatter matrix")
    pick_scatter = st.multiselect("Select up to 6 columns", df.columns.tolist(),
                                  default=num_for_corr[:4])
    if pick_scatter:
        pick_scatter = pick_scatter[:6]
        try:
            figsm = px.scatter_matrix(df[pick_scatter].dropna(),
                                      dimensions=pick_scatter, opacity=0.6)
            figsm.update_layout(template="plotly_white", height=600)
            st.plotly_chart(figsm, use_container_width=True)
        except Exception as e:
            st.info(f"Could not draw scatter matrix: {e}")

    # --- Time-series explorer (now supports integer YEAR columns) ---
    st.markdown("### Time-series explorer")
    df_ts, time_candidates = build_time_candidates(df)

    if time_candidates:
        tcol = st.selectbox("Time column", time_candidates, index=0, key="ts_timecol")
        tsd = df_ts.dropna(subset=[tcol]).sort_values(tcol)
        # choose series to plot
        numeric_series = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_series:
            st.info("No numeric series to plot. Add some numeric columns.")
        else:
            y1 = st.selectbox("Series 1", numeric_series, index=0, key="ts_y1")
            y2 = st.selectbox("Series 2 (optional)", ["(none)"] + numeric_series, index=0, key="ts_y2")

            figts = go.Figure()
            figts.add_trace(go.Scatter(x=tsd[tcol], y=tsd[y1], mode="lines", name=y1))
            if y2 != "(none)":
                figts.add_trace(go.Scatter(x=tsd[tcol], y=tsd[y2], mode="lines", name=y2, yaxis="y2"))
                figts.update_layout(yaxis2=dict(overlaying="y", side="right"))
            figts.update_layout(template="plotly_white",
                                xaxis_title=str(tcol),
                                yaxis_title=y1,
                                height=450)
            st.plotly_chart(figts, use_container_width=True)
    else:
        st.info("No time-like column found. Add a datetime column or an integer 'year' column (e.g., 1990..2025).")

# ----------------------- Footer -----------------------
st.markdown("---")
st.caption("Made for teaching and exploration. Please don’t use HARKed findings for policy or clinical decisions. ✔️")
