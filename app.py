import streamlit as st, sys
from utils.style import apply_global_style, hero

skip_css = "nocss" in st.query_params  # visit ?nocss=1 to disable custom CSS
if not skip_css:
    st.markdown("""<style> ... your CSS ... </style>""", unsafe_allow_html=True)

st.set_page_config(page_title="HARKing + Regression Lab", page_icon="📈", layout="wide")
apply_global_style()

hero("📈 HARKing + Regression Lab",
     "**A professional sandbox for exploring correlations, validating models, and visualizing data.** "
     "Use the sidebar to select data on each page. The pages menu is at the left (or top on mobile).")

st.sidebar.markdown("### Pages")
st.sidebar.caption("Use the selector at the top of the sidebar to switch pages. If links below aren't clickable on your host, they're just labels.")

# Try clickable links; if not supported in this environment, fall back to plain labels
def safe_page_link(path: str, label: str):
    try:
        # Works on recent Streamlit, fails gracefully otherwise
        st.sidebar.page_link(path, label=label)  # may raise KeyError('url_pathname') on some hosts
    except Exception:
        st.sidebar.markdown(f"- {label}")

safe_page_link("app.py", "🏠 Home")
safe_page_link("pages/01_HARKing.py", "🎣 HARKing")
safe_page_link("pages/02_Regression_Lab.py", "📐 Regression Lab")
safe_page_link("pages/03_DataViz_Studio.py", "📊 DataViz Studio")
# =========================
# Deep explanations (Home)
# =========================

st.markdown("## 👋 Welcome — What this app is and isn’t")
st.markdown("""
This site is a **hands-on sandbox** to (i) deliberately **HARK** (Hypothesizing After Results are Known) and 
see how easy it is to find spurious significance, (ii) **validate** proper models with diagnostics, and 
(iii) **visualize** your data quickly and cleanly.

**Intended use:** teaching, demos, exploration, and prototyping.  
**Not intended for:** policy, clinical, safety-critical, or publication-grade inference *without* rigorous follow-up.
""")

with st.expander("🧭 Quick start (60 seconds)", expanded=False):
    st.markdown("""
**HARKing page**
1. **Load data**: Sidebar ▸ **Simulate** (default) or **Upload CSV**.
2. Optional: Cap rows for speed (sampling).
3. Pick **HARKing knobs** (α, lags, min R², transforms, binning).
4. Hit **Explore specs** → see top hits, a “headline”, an auto-abstract, and a plot.

**Regression Lab**
1. Choose **Outcome** and **Predictors**.
2. Toggle **dummies**, **standardization**, **intercept**.
3. Inspect **summary**, **coefficients + CI**, **VIF**, and **diagnostics**.
4. Download predictions & residuals.

**DataViz Studio**
1. Explore **correlation heatmap**, **scatter matrix**, and **time-series**.
2. Time explorer accepts **datetime** or **integer year** (e.g., a column literally named `year`).
""")

with st.expander("🎣 What happens on the HARKing page (under the hood)?", expanded=False):
    st.markdown("""
We run a **massive grid** of simple OLS models **Y ~ X**:
- **Transforms** (z-score, log1p|signed, diff, pct_change, rolling means, detrend).
- **Lags** (negative & positive).
- **Bin filters** (quartiles/deciles on selected columns).
- Optional **interactions** (pairwise products on a subset of X’s for speed).

For each spec, we compute:
- **p-value** of X (two-sided),
- **R²**,
- **n** (effective sample),
- **β₁** (slope).

Then we:
- **Drop** specs with **non-finite R²**, zero variance, or **too few observations**.
- **Filter** by your **Minimum R²** setting.
- **Rank** the survivors by **p-value** (ascending).
- Show a cheeky **headline**, an **auto-abstract** explaining pitfalls, and an **OLS fit plot**.

**Multiple testing:** the table includes a `p_FDR<=α` (Benjamini–Hochberg) flag to show what would survive **FDR** correction.  
Expect most HARKed hits to **fail** this test — this is the *lesson*.
""")

with st.expander("📐 Regression Lab — how to read outputs", expanded=False):
    st.markdown("""
- **Model summary**: classic statsmodels table (R², Adj. R², F-stat, AIC/BIC, etc.).
- **Coefficients table**: estimate, standard error, t, p, and **95% CI**.
- **VIF**: checks **multicollinearity** among predictors (excluding the constant). VIF ≫ 10 is a red flag.
- **Diagnostics**:
  - **Residuals vs Fitted**: look for random scatter around 0 (linearity + homoskedasticity).
  - **QQ plot**: residual normality (approximate straight line).
  - **Scale–Location**: constant variance (no funnel shape).
  - **Leverage vs Residuals²**: influential points (high leverage or large residual^2).
- **Downloads**: get `y`, `yhat`, and `resid` as CSV for your own plots / tests.
""")

with st.expander("📊 DataViz Studio tips", expanded=False):
    st.markdown("""
- **Correlation heatmap**: fast orientation; beware of outliers and nonlinearity.
- **Scatter matrix**: eyeball pairwise relationships and clusters.
- **Time-series explorer**:
  - Accepts a true datetime column (e.g., `date`) **or** an **integer column named `year`**.
  - Plots 1–2 series together; second series overlays a right-axis for scale comparison.
""")

with st.expander("🧼 Data hygiene & guardrails", expanded=False):
    st.markdown("""
- **R² hygiene**: We skip any spec with **non-finite R²**, **zero variance** (X or Y), or **n < 3**.
- **Row cap**: Use sampling to keep UI responsive with large data.
- **Transforms**: Many transforms introduce **NaN at edges** (e.g., `diff`, `pct_change`).  
  The app **aligns and drops** missing values before OLS.
- **Bin filters**: When using `qcut` on data with ties or constant segments, we gracefully fall back to `cut`.
""")

with st.expander("⚡ Performance tips", expanded=False):
    st.markdown("""
- Start with **fewer transforms**, **smaller lags**, and **no interactions**.  
  Increase search space once you see the flow.
- Use the **Early stop** slider: stop after K “good” hits (**p<0.01** & **R²≥min**).
- Limit **binning** to fewer columns (we already cap to 3 for speed).
- Use the **Row cap** to sample large CSVs.
""")

with st.expander("🧪 Reproducibility & ethics", expanded=False):
    st.markdown("""
- The **HARKing page** is designed to show how **easy** it is to get flashy but misleading results.
- Always **confirm** findings in the **Regression Lab** with diagnostics.
- Prefer **held-out validation**, **pre-registration**, and **theory-driven** specifications.
- If you report results, disclose **multiple-testing corrections** and **specification search**.
""")

with st.expander("❓ FAQ", expanded=False):
    st.markdown("""
**Q: My time series is a column of integers (e.g., 2000–2024). Will it work?**  
A: Yes. The time explorer treats a column literally named **`year`** as temporal and converts it internally.

**Q: Why do I only see a handful of “discoveries”?**  
A: Your **Minimum R²** filter may be removing many specs. Lower it, or broaden transforms/lags.

**Q: Why do some transforms look empty?**  
A: Transforms like `diff` or `pct_change` create NaNs at edges. We drop rows with missing values before OLS.

**Q: Can I add my own transforms?**  
A: Yes—extend the `TRANSFORMS` dict in code. Keep them **pure functions** `array -> array`.

**Q: Why does VIF sometimes fail?**  
A: If the design matrix is singular (perfect collinearity) or too small, VIF can’t be computed; we show a friendly note.
""")

with st.expander("🧩 How to prepare a CSV for upload", expanded=False):
    st.markdown("""
Minimum viable CSV:
- Header row with variable names (ASCII is safest).
- At least **two numeric columns** (floats/ints).
- Optionally a **`date`** (ISO-8601) or **`year`** (integer) column.
- Avoid mixed types (e.g., strings mixed with numbers) in the same column.
""")
    st.code("""example_min.csv
date,year,household_income,unemployment_rate,energy_price,target_policy
2010-01-01,2010,42000,8.1,100.2,0.5
2010-02-01,2010,42120,8.0,99.9,0.3
...""", language="text")

with st.expander("🧱 Architecture (high level)", expanded=False):
    st.markdown("""
- **Home** (this page) explains features and navigation.
- **pages/01_HARKing.py**: grid search over Y×X×Transforms×Lags×Filters with guardrails and early stop.
- **pages/02_Regression_Lab.py**: clean OLS builder + diagnostics + VIF.
- **pages/03_DataViz_Studio.py**: quick visual analytics (heatmap, scatter matrix, time series).
- **utils/style.py**: theme + components (`apply_global_style`, `hero`, cards).
""")

st.markdown("---")
st.info("**Pro tip:** You can disable the custom CSS theme by adding `?nocss=1` to the URL if you need a bare Streamlit look for debugging.")
st.caption("This app is educational; results are illustrative. Validate seriously before trusting insights.")
