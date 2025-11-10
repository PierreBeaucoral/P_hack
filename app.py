import streamlit as st, sys
from utils.style import apply_global_style, hero

# Optional CSS toggle (?nocss=1)
skip_css = "nocss" in st.query_params
if not skip_css:
    st.markdown("""<style> /* your CSS here if any */ </style>""", unsafe_allow_html=True)

st.set_page_config(page_title="HARKing + Regression Lab", page_icon="📈", layout="wide")
apply_global_style()

# -------------------------
# Sidebar: Pages navigation
# -------------------------
st.sidebar.markdown("### Pages")
st.sidebar.caption("Use the selector at the top of the sidebar to switch pages. If links below aren't clickable on your host, they're just labels.")

def safe_page_link(path: str, label: str):
    try:
        st.sidebar.page_link(path, label=label)  # may raise KeyError on some hosts
    except Exception:
        st.sidebar.markdown(f"- {label}")

safe_page_link("app.py", "🏠 Home")
safe_page_link("pages/01_HARKing.py", "🎣 HARKing (single X)")
safe_page_link("pages/04_MultiX_HARKing.py", "🧪 Multivariate HARKing (main X + controls)")
safe_page_link("pages/02_Regression_Lab.py", "📐 Regression Lab")
safe_page_link("pages/03_DataViz_Studio.py", "📊 DataViz Studio")

st.markdown("## 👋 Welcome — What this app is and isn’t")
st.markdown("""
This site is a **hands-on sandbox** to (i) deliberately **HARK** (Hypothesizing After Results are Known) and 
see how easy it is to find spurious significance, (ii) **validate** proper models with diagnostics, and 
(iii) **visualize** your data quickly and cleanly.

**Intended use:** teaching, demos, exploration, and prototyping.  
**Not intended for:** policy, clinical, safety-critical, or publication-grade inference *without* rigorous follow-up.
""")

# -----------------------
# Quick start + Deep info
# -----------------------
with st.expander("🧭 Quick start (60 seconds)", expanded=False):
    st.markdown("""
**🎣 HARKing (single X)**
1. Load data (Simulate or Upload CSV).
2. Set α, lags, transforms, and **Minimum R²**.
3. Click explore → see top hits, cheeky headline, abstract, and a fit plot.

**🧪 Multivariate HARKing (main X + controls)**
1. Pick **Y** and **Main X**; choose a **pool of controls**.
2. Choose Y/X transforms, **lags on Main X**, and control-set size bounds.
3. The page tries many control combinations; **ranks by p-value of Main X**; filters by **model R²**.
4. Inspect the **Partial Regression** (FWL) plot to see the marginal effect of Main X after controls.

**📐 Regression Lab**
1. Pick outcome/predictors; toggle dummies/standardization/intercept.
2. Read the summary, **coef table with CIs**, **VIF**, and **diagnostics**.
3. Download predictions & residuals.

**📊 DataViz Studio**
1. Heatmap, scatter matrix, and time-series explorer.
2. Time accepts true datetime **or** integer **`year`**.
""")

with st.expander("🎣 What happens on the (single-X) HARKing page?", expanded=False):
    st.markdown("""
We run a **grid** of OLS **Y ~ X** across:
- **Transforms** (zscore, log1p|signed, diff, pct_change, rolling means, detrend).
- **Lags** (negative & positive).
- **Bin filters** (quartiles/deciles).
- Optional **interactions** (pairwise products subset).

For each spec we compute **p**, **R²**, **n**, **β₁**; we **drop** specs with non-finite R², zero variance, or too few obs, apply **Minimum R²**, then **rank by p**.  
We display a headline, auto-abstract, and OLS fit. `p_FDR<=α` indicates BH-FDR survivals (expect **few**).
""")

with st.expander("🧪 What happens on **Multivariate HARKing (Main X + Controls)**?", expanded=True):
    st.markdown("""
We focus on a **main regressor** and search across **control sets**:

- You select **Y**, **Main X**, and a **pool of potential controls**.
- The app enumerates (or samples) control combinations between your chosen **min/max size** (capped for speed).
- We try **transforms** for Y and Main X and **lags on Main X**.
- Each model is **Y ~ const + Main X + controls**.
- We record **p(Main X)**, **R²**, **n**, and **β(Main X)**.
- We **keep** only models with **finite R²**, **n≥3**, non-zero variance, and **R² ≥ Minimum model R²**.
- Results are **sorted by p(Main X)** (ascending), then by R² (descending).
- A **Partial Regression (FWL) plot** shows the marginal relationship of Main X after netting out controls.

**Guardrails**
- Early-stop after K good hits (**p<0.01** & **R²≥min**).
- Maximum control sets and model caps to control runtime.
- Controls share a simple transform (identity or zscore) to keep search tractable.
""")

with st.expander("📐 Regression Lab — how to read outputs", expanded=False):
    st.markdown("""
- **Summary** (R², Adj.R², F-stat, AIC/BIC…)
- **Coefficients**: estimate, SE, t, p, **95% CI**
- **VIF**: multicollinearity (VIF≫10 red flag)
- **Diagnostics**: Residuals vs Fitted, QQ, Scale–Location, Leverage vs Residuals²
- **Downloads**: `y`, `yhat`, `resid`
""")

with st.expander("📊 DataViz Studio tips", expanded=False):
    st.markdown("""
- **Heatmap**: quick orientation (watch outliers/nonlinearity)
- **Scatter matrix**: clusters / shape
- **Time-series explorer**: accepts `date` or integer `year`; dual-axis overlay for a second series
""")

with st.expander("🧼 Data hygiene & guardrails", expanded=False):
    st.markdown("""
- Skip specs with **non-finite R²**, **zero variance**, or **n < 3**.
- Transforms like `diff`/`pct_change` create NaNs at edges → we **align & drop**.
- `qcut` with ties falls back to `cut`.
- Use **Row cap** to sample large CSVs for snappy UI.
""")

with st.expander("⚡ Performance tips", expanded=False):
    st.markdown("""
- Start small: fewer transforms, smaller lags, no interactions.
- **Early stop** is your friend.
- Limit binning columns; cap #control sets.
""")

with st.expander("🧪 Reproducibility & ethics", expanded=False):
    st.markdown("""
- HARKing is a demonstration of how easy **spurious significance** can be.
- Confirm in **Regression Lab**, use **held-out validation**, **pre-registration**, and **multiple-testing corrections**.
""")

with st.expander("❓ FAQ", expanded=False):
    st.markdown("""
**Q: My time series uses integers (2000–2024). Will it work?**  
A: Yes. If the column is named **`year`**, the explorer treats it as temporal.

**Q: Why so few discoveries?**  
A: Your **Minimum R²** filter may be aggressive; lower it or widen search knobs.

**Q: Can I add custom transforms?**  
A: Extend the `TRANSFORMS` dict on the relevant page; keep them pure `array -> array`.
""")

with st.expander("🧱 Architecture (high level)", expanded=False):
    st.markdown("""
- **Home** (this page): documentation + navigation
- **pages/01_HARKing.py**: grid search over Y×X×Transforms×Lags×Filters (single X)
- **pages/04_MultiX_HARKing.py**: **Main X + controls** search, ranked by p(Main X), partial regression plot
- **pages/02_Regression_Lab.py**: clean OLS builder + diagnostics + VIF
- **pages/03_DataViz_Studio.py**: quick visual analytics
- **utils/style.py**: theme + components (`apply_global_style`, `hero`, cards)
""")

st.markdown("---")
st.info("**Pro tip:** Add `?nocss=1` to the URL to disable custom CSS for debugging.")
st.caption("Educational app; results are illustrative. Validate seriously before trusting insights.")
