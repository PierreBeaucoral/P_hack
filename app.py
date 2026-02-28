# app.py — The P-Hacker's Toolkit  (v2)
# Educational tool demonstrating HARKing, multiple testing, and p-hacking mechanics.
# Guards: skip specs with zero variance / too few obs; keep R² even if NaN (per request).

import itertools, time, random
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import probplot
import statsmodels.api as sm

# ═══════════════════════════ PAGE CONFIG ════════════════════════════
st.set_page_config(
    page_title="The P-Hacker's Toolkit",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════ CSS THEME ══════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0e1117; }
.main .block-container { padding-top: 0.5rem; max-width: 1280px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #16213e 60%, #0d2137 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem 2rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #e94560 0%, #f5803a 50%, #f7c948 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-sub { font-size: 1.05rem; color: #8899aa; max-width: 680px; line-height: 1.5; }
.badge {
    display: inline-block; background: rgba(233,69,96,0.15);
    border: 1px solid rgba(233,69,96,0.4); color: #e94560;
    border-radius: 20px; padding: 2px 12px; font-size: 0.72rem;
    font-weight: 700; letter-spacing: 0.06em; margin-right: 6px; margin-top: 0.8rem;
}
.badge.green { background: rgba(72,187,120,0.15); border-color: rgba(72,187,120,0.4); color: #68d391; }
.badge.blue  { background: rgba(66,153,225,0.15); border-color: rgba(66,153,225,0.4); color: #90cdf4; }

/* ── Stat row ── */
.stat-row { display: flex; gap: 1rem; margin: 1.2rem 0; }
.stat-box {
    flex: 1; background: #161b2a; border: 1px solid #252d42;
    border-radius: 10px; padding: 0.9rem 1rem; text-align: center;
}
.stat-val { font-size: 1.7rem; font-weight: 800; color: #e94560; line-height: 1; }
.stat-lbl { font-size: 0.68rem; color: #5a6a80; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 3px; }

/* ── Discovery card ── */
.disc-card {
    background: #161b2a; border: 1px solid #2a3350;
    border-left: 4px solid #e94560; border-radius: 12px;
    padding: 1.1rem 1.4rem; margin: 0.8rem 0;
}
.disc-card.fdr-pass { border-left-color: #48bb78; }
.disc-card-title { color: #e2e8f0; font-size: 0.97rem; font-style: italic; margin: 0 0 0.5rem 0; font-weight: 600; }
.disc-meta { color: #5a6a80; font-size: 0.78rem; }
.pill {
    display: inline-block; border-radius: 5px; padding: 0 7px;
    font-size: 0.76rem; font-family: monospace; margin-right: 5px;
}
.pill-red   { background: rgba(229,62,62,0.15);  color: #fc8181; }
.pill-green { background: rgba(72,187,120,0.15); color: #68d391; }
.pill-blue  { background: rgba(66,153,225,0.15); color: #90cdf4; }
.pill-gray  { background: rgba(74,85,104,0.3);   color: #a0aec0; }

/* ── Paper mockup ── */
.paper {
    background: #fdfaf3; color: #1a202c; border-radius: 10px;
    padding: 2rem 2.4rem; font-family: Georgia, serif;
    border: 1px solid #e2d9c5; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.paper-risk {
    float: right; border: 2.5px solid #e53e3e; color: #e53e3e;
    border-radius: 6px; padding: 3px 10px; font-size: 0.8rem; font-weight: 800;
    letter-spacing: .1em; transform: rotate(-5deg); display: inline-block;
    margin: 0 0 0.5rem 1rem; background: rgba(229,62,62,0.06);
}
.paper-journal { text-align:center; font-size:0.7rem; color:#718096; letter-spacing:.18em; text-transform:uppercase; border-bottom:2px solid #e2d9c5; padding-bottom:.5rem; margin-bottom:1rem; clear:both; }
.paper-title   { font-size:1.2rem; font-weight:700; text-align:center; color:#1a202c; margin-bottom:.4rem; line-height:1.45; }
.paper-authors { text-align:center; font-size:0.82rem; color:#4a5568; font-style:italic; margin-bottom:1rem; }
.paper-abs-hd  { font-size:0.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#4a5568; margin-bottom:.25rem; }
.paper-abs-txt { font-size:0.88rem; line-height:1.65; color:#2d3748; }
.paper-kw      { font-size:0.78rem; color:#718096; margin-top:.8rem; }

/* ── Citation box ── */
.cite { background:#161b2a; border-left:3px solid #4299e1; border-radius:0 6px 6px 0; padding:.4rem .9rem; font-size:.8rem; color:#90cdf4; margin:.3rem 0; }

/* ── Checklist ── */
.checklist-item { display:flex; align-items:flex-start; gap:.5rem; padding:.3rem 0; font-size:.9rem; color:#cbd5e0; }
.check-yes { color:#68d391; font-size:1rem; }
.check-no  { color:#fc8181; font-size:1rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════ CONSTANTS ══════════════════════════════
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

FRIENDLY_VARS = [
    "household_income", "inflation_rate", "unemployment_rate", "policy_rate",
    "industrial_output", "consumer_confidence", "housing_prices", "exports_value",
    "imports_value", "net_trade", "exchange_rate", "energy_price", "co2_emissions",
    "tourism_arrivals", "public_spending", "private_investment", "bank_credit",
    "stock_index", "wage_index", "productivity_index",
]

FAKE_JOURNALS = [
    "Journal of Irreproducible Results",
    "International Quarterly of Post-Hoc Analysis",
    "Review of Speculative Econometrics",
    "Proceedings of the Society for Data Dredging",
    "Annals of Exploratory Research (Post-Publication)",
    "Cambridge Journal of Garden-Variety Regressions",
    "American Economic Digest of Marginally Significant Findings",
    "Nature-adjacent Letters",
    "PLOS Convenient",
    "European Journal of Convenient Correlations",
]

FAKE_AUTHORS = [
    "H. Arking, R.E. Sults & P. Hacker",
    "A. Confirmation, B. Ias & S. Election",
    "P. Value, T. Test & R. Square",
    "D. Redge & C. Orrelation",
    "F. Ishin, P. Ost-Hoc & I. Dentification",
    "A.N. Other, T. Rollope & M. Ultipletest",
]

HEADLINE_VERBS = [
    "predicts", "explains", "drives", "is tightly coupled with",
    "foretells", "perfectly mirrors", "causally influences",
    "is a robust determinant of", "Granger-causes",
]

SPICE = [
    "per aggressive model exploration",
    "after extensive post-hoc analysis",
    "using cutting-edge curve fitting",
    "consistent with our priors (retroactively)",
    "following a thorough p-hack",
    "via undisclosed specification search",
    "through 47 unreported robustness checks",
    "after controlling for the kitchen sink",
    "as expected by our theory (written this morning)",
]

METHODS_BOILERPLATE = [
    "employing a rigorous multi-specification framework",
    "leveraging state-of-the-art transformation techniques",
    "through comprehensive sensitivity analysis across {n} candidate models",
    "using a novel data-driven methodological approach",
    "after controlling for all conceivable confounders post-hoc",
    "in a landmark exploratory study of unprecedented scope",
]

IMPLICATIONS = [
    "These findings have far-reaching policy implications.",
    "Policymakers should act on these results immediately.",
    "This result is robust to perturbations we have not yet checked.",
    "Future research (by others) should replicate this finding.",
    "The mechanism is left as an exercise to the reader.",
    "We expect a Nobel citation within the decade.",
]

PHACKING_FACTS = [
    "💡 **Did you know?** Simmons et al. (2011) showed that with just 4 flexible researcher decisions, you can push a false-positive rate from 5% to **61%**.",
    "💡 **Did you know?** Kerr (1998) coined 'HARKing' — presenting exploratory findings as if they were confirmatory hypotheses.",
    "💡 **Did you know?** In the 2015 Reproducibility Project, only **36%** of psychology findings replicated at p < .05.",
    "💡 **Did you know?** Benjamini & Hochberg (1995) controls the *expected proportion* of false discoveries, not the probability of *any* false discovery.",
    "💡 **Did you know?** Gelman & Loken (2014) called post-hoc hypothesis selection 'the garden of forking paths'.",
    "💡 **Did you know?** Pre-registration dramatically reduces p-hacking — and top journals increasingly require it.",
    "💡 **Did you know?** With 20 independent tests at α=0.05, you expect at least one false positive ~**64%** of the time.",
    "💡 **Did you know?** The term 'p-hacking' was coined by Simonsohn et al. in their influential 2014 paper.",
]

# ═══════════════════════════ UTILITIES ══════════════════════════════

def pick_friendly_names(k: int):
    base = FRIENDLY_VARS.copy()
    rng.shuffle(base)
    if k <= len(base):
        return base[:k]
    extra, i = [], 1
    while len(base) + len(extra) < k:
        extra.extend([f"{name}_{i}" for name in FRIENDLY_VARS])
        i += 1
    return (base + extra)[:k]


def simulate_data(n=800, k=8):
    names = pick_friendly_names(k)
    X = {names[i]: rng.normal(0, 1, n) for i in range(k)}
    t = pd.date_range("2010-01-01", periods=n, freq="D")
    X["time"]  = np.arange(n)
    X["month"] = pd.to_datetime(t).month
    X["date"]  = t
    df = pd.DataFrame(X)
    x1, x2 = df[names[0]].values, df[names[1]].values
    df["target_null"]   = rng.normal(0, 1, n)
    df["target_policy"] = 0.3*x1 - 0.2*(x2 > 0).astype(int) + rng.normal(0, 1, n)
    df["target_trendy"] = rng.normal(0, 1, n).cumsum() + rng.normal(0, 5, n)
    return df


def build_time_candidates(df: pd.DataFrame):
    df = df.copy()
    time_cols = []
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            time_cols.append(c)
    for c in df.columns:
        name = str(c).lower()
        if "year" in name and (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c])):
            ser_num = pd.to_numeric(df[c], errors="coerce")
            if (ser_num.between(1800, 2100)).mean() >= 0.8:
                cname = f"{c}__as_date"
                df[cname] = pd.to_datetime(ser_num.astype("Int64").astype(str), format="%Y", errors="coerce")
                time_cols.append(cname)
                continue
        if ("date" in name) or ("time" in name):
            ser_parsed = pd.to_datetime(df[c], errors="coerce")
            if ser_parsed.notna().mean() >= 0.6:
                cname = f"{c}__as_date"
                df[cname] = ser_parsed
                time_cols.append(cname)
    seen, out = set(), []
    for c in time_cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return df, out


def make_bins(x, q=4):
    s = pd.Series(x)
    try:
        return pd.qcut(s, q=q, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=q)


def align(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def detrend_linear(s):
    s = np.asarray(s, float)
    idx = np.arange(len(s))
    ok = np.isfinite(s)
    if ok.sum() < 3:
        return s
    b = np.polyfit(idx[ok], s[ok], 1)
    return s - (b[0]*idx + b[1])


def ihs(s):
    """Inverse hyperbolic sine — handles zeros and negatives gracefully."""
    return np.arcsinh(np.asarray(s, float))


TRANSFORMS = {
    "identity":       lambda s: np.asarray(s, float),
    "zscore":         lambda s: (np.asarray(s, float) - np.nanmean(s)) / (np.nanstd(s) + 1e-12),
    "log1p|signed":   lambda s: np.sign(s) * np.log1p(np.abs(s)),
    "IHS":            ihs,
    "diff":           lambda s: pd.Series(s).diff().values,
    "pct_change":     lambda s: pd.Series(s).pct_change().replace([np.inf, -np.inf], np.nan).values,
    "rollmean_3":     lambda s: pd.Series(s).rolling(3, min_periods=1).mean().values,
    "rollmean_6":     lambda s: pd.Series(s).rolling(6, min_periods=1).mean().values,
    "sqrt|signed":    lambda s: np.sign(s) * np.sqrt(np.abs(s)),
    "square":         lambda s: np.square(s),
    "rank":           lambda s: pd.Series(s).rank(pct=True).values,
    "detrend_linear": detrend_linear,
}


def regress_simple(x, y):
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, 0, np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, 0, np.nan
    X = sm.add_constant(x, has_constant="add")
    try:
        model = sm.OLS(y, X, missing="drop").fit()
        pv = model.pvalues[1] if len(model.params) > 1 else np.nan
        r2 = model.rsquared
        n  = int(model.nobs)
        b1 = model.params[1] if len(model.params) > 1 else np.nan
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


# ─── Text generators ────────────────────────────────────────────────

def retraction_probability(p, r2, n, n_specs_tried):
    """Heuristic retraction-risk score for fun display — not a real statistic."""
    p_adj = min(p * n_specs_tried, 1.0)   # rough Bonferroni adjustment
    base  = 1.0 - p_adj
    if np.isfinite(r2):
        base *= (1 - r2 * 0.5)
    n_penalty = max(0, 1 - n / 500)
    score = (base * 0.6 + n_penalty * 0.4) * 100
    return round(min(max(score, 5), 98), 1)


def silly_headline(xn, yn, p, r2, lag, tfx, tfy, filt_desc):
    lagbit  = f" (lag {lag})" if lag else ""
    fx      = "" if tfx == "identity" else f" [{tfx}]"
    fy      = "" if tfy == "identity" else f" [{tfy}]"
    filt    = f" {filt_desc}" if filt_desc else ""
    r2_text = f"{r2:.2f}" if np.isfinite(r2) else "?"
    return (f"{xn}{fx} {random.choice(HEADLINE_VERBS)} {yn}{fy}{lagbit}"
            f" (p={p:.3g}, R²={r2_text}) — {random.choice(SPICE)}{filt}")


def paper_title(xn, yn, tfx, tfy, lag):
    xn_h = xn.replace("_", " ").title()
    yn_h = yn.replace("_", " ").title()
    templates = [
        f"The Causal Effect of {xn_h} on {yn_h}: New Evidence",
        f"Does {xn_h} Drive {yn_h}? A Longitudinal Analysis",
        f"Revisiting the {xn_h}–{yn_h} Nexus",
        f"On the Robust Association Between {xn_h} and {yn_h}",
        f"A Novel {tfx.replace('_',' ').title()} Approach to Understanding {yn_h}",
    ]
    title = random.choice(templates)
    if lag != 0:
        title += f": The Role of Temporal Dynamics (Lag = {abs(lag)})"
    return title


def bogus_abstract(xn, yn, p, r2, n, lag, tfx, tfy, filt_desc, n_specs):
    lagbit     = f" with a temporal lag of {lag} period(s)" if lag else ""
    fx         = "" if tfx == "identity" else f" after a {tfx} transform"
    fy         = "" if tfy == "identity" else f" on a {tfy} scale"
    filt       = f" restricted to {filt_desc}" if filt_desc else ""
    method_bit = random.choice(METHODS_BOILERPLATE).format(n=n_specs)
    implication = random.choice(IMPLICATIONS)
    r2_text    = f"{r2:.2f}" if np.isfinite(r2) else "NaN"
    return (
        f"<strong>Background:</strong> Theoretical models suggest a link between "
        f"{xn.replace('_',' ')} and {yn.replace('_',' ')}, yet empirical evidence remains scarce. "
        f"<strong>Methods:</strong> We examine this relationship {method_bit}{fx}{lagbit}{filt}. "
        f"Drawing on {n:,} observations, we estimate a bivariate OLS specification. "
        f"<strong>Results:</strong> {xn.replace('_',' ').title()}{fy} is significantly associated with "
        f"{yn.replace('_',' ')} (two-sided p = {p:.3g}, R² = {r2_text}, n = {n:,}). "
        f"<strong>Conclusion:</strong> {implication} "
        f"<em>Note: This result was selected from {n_specs:,} candidate specifications.</em>"
    )


def fake_keywords(xn, yn, tfx):
    base = [xn.replace("_", " "), yn.replace("_", " "), "OLS regression", "panel data"]
    if tfx not in ("identity", "zscore"):
        base.append(tfx.replace("_", " "))
    random.shuffle(base)
    return "; ".join(base[:4])


# ═══════════════════════════ SIDEBAR & DATA ══════════════════════════
st.sidebar.header("📂 Data")
st.sidebar.markdown("""
**Quick start**
1. Keep **Simulate** if you're exploring.
2. Go to **🎣 HARKing** and hit Search.
3. Admire your "discovery".
4. Check it in **📐 Regression Lab**.
""")

mode = st.sidebar.radio("Data source", ["Simulate", "Upload CSV"], horizontal=True)
if mode == "Simulate":
    n_rows = st.sidebar.slider("Rows", 200, 5_000, 800, step=100)
    n_vars = st.sidebar.slider("Variables", 3, 20, 8, step=1)
    df = simulate_data(n=n_rows, k=n_vars)
else:
    up = st.sidebar.file_uploader("CSV file", type=["csv"])
    if up is None:
        st.stop()
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

max_rows = st.sidebar.slider("Row cap (HARKing speed)", 1_000, 100_000, 10_000, step=1_000)
if len(df) > max_rows:
    df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    st.sidebar.caption(f"Sampled to {max_rows:,} rows for speed.")

st.sidebar.markdown("---")
st.sidebar.markdown(random.choice(PHACKING_FACTS))

with st.expander("🔍 Data preview", expanded=False):
    st.dataframe(df.head(12), use_container_width=True)

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(num_cols) < 2:
    st.error("Need at least two numeric columns in the data.")
    st.stop()

# ═══════════════════════════ HERO BANNER ════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">🎣 The P-Hacker's Toolkit</div>
  <div class="hero-sub">
    A playful but rigorous sandbox for understanding HARKing, data dredging, and
    multiple-testing failures — and how to guard against them.
  </div>
  <div style="margin-top:1rem">
    <span class="badge">Educational</span>
    <span class="badge green">Benjamini–Hochberg FDR</span>
    <span class="badge blue">OLS Diagnostics</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════ TABS ════════════════════════════════════
tab_home, tab_hark, tab_reg, tab_viz = st.tabs(
    ["🏠 Overview", "🎣 HARKing", "📐 Regression Lab", "📊 DataViz"]
)

# ══════════════════════════ TAB 0: OVERVIEW ══════════════════════════
with tab_home:
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("## What is HARKing?")
        st.markdown("""
**HARKing** — *Hypothesizing After Results are Known* — is the practice of presenting
exploratory findings as if they were confirmatory hypotheses stated *before* the analysis.

Combined with **specification searching** (trying many model variants until something
is "significant"), it inflates the false-positive rate dramatically — even with perfectly
innocent intentions.

This app lets you **experience it first-hand** and see exactly how it works.
        """)
        st.markdown("### How to use this app")
        st.markdown("""
1. **Pick data** in the left sidebar (simulated or your own CSV).
2. **🎣 HARKing tab** — run the brute-force spec search. Watch "discoveries" pile up.
3. **📐 Regression Lab** — take your best HARKed finding and test it properly,
   with full diagnostics and VIF.
4. **📊 DataViz** — correlations, scatter matrices, time-series overlays.
        """)
        st.markdown("### The mechanism at a glance")
        st.markdown("""
| Researcher action | False-positive cost |
|---|---|
| Try 2 outcome variables | ~2× |
| Try 4 transforms | ~4× |
| Try 5 lag windows | ~5× |
| Add bin-filtering (quartiles) | ~4× |
| **All combined** | **~160×** |

At α = 0.05, you now expect a false positive in almost **every single run**.
        """)

    with col_r:
        st.markdown("## Key literature")
        st.markdown("""
<div class="cite">Kerr, N.L. (1998). HARKing: Hypothesizing after the results are known.
<em>Personality & Social Psychology Review</em>, 2(3), 196–217.</div>
<div class="cite">Simmons, J.P., Nelson, L.D., & Simonsohn, U. (2011). False-positive psychology.
<em>Psychological Science</em>, 22(11), 1359–1366.</div>
<div class="cite">Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
<em>JRSS-B</em>, 57(1), 289–300.</div>
<div class="cite">Gelman, A., & Loken, E. (2014). The statistical crisis in science.
<em>American Scientist</em>, 102(6), 460–465.</div>
<div class="cite">Open Science Collaboration (2015). Estimating the reproducibility of psychological science.
<em>Science</em>, 349(6251).</div>
        """, unsafe_allow_html=True)

        st.markdown("## Good practice checklist")
        st.markdown("""
<div class="checklist-item"><span class="check-no">✗</span>&nbsp; Run thousands of regressions and report only the best</div>
<div class="checklist-item"><span class="check-no">✗</span>&nbsp; Choose transforms and lags after seeing the p-values</div>
<div class="checklist-item"><span class="check-no">✗</span>&nbsp; Bin predictors to find "the right quartile"</div>
<div class="checklist-item"><span class="check-yes">✓</span>&nbsp; Pre-register your hypothesis and specification</div>
<div class="checklist-item"><span class="check-yes">✓</span>&nbsp; Apply FDR / Bonferroni correction for multiple tests</div>
<div class="checklist-item"><span class="check-yes">✓</span>&nbsp; Report all specifications tested, not just the winners</div>
<div class="checklist-item"><span class="check-yes">✓</span>&nbsp; Validate on out-of-sample or held-out data</div>
        """, unsafe_allow_html=True)

# ══════════════════════════ TAB 1: HARKING ═══════════════════════════
with tab_hark:
    st.subheader("🎣 HARKing Generator")
    st.markdown("Run a brute-force spec search. Revel in your findings. Then feel appropriately bad about it.")

    # ── Controls ──────────────────────────────────────────────────────
    st.sidebar.header("🎣 HARKing knobs")
    alpha       = st.sidebar.slider("Significance α", 0.001, 0.10, 0.05, 0.001, key="alpha")
    max_lag     = st.sidebar.slider("Max lag to try", 0, 12, 2, 1, key="max_lag")
    min_r2      = st.sidebar.slider("Minimum R²", 0.00, 0.99, 0.30, 0.01,
                                    help="Skip specs with R² below this. NaN R² bypasses the check.",
                                    key="min_r2")
    do_bins     = st.sidebar.checkbox("Try binning (quartiles/deciles)", True, key="do_bins")
    do_interact = st.sidebar.checkbox("Try interactions (X₁×X₂)", False, key="do_interact")

    col_k, col_gs, col_cap = st.columns(3)
    max_results = col_k.slider("Keep top K results", 5, 100, 20, 5, key="hark_k")
    good_needed = col_gs.slider("Early stop after K good hits", 0, 200, 20, 5, key="good_needed")
    max_models  = col_cap.slider("Hard model cap", 1_000, 300_000, 30_000, 1_000, key="max_models")

    # ── Variable pools ─────────────────────────────────────────────────
    st.markdown("#### Variable pools")
    blacklist = {"target_trendy", "time", "month"}
    pool_cols = list(num_cols)
    c_yp, c_xp = st.columns(2)
    Y_pool = c_yp.multiselect("Y candidates", pool_cols, default=pool_cols, key="y_pool")
    X_pool = c_xp.multiselect("X candidates", pool_cols,
                               default=[c for c in pool_cols if c not in blacklist], key="x_pool")
    if not Y_pool or not X_pool:
        st.warning("Select at least one variable in each pool.")
        st.stop()

    # ── Transform choices ──────────────────────────────────────────────
    all_tf = list(TRANSFORMS.keys())
    default_tf = [t for t in ["identity", "zscore", "log1p|signed", "IHS", "diff", "detrend_linear"] if t in all_tf]
    trans_choices = st.multiselect("Transforms to consider", all_tf, default=default_tf, key="tf_choices")

    # ── Build working df & interaction columns ─────────────────────────
    df_work = df.copy()
    eng_cols = []
    if do_interact and len(X_pool) >= 2:
        for a, b in itertools.combinations(X_pool[:min(6, len(X_pool))], 2):
            name = f"{a}×{b}"
            df_work[name] = df_work[a] * df_work[b]
            eng_cols.append(name)

    Ycands = list(Y_pool)
    Xcands = list(X_pool) + eng_cols

    # ── Bin filters ────────────────────────────────────────────────────
    filters, filt_labels = [None], [""]
    if do_bins:
        for col in X_pool[:min(3, len(X_pool))]:
            for q, label_tmpl in [(4, f"{col} quartile Q"), (10, f"{col} decile D")]:
                bins = make_bins(df_work[col], q=q)
                if hasattr(bins, "codes"):
                    codes = pd.Series(bins.cat.codes).values
                    for code in np.unique(codes[codes >= 0])[:4]:
                        filters.append(codes == code)
                        filt_labels.append(label_tmpl.replace("Q", str(code+1)).replace("D", str(code+1)))

    # ── Estimation ─────────────────────────────────────────────────────
    n_filters = len(filters)
    n_lags    = 2*max_lag + 1
    n_tx      = max(len(trans_choices), 1)
    est_total = len(Ycands) * max(0, len(Xcands)) * n_tx * n_tx * n_lags * n_filters

    st.info(
        f"About to try ~ **{est_total:,}** specs "
        f"(Y={len(Ycands)}, X={len(Xcands)}, "
        f"transforms={n_tx}², lags={n_lags}, filters={n_filters})"
    )

    results = []
    prog     = st.progress(0)
    status   = st.empty()
    model_counter, good_hits = 0, 0
    last_t   = time.time()
    stop_now = False

    def try_one(xname, yname, lag, tfx, tfy, filt_mask=None, label_filt=""):
        x = df_work[xname].values
        y = df_work[yname].values
        x = TRANSFORMS[tfx](x) if tfx in TRANSFORMS else x
        y = TRANSFORMS[tfy](y) if tfy in TRANSFORMS else y
        if lag > 0:   x, y = x[:-lag], y[lag:]
        elif lag < 0: x, y = x[-lag:], y[:lag]
        if filt_mask is not None:
            x, y = x[filt_mask], y[filt_mask]
        x, y = align(x, y)
        if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return None
        pv, r2, n, b1 = regress_simple(x, y)
        if not np.isfinite(pv):
            return None
        if np.isfinite(r2) and r2 < min_r2:
            return None
        results.append({
            "x": xname, "y": yname, "lag": lag, "tfx": tfx, "tfy": tfy,
            "p": float(pv), "R2": float(r2), "n": int(n), "b1": float(b1),
            "filter": label_filt,
        })
        return results[-1]

    try:
        with st.spinner("🌪️ Trawling through specifications…"):
            total_loops = est_total if est_total > 0 else 1
            loop_done = 0
            for yname in Ycands:
                for xname in Xcands:
                    if xname == yname:
                        loop_done += n_tx * n_tx * n_lags * n_filters
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
                                        status.text(f"Tried {model_counter:,} / ~{est_total:,} specs…")
                                        last_t = time.time()
                                    if row and (row["p"] < 0.01) and (np.isfinite(row["R2"]) and row["R2"] >= min_r2):
                                        good_hits += 1
                                    if (good_needed and good_hits >= good_needed) or (model_counter >= max_models):
                                        stop_now = True
                                        raise SystemExit
    except SystemExit:
        pass

    prog.progress(1.0)
    status.empty()

    resdf = pd.DataFrame(results)
    if resdf.empty:
        st.warning("No models met the criteria. Lower R², change pools, or widen transforms/lags/filters.")
    else:
        resdf.sort_values("p", inplace=True)
        resdf["rank"]  = np.arange(1, len(resdf)+1)
        resdf["p_FDR"] = bh_fdr(resdf["p"].values, alpha=alpha)
        top = resdf.head(max_results)

        n_sig    = (resdf["p"] <= alpha).sum()
        n_fdr    = int(resdf["p_FDR"].sum())
        fdr_pct  = 100 * (1 - n_fdr / max(n_sig, 1))
        share_sig = 100 * n_sig / max(model_counter, 1)

        # ── Wall of Shame stats ─────────────────────────────────────────
        st.markdown(f"""
<div class="stat-row">
  <div class="stat-box">
    <div class="stat-val">{model_counter:,}</div>
    <div class="stat-lbl">Models tried</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{n_sig:,}</div>
    <div class="stat-lbl">p ≤ {alpha} "discoveries"</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{n_fdr}</div>
    <div class="stat-lbl">FDR survivors</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{fdr_pct:.0f}%</div>
    <div class="stat-lbl">Est. false positives</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{share_sig:.1f}%</div>
    <div class="stat-lbl">% significant</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Top results table ───────────────────────────────────────────
        st.markdown("### 🏆 Top 'discoveries' (ranked by p-value)")
        st.dataframe(
            top[["rank", "x", "y", "tfx", "tfy", "lag", "filter", "p", "R2", "n", "b1", "p_FDR"]],
            use_container_width=True,
        )

        # ── Discovery cards for top 3 ───────────────────────────────────
        st.markdown("### 📰 Your finest discoveries")
        for _, row in top.head(3).iterrows():
            fdr_class  = "disc-card fdr-pass" if row["p_FDR"] else "disc-card"
            r2_text    = f"{row['R2']:.2f}" if np.isfinite(row["R2"]) else "?"
            fdr_label  = "✅ FDR-corrected" if row["p_FDR"] else "❌ Fails FDR"
            headline   = silly_headline(row.x, row.y, row.p, row.R2, int(row.lag), row.tfx, row.tfy, row["filter"])
            st.markdown(f"""
<div class="{fdr_class}">
  <div class="disc-card-title">"{headline}"</div>
  <div class="disc-meta">
    <span class="pill pill-red">p = {row['p']:.4g}</span>
    <span class="pill pill-blue">R² = {r2_text}</span>
    <span class="pill pill-gray">n = {int(row['n'])}</span>
    <span class="pill {'pill-green' if row['p_FDR'] else 'pill-red'}">{fdr_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Fake paper for the best finding ────────────────────────────
        best          = top.iloc[0]
        n_specs_tried = model_counter
        ret_prob      = retraction_probability(best.p, best.R2, int(best.n), n_specs_tried)
        journal       = random.choice(FAKE_JOURNALS)
        authors       = random.choice(FAKE_AUTHORS)
        title         = paper_title(best.x, best.y, best.tfx, best.tfy, int(best.lag))
        abstract      = bogus_abstract(
            best.x, best.y, best.p, best.R2, int(best.n),
            int(best.lag), best.tfx, best.tfy, best["filter"], n_specs_tried,
        )
        keywords = fake_keywords(best.x, best.y, best.tfx)

        st.markdown("### 📄 Auto-generated paper")
        risk_badge = (
            f'<span class="paper-risk">RETRACTION RISK: {ret_prob}%</span>'
            if ret_prob > 65 else ""
        )
        st.markdown(f"""
<div class="paper">
  {risk_badge}
  <div class="paper-journal">
    {journal} &nbsp;·&nbsp; Vol. {random.randint(12,48)}, No. {random.randint(1,4)} (2025)
  </div>
  <div class="paper-title">{title}</div>
  <div class="paper-authors">{authors}</div>
  <div class="paper-abs-hd">Abstract</div>
  <div class="paper-abs-txt">{abstract}</div>
  <div class="paper-kw"><strong>Keywords:</strong> {keywords}</div>
</div>
""", unsafe_allow_html=True)

        # ── Scatter plot of best spec ───────────────────────────────────
        st.markdown("### 📉 Plot of the 'best' spec")

        def plot_spec(row):
            x   = df_work[row.x].values
            y   = df_work[row.y].values
            x   = TRANSFORMS[row.tfx](x) if row.tfx in TRANSFORMS else x
            y   = TRANSFORMS[row.tfy](y) if row.tfy in TRANSFORMS else y
            lag = int(row.lag)
            if lag > 0:   x, y = x[:-lag], y[lag:]
            elif lag < 0: x, y = x[-lag:], y[:lag]
            x, y = align(x, y)
            if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
                fig = go.Figure()
                fig.update_layout(title="Unable to plot: insufficient variance/obs", template="plotly_dark")
                return fig
            Xm    = sm.add_constant(x, has_constant="add")
            mod   = sm.OLS(y, Xm).fit()
            x_ord = np.sort(x)
            yhat  = mod.predict(sm.add_constant(x_ord, has_constant="add"))
            r2_text = f"{mod.rsquared:.3f}" if np.isfinite(mod.rsquared) else "NaN"
            p_text  = f"{mod.pvalues[1]:.4g}" if len(mod.pvalues) > 1 and np.isfinite(mod.pvalues[1]) else "NaN"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name="Data",
                marker=dict(color="#4299e1", opacity=0.45, size=5),
            ))
            fig.add_trace(go.Scatter(
                x=x_ord, y=yhat, mode="lines", name="OLS fit",
                line=dict(color="#e94560", width=2.5),
            ))
            fig.update_layout(
                template="plotly_dark",
                xaxis_title=f"{row.x} [{row.tfx}]",
                yaxis_title=f"{row.y} [{row.tfy}]",
                title=f"p = {p_text}  ·  R² = {r2_text}  ·  n = {int(mod.nobs)}  ·  lag = {lag}",
                height=430,
            )
            return fig

        st.plotly_chart(plot_spec(best), use_container_width=True)

        # ── Downloads ──────────────────────────────────────────────────
        st.download_button(
            "💾 Download top-K table (CSV)",
            data=top.to_csv(index=False).encode("utf-8"),
            file_name="harking_topK.csv",
            mime="text/csv",
        )

        if stop_now:
            st.warning(
                f"Early stop after {model_counter:,} specs "
                f"(≥ {good_needed} good hits or cap {max_models:,} reached)."
            )

        with st.expander("ℹ️ Multiple-testing note"):
            st.markdown(f"""
The main table does **not** correct for the massive search above.
The `p_FDR` column applies **Benjamini–Hochberg** at your chosen α to the *displayed results only*.

True FDR correction would need to account for all **{model_counter:,}** tested specs —
in which case virtually nothing survives. That's the whole lesson. 🚨
            """)

# ══════════════════════════ TAB 2: REGRESSION LAB ═══════════════════
with tab_reg:
    st.subheader("📐 OLS Regression Lab")
    st.markdown(
        "Estimate a specification properly — with full diagnostics. "
        "Use this to **validate** (or deflate) a HARKed finding."
    )

    c1, c2 = st.columns(2)
    num_cols_r = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    all_cols_r = list(df.columns)
    ycol  = c1.selectbox("Outcome (Y)", num_cols_r, index=0)
    x_sel = c2.multiselect(
        "Predictors (X)",
        [c for c in all_cols_r if c != ycol],
        default=[c for c in num_cols_r if c != ycol][:2],
    )

    c3, c4, c5 = st.columns(3)
    add_constant  = c3.checkbox("Add intercept", True)
    standardize_X = c4.checkbox("Standardize X (z-score)", False)
    dummies       = c5.checkbox("One-hot encode categoricals", True)

    if not x_sel:
        st.info("Select at least one predictor.")
    else:
        D = df[[ycol] + x_sel].copy()
        if dummies:
            cat_like = [c for c in x_sel if D[c].dtype == "object" or pd.api.types.is_categorical_dtype(D[c])]
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

        M      = pd.concat([y, X], axis=1).dropna()
        y_al   = M[ycol].values
        X_al   = M.drop(columns=[ycol]).values
        X_cols = list(M.drop(columns=[ycol]).columns)

        if M.shape[0] < len(X_cols) + 5:
            st.warning("Not enough observations after cleaning for this many predictors.")
        else:
            model = sm.OLS(y_al, X_al).fit()

            with st.expander("📋 Full statsmodels summary", expanded=False):
                st.text(model.summary().as_text())

            # ── Coefficient table & forest plot ───────────────────────
            params   = pd.Series(model.params).reset_index(drop=True)
            bse      = pd.Series(model.bse).reset_index(drop=True)
            tvals    = pd.Series(model.tvalues).reset_index(drop=True)
            pvals    = pd.Series(model.pvalues).reset_index(drop=True)
            conf_arr = model.conf_int()
            conf_df  = (conf_arr if isinstance(conf_arr, pd.DataFrame)
                        else pd.DataFrame(conf_arr, columns=["ci_low", "ci_high"]))
            conf_df  = conf_df.reset_index(drop=True)

            coeft = pd.DataFrame({
                "term": X_cols, "coef": params, "se": bse, "t": tvals, "p": pvals,
            })
            coeft = pd.concat([coeft, conf_df[["ci_low", "ci_high"]]], axis=1)

            col_tbl, col_forest = st.columns([2, 3])
            with col_tbl:
                st.markdown("### Coefficients")
                st.dataframe(coeft, use_container_width=True)

            with col_forest:
                st.markdown("### Coefficient plot")
                non_const = coeft[coeft["term"] != "const"].copy()
                if not non_const.empty:
                    colors = ["#e94560" if p <= alpha else "#4299e1" for p in non_const["p"]]
                    fig_f  = go.Figure()
                    fig_f.add_trace(go.Scatter(
                        x=non_const["coef"], y=non_const["term"],
                        mode="markers",
                        marker=dict(color=colors, size=10, symbol="diamond"),
                        error_x=dict(
                            type="data", symmetric=False,
                            array=(non_const["ci_high"] - non_const["coef"]).tolist(),
                            arrayminus=(non_const["coef"] - non_const["ci_low"]).tolist(),
                            color="#718096", thickness=2, width=6,
                        ),
                        name="Coef ± 95% CI",
                    ))
                    fig_f.add_vline(x=0, line_dash="dot", line_color="#718096")
                    fig_f.update_layout(
                        template="plotly_dark",
                        height=max(300, len(non_const) * 50 + 100),
                        xaxis_title="Coefficient", yaxis_title="",
                        showlegend=False, margin=dict(l=160),
                    )
                    st.plotly_chart(fig_f, use_container_width=True)
                    st.caption(f"🔴 p ≤ {alpha}  ·  🔵 not significant at α = {alpha}")

            # ── VIF ─────────────────────────────────────────────────
            st.markdown("### Multicollinearity (VIF)")
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                X_nc = M.drop(columns=[ycol]).copy()
                if "const" in X_nc.columns:
                    X_nc = X_nc.drop(columns=["const"])
                arr = X_nc.values
                vif_rows = []
                for i, name in enumerate(X_nc.columns):
                    v = variance_inflation_factor(arr, i)
                    vif_rows.append({"variable": name, "VIF": round(v, 2),
                                     "flag": "⚠️ high (> 10)" if v > 10 else "✅ ok"})
                st.dataframe(pd.DataFrame(vif_rows), use_container_width=True)
            except Exception as e:
                st.info(f"VIF not available: {e}")

            # ── Residual diagnostics ─────────────────────────────────
            yhat  = model.predict(X_al)
            resid = y_al - yhat
            st.markdown("### Diagnostics")

            # 1) Residuals vs Fitted
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=yhat, y=resid, mode="markers",
                                      marker=dict(color="#4299e1", opacity=0.5, size=4)))
            fig1.add_hline(y=0, line_dash="dot", line_color="#fc8181")
            fig1.update_layout(template="plotly_dark", title="Residuals vs Fitted",
                               xaxis_title="Fitted", yaxis_title="Residuals", height=320)

            # 2) QQ
            osm, _ = probplot(resid, dist="norm")[:2]
            xqq, _ = osm
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=np.sort(xqq), y=np.sort(resid), mode="markers",
                                      marker=dict(color="#90cdf4", opacity=0.6, size=4)))
            fig2.add_trace(go.Scatter(x=[xqq.min(), xqq.max()], y=[xqq.min(), xqq.max()],
                                      mode="lines", line=dict(dash="dot", color="#fc8181")))
            fig2.update_layout(template="plotly_dark", title="QQ Plot",
                               xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles", height=320)

            # 3) Scale-Location
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=yhat, y=np.sqrt(np.abs(resid)), mode="markers",
                                      marker=dict(color="#68d391", opacity=0.5, size=4)))
            fig3.update_layout(template="plotly_dark", title="Scale–Location",
                               xaxis_title="Fitted", yaxis_title="√|Residuals|", height=320)

            # 4) Leverage
            infl     = model.get_influence()
            leverage = infl.hat_matrix_diag
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=leverage, y=resid**2, mode="markers",
                                      marker=dict(color="#f6ad55", opacity=0.5, size=4)))
            fig4.update_layout(template="plotly_dark", title="Leverage vs Residual²",
                               xaxis_title="Leverage (hat diag)", yaxis_title="Residual²", height=320)

            cdiag1, cdiag2 = st.columns(2)
            cdiag1.plotly_chart(fig1, use_container_width=True)
            cdiag1.plotly_chart(fig3, use_container_width=True)
            cdiag2.plotly_chart(fig2, use_container_width=True)
            cdiag2.plotly_chart(fig4, use_container_width=True)

            pred_df = pd.DataFrame({"y": y_al, "yhat": yhat, "resid": resid})
            st.download_button(
                "💾 Download predictions & residuals (CSV)",
                data=pred_df.to_csv(index=False).encode("utf-8"),
                file_name="pred_resid.csv", mime="text/csv",
            )

# ══════════════════════════ TAB 3: DATAVIZ ═══════════════════════════
with tab_viz:
    st.subheader("📊 DataViz Studio")

    # ── Interactive correlation heatmap ───────────────────────────────
    st.markdown("### Correlation heatmap")
    num_for_corr = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_for_corr) >= 2:
        hm_vars = st.multiselect(
            "Variables to include", num_for_corr, default=num_for_corr, key="hm_vars"
        )
        if len(hm_vars) >= 2:
            corr = df[hm_vars].corr()
            figc = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1, origin="lower",
                labels=dict(color="r"),
            )
            figc.update_traces(
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>"
            )
            figc.update_layout(
                template="plotly_dark", height=500,
                coloraxis_colorbar=dict(title="Pearson r"),
            )
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.info("Select at least 2 variables.")
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

    # ── Scatter matrix ────────────────────────────────────────────────
    st.markdown("### Scatter matrix")
    pick_scatter = st.multiselect(
        "Select up to 6 columns", df.columns.tolist(),
        default=num_for_corr[:4], key="scatter_vars",
    )
    if pick_scatter:
        pick_scatter = pick_scatter[:6]
        try:
            figsm = px.scatter_matrix(
                df[pick_scatter].dropna(), dimensions=pick_scatter, opacity=0.5,
            )
            figsm.update_traces(marker=dict(size=3, color="#4299e1"))
            figsm.update_layout(template="plotly_dark", height=620)
            st.plotly_chart(figsm, use_container_width=True)
        except Exception as e:
            st.info(f"Could not draw scatter matrix: {e}")

    # ── Time-series explorer ──────────────────────────────────────────
    st.markdown("### Time-series explorer")
    df_ts, time_candidates = build_time_candidates(df)
    if time_candidates:
        tcol = st.selectbox("Time column", time_candidates, index=0, key="ts_tcol")
        tsd  = df_ts.dropna(subset=[tcol]).sort_values(tcol)
        numeric_series = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_series:
            c_y1, c_y2 = st.columns(2)
            y1 = c_y1.selectbox("Series 1", numeric_series, index=0, key="ts_y1")
            y2 = c_y2.selectbox("Series 2 (optional)", ["(none)"] + numeric_series, index=0, key="ts_y2")
            figts = go.Figure()
            figts.add_trace(go.Scatter(
                x=tsd[tcol], y=tsd[y1], mode="lines", name=y1,
                line=dict(color="#4299e1", width=1.5),
            ))
            if y2 != "(none)":
                figts.add_trace(go.Scatter(
                    x=tsd[tcol], y=tsd[y2], mode="lines", name=y2,
                    yaxis="y2", line=dict(color="#e94560", width=1.5),
                ))
                figts.update_layout(
                    yaxis2=dict(overlaying="y", side="right", showgrid=False, color="#e94560")
                )
            figts.update_layout(
                template="plotly_dark",
                xaxis_title=str(tcol), yaxis_title=y1, height=460,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(figts, use_container_width=True)
    else:
        st.info("No time-like column detected. Add a datetime or integer 'year' column.")

# ═══════════════════════════ FOOTER ══════════════════════════════════
st.markdown("---")
st.caption(
    "Made for teaching and exploration · "
    "Do not use HARKed findings for policy or clinical decisions · "
    "Kerr (1998) · Simmons et al. (2011) · Gelman & Loken (2014)"
)
