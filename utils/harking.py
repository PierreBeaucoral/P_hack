import numpy as np, pandas as pd, statsmodels.api as sm, streamlit as st

def detrend_linear(s):
    s = np.asarray(s, float); idx = np.arange(len(s)); ok = np.isfinite(s)
    if ok.sum() < 3: return s
    b = np.polyfit(idx[ok], s[ok], 1); trend = b[0]*idx + b[1]
    return s - trend

TRANSFORMS = {
    "identity":     lambda s: s,
    "zscore":       lambda s: (np.asarray(s, float)-np.nanmean(s))/(np.nanstd(s)+1e-12),
    "log1p|signed": lambda s: np.sign(s) * np.log1p(np.abs(s)),
    "diff":         lambda s: pd.Series(s).diff().values,
    "pct_change":   lambda s: pd.Series(s).pct_change().replace([np.inf,-np.inf],np.nan).values,
    "rollmean_3":   lambda s: pd.Series(s).rolling(3, min_periods=1).mean().values,
    "rollmean_6":   lambda s: pd.Series(s).rolling(6, min_periods=1).mean().values,
    "sqrt|signed":  lambda s: np.sign(s) * np.sqrt(np.abs(s)),
    "square":       lambda s: np.square(s),
    "rank":         lambda s: pd.Series(s).rank(pct=True).values,
    "detrend_linear": detrend_linear,
}

def align(a,b):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]

def regress_simple(x, y):
    if len(x) < 3 or len(y) < 3: return np.nan, np.nan, 0, np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0: return np.nan, np.nan, 0, np.nan
    X = sm.add_constant(x, has_constant="add")
    try:
        model = sm.OLS(y, X, missing='drop').fit()
        pv = model.pvalues[1] if len(model.params)>1 else np.nan
        r2 = model.rsquared
        n  = int(model.nobs)
        b1 = model.params[1] if len(model.params)>1 else np.nan
        return pv, r2, n, b1
    except Exception:
        return np.nan, np.nan, 0, np.nan

def make_bins(x, q=4):
    s = pd.Series(x)
    try: return pd.qcut(s, q=q, duplicates="drop")
    except Exception: return pd.cut(s, bins=q)

@st.cache_data(show_spinner=False)
def run_harking(df: pd.DataFrame,
                Ycands: list, Xcands: list,
                transforms: list, max_lag: int,
                do_bins: bool, min_r2: float,
                progress_every: int = 2000) -> pd.DataFrame:
    """Return results DataFrame with columns: x,y,tfx,tfy,lag,filter,p,R2,n,b1"""
    filters, filt_labels = [None], [""]
    if do_bins:
        for col in Xcands[:min(3, len(Xcands))]:
            bins4 = make_bins(df[col], q=4); bins10 = make_bins(df[col], q=10)
            for series, label in [(bins4, f"{col} in quartile q"), (bins10, f"{col} in decile d")]:
                if hasattr(series, "codes"):
                    codes = pd.Series(series.cat.codes).values
                    uniq = np.unique(codes[codes>=0])[:4]
                    for code in uniq:
                        filters.append((codes == code)); filt_labels.append(label.replace("q", str(code+1)).replace("d", str(code+1)))

    results = []
    counter = 0
    for yname in Ycands:
        for xname in Xcands:
            if xname == yname: continue
            x_raw = df[xname].values; y_raw = df[yname].values
            for lag in range(-max_lag, max_lag+1):
                for tfx in transforms:
                    for tfy in transforms:
                        x = TRANSFORMS[tfx](x_raw) if tfx in TRANSFORMS else x_raw
                        y = TRANSFORMS[tfy](y_raw) if tfy in TRANSFORMS else y_raw
                        if lag > 0: x, y = x[:-lag], y[lag:]
                        elif lag < 0: x, y = x[-lag:], y[:lag]
                        for F, Flab in zip(filters, filt_labels):
                            if F is not None:
                                x1, y1 = x[F], y[F]
                            else:
                                x1, y1 = x, y
                            x1, y1 = align(x1, y1)
                            pv, r2, n, b1 = regress_simple(x1, y1)
                            if not np.isfinite(pv): continue
                            if np.isfinite(r2) and (r2 < min_r2): continue
                            results.append(dict(x=xname,y=yname,lag=lag,tfx=tfx,tfy=tfy,filter=Flab,
                                                p=float(pv),R2=float(r2),n=int(n),b1=float(b1)))
                            counter += 1
                            if counter % progress_every == 0:
                                st.session_state["hark_counter"] = counter  # cheap progress hook
    return pd.DataFrame(results)

def bh_fdr(pvals, alpha=0.05):
    ps = np.array(pvals); order = np.argsort(ps); ranked = ps[order]; m = float(len(ps))
    thresh = (np.arange(1, len(ps)+1) / m) * alpha
    passed = ranked <= thresh
    if not passed.any(): return np.full_like(ps, False, dtype=bool)
    kmax = np.max(np.where(passed)); cutoff = ranked[kmax]
    return ps <= cutoff
