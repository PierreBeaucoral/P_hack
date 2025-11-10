import numpy as np, pandas as pd, streamlit as st

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

FRIENDLY_VARS = [
    "household_income","inflation_rate","unemployment_rate","policy_rate",
    "industrial_output","consumer_confidence","housing_prices","exports_value",
    "imports_value","net_trade","exchange_rate","energy_price","co2_emissions",
    "tourism_arrivals","public_spending","private_investment","bank_credit",
    "stock_index","wage_index","productivity_index"
]

def _pick_friendly_names(k: int):
    base = FRIENDLY_VARS.copy()
    rng.shuffle(base)
    if k <= len(base): return base[:k]
    extra, i = [], 1
    while len(base) + len(extra) < k:
        extra.extend([f"{name}_{i}" for name in FRIENDLY_VARS]); i += 1
    return base + extra[:k]

@st.cache_data(show_spinner=False)
def simulate_data(n=800, k=8, with_time=True, readable_names=True) -> pd.DataFrame:
    names = _pick_friendly_names(k) if readable_names else [f"X{i+1}" for i in range(k)]
    X = {names[i]: rng.normal(0, 1, n) for i in range(k)}
    if with_time:
        t = pd.date_range("2010-01-01", periods=n, freq="D")
        X["time"] = np.arange(n); X["month"] = pd.to_datetime(t).month; X["date"] = t
    df = pd.DataFrame(X)
    # outcomes
    df["target_null"]   = rng.normal(0,1,n)
    x1, x2 = df[names[0]].values, df[names[1]].values
    df["target_policy"] = 0.3*x1 - 0.2*(x2 > 0).astype(int) + rng.normal(0,1,n)
    df["target_trendy"] = rng.normal(0,1,n).cumsum() + rng.normal(0,5,n)
    return df

@st.cache_data(show_spinner=False)
def read_csv_safely(file) -> pd.DataFrame:
    return pd.read_csv(file)

def build_time_candidates(df: pd.DataFrame):
    """Add <col>__as_date for year/date/time columns; return (df_augmented, time_col_names)."""
    df = df.copy()
    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in df.columns:
        name = str(c).lower()
        # YEAR → datetime (Jan 1)
        if "year" in name and (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c])):
            ser = pd.to_numeric(df[c], errors="coerce")
            if (ser.between(1800, 2100)).mean() >= 0.8:
                cname = f"{c}__as_date"
                df[cname] = pd.to_datetime(ser.astype("Int64").astype(str), format="%Y", errors="coerce")
                time_cols.append(cname)
        # DATE/TIME strings
        if ("date" in name) or ("time" in name):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() >= 0.6:
                cname = f"{c}__as_date"
                df[cname] = parsed
                time_cols.append(cname)
    # unique order
    seen, uniq = set(), []
    for c in time_cols:
        if c not in seen: uniq.append(c); seen.add(c)
    return df, uniq
