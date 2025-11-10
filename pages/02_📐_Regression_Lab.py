import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from scipy.stats import probplot
import statsmodels.api as sm
from utils.data import simulate_data, read_csv_safely

st.title("📐 Regression Lab")
st.sidebar.header("Data")
mode = st.sidebar.radio("Choose data", ["Simulate", "Upload CSV"], horizontal=True, key="reg_mode")
if mode == "Simulate":
    n = st.sidebar.slider("Rows", 200, 5000, 800, step=100, key="reg_rows")
    k = st.sidebar.slider("Readable variables", 3, 20, 8, step=1, key="reg_k")
    df = simulate_data(n=n, k=k, with_time=True, readable_names=True)
else:
    up = st.sidebar.file_uploader("CSV file", type=["csv"], key="reg_file")
    if up is None: st.stop()
    try:
        df = read_csv_safely(up)
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); st.stop()

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(num_cols) < 2: st.error("Need at least two numeric columns."); st.stop()

c1, c2 = st.columns(2)
ycol = c1.selectbox("Outcome (y)", num_cols, index=0)
x_candidates = [c for c in df.columns if c != ycol]
x_sel = c2.multiselect("Predictors (X)", x_candidates,
                       default=[c for c in num_cols if c != ycol][:2])

c3, c4, c5 = st.columns(3)
add_constant  = c3.checkbox("Add intercept", True)
standardize_X = c4.checkbox("Standardize X", False, help="z-score numeric predictors")
dummies       = c5.checkbox("One-hot encode categoricals", True)

if not x_sel:
    st.info("Select at least one predictor."); st.stop()

D = df[[ycol] + x_sel].copy()
if dummies:
    cat_like = [c for c in x_sel if (D[c].dtype == "object" or pd.api.types.is_categorical_dtype(D[c]))]
    if cat_like: D = pd.get_dummies(D, columns=cat_like, drop_first=True)

y = D[ycol].astype(float); X = D.drop(columns=[ycol])
if standardize_X:
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            mu, sd = X[c].mean(), X[c].std(ddof=0)
            if sd > 0: X[c] = (X[c] - mu) / sd
if add_constant: X = sm.add_constant(X, has_constant="add")

M = pd.concat([y, X], axis=1).dropna()
y_al = M[ycol].values; X_al = M.drop(columns=[ycol]).values; X_cols = list(M.drop(columns=[ycol]).columns)
if M.shape[0] < len(X_cols) + 5:
    st.warning("Not enough observations after cleaning for this many predictors."); st.stop()

model = sm.OLS(y_al, X_al).fit()
st.markdown("### Model summary")
st.text(model.summary().as_text())

# Coefs + CI
params = pd.Series(model.params).reset_index(drop=True)
bse    = pd.Series(model.bse).reset_index(drop=True)
tvals  = pd.Series(model.tvalues).reset_index(drop=True)
pvals  = pd.Series(model.pvalues).reset_index(drop=True)
conf_arr = model.conf_int()
conf_df  = (conf_arr if isinstance(conf_arr, pd.DataFrame)
            else pd.DataFrame(conf_arr, columns=["ci_low","ci_high"])).reset_index(drop=True)
coeft = pd.DataFrame({"term": X_cols, "coef": params, "se": bse, "t": tvals, "p": pvals})
coeft = pd.concat([coeft, conf_df[["ci_low","ci_high"]]], axis=1)
st.markdown("### Coefficients")
st.dataframe(coeft, use_container_width=True, hide_index=True)

# VIF
st.markdown("### Multicollinearity (VIF)")
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_noconst = M.drop(columns=[ycol]).copy()
    if "const" in X_noconst.columns: X_noconst = X_noconst.drop(columns=["const"])
    arr = X_noconst.values
    vif_df = [{"variable": nm, "VIF": variance_inflation_factor(arr, i)} for i, nm in enumerate(X_noconst.columns)]
    st.dataframe(pd.DataFrame(vif_df), use_container_width=True, hide_index=True)
except Exception as e:
    st.info(f"VIF not available: {e}")

# Diagnostics
yhat = model.predict(X_al); resid = y_al - yhat
st.markdown("### Diagnostics")
fig1 = go.Figure(); fig1.add_trace(go.Scatter(x=yhat, y=resid, mode="markers", opacity=0.65))
fig1.add_hline(y=0, line_dash="dot"); fig1.update_layout(title="Residuals vs Fitted",
    xaxis_title="Fitted values", yaxis_title="Residuals", height=420)
from scipy.stats import probplot; osm, osr = probplot(resid, dist="norm")[:2]; xqq,_=osm
fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=np.sort(xqq), y=np.sort(resid), mode="markers"))
fig2.add_trace(go.Scatter(x=[np.min(xqq), np.max(xqq)], y=[np.min(resid), np.max(resid)],
                          mode="lines", line=dict(dash="dot"))); fig2.update_layout(title="QQ Plot",
                          xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles", height=420)
fig3 = go.Figure(); fig3.add_trace(go.Scatter(x=yhat, y=np.sqrt(np.abs(resid)), mode="markers", opacity=0.65))
fig3.update_layout(title="Scale–Location", xaxis_title="Fitted values", yaxis_title="√|Residuals|", height=420)
infl = model.get_influence(); leverage = infl.hat_matrix_diag
fig4 = go.Figure(); fig4.add_trace(go.Scatter(x=leverage, y=resid**2, mode="markers", opacity=0.65))
fig4.update_layout(title="Leverage vs Residuals²", xaxis_title="Leverage (hat diag)", yaxis_title="Residual²", height=420)
c1, c2 = st.columns(2); c1.plotly_chart(fig1, use_container_width=True); c1.plotly_chart(fig3, use_container_width=True)
c2.plotly_chart(fig2, use_container_width=True); c2.plotly_chart(fig4, use_container_width=True)

pred_df = pd.DataFrame({"y": y_al, "yhat": yhat, "resid": resid})
st.download_button("Download predictions & residuals (CSV)", data=pred_df.to_csv(index=False).encode("utf-8"),
                   file_name="pred_resid.csv", mime="text/csv")
