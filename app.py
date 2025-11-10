import streamlit as st
from utils.style import apply_global_style, hero

skip_css = "nocss" in st.query_params  # visit ?nocss=1 to disable custom CSS
if not skip_css:
    st.markdown("""<style> ... your CSS ... </style>""", unsafe_allow_html=True)

st.set_page_config(page_title="HARKing + Regression Lab", page_icon="📈", layout="wide")
apply_global_style()

hero("📈 HARKing + Regression Lab",
     "**A professional sandbox for exploring correlations, validating models, and visualizing data.** "
     "Use the sidebar to select data on each page. The pages menu is at the left (or top on mobile).")

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card">**1) 🎣 HARKing**</div>', unsafe_allow_html=True)
        st.caption("Brute-force Y~X with transforms, lags, bin-filters; min R²; early-stop; rank by p.")
    with c2:
        st.markdown('<div class="card">**2) 📐 Regression Lab**</div>', unsafe_allow_html=True)
        st.caption("Choose outcome/predictors; dummies/standardization; coefficients + CIs; VIF; diagnostics.")
    with c3:
        st.markdown('<div class="card">**3) 📊 DataViz Studio**</div>', unsafe_allow_html=True)
        st.caption("Correlation heatmap, scatter matrix, time-series explorer (supports integer year).")

st.sidebar.info(
    "Quick start:\n"
    "• Open **🎣 HARKing** → keep defaults → run.\n"
    "• Validate in **📐 Regression Lab**.\n"
    "• Explore **📊 DataViz**."
)

st.markdown("---")
st.caption("Made for teaching and exploration. Don’t use HARKed findings for policy/clinical decisions.")
