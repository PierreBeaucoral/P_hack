import plotly.io as pio
import streamlit as st

def apply_global_style():
    # Page config (do once, from app.py)
    # st.set_page_config(page_title="HARKing + Regression Lab", page_icon="📈", layout="wide")

    # Plotly defaults
    if "pro_theme" not in pio.templates:
        pio.templates["pro_theme"] = pio.templates["plotly_white"]
        pio.templates["pro_theme"].layout.colorway = [
            "#2563EB", "#0EA5E9", "#22C55E", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"
        ]
        pio.templates["pro_theme"].layout.font.family = (
            "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, "
            "Ubuntu, Cantarell, Helvetica Neue, Arial, sans-serif"
        )
        pio.templates["pro_theme"].layout.font.size = 14
    pio.templates.default = "pro_theme"

    st.markdown("""
    <style>
    .hero { padding: 1rem 1.25rem; background: linear-gradient(135deg,#eef3ff 0%,#f7fafc 100%);
            border: 1px solid #e6ecf8; border-radius: 14px; margin-bottom: 1rem; }
    .card { padding: 1rem 1.1rem; background: #fff; border: 1px solid #E6ECF8;
            border-radius: 14px; box-shadow: 0 1px 2px rgba(16,24,40,.04); margin-bottom: 1rem; }
    .stDataFrame tbody tr:hover { background-color: #F6F8FB !important; }
    .stButton>button, .stDownloadButton>button {
      border-radius: 10px !important; padding: .6rem 1rem !important; border: 1px solid #DCE3F0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def hero(title: str, subtitle_md: str):
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    st.markdown(subtitle_md)
    st.markdown('</div>', unsafe_allow_html=True)
