# pages/05_Auto_Article.py
import streamlit as st, textwrap, os
import pandas as pd

st.set_page_config(page_title="📝 Auto Article (Local Tiny LLM)", page_icon="📝", layout="wide")
st.title("📝 Auto-generate an Article from Your Results (Local Tiny LLM — no API)")

st.markdown("""
This page uses a **tiny local model** (GGUF via `llama-cpp-python`) downloaded on demand from **Hugging Face**.
Nothing large is committed to GitHub; the model is cached on first use.
""")

# ---- Inputs from the other pages (paste or upload) ----
c1, c2 = st.columns(2)
with c1:
    hark_csv = st.file_uploader("Top-K HARKing CSV (univariate)", type=["csv"], key="art_hark")
    multix_csv = st.file_uploader("Top-K HARKing CSV (multi-X)", type=["csv"], key="art_multix")
with c2:
    reg_text = st.text_area("Key regression notes / diagnostics", height=180)
    viz_text = st.text_area("Viz takeaways / figure captions", height=180)

title = st.text_input("Paper title", "Exploratory Patterns and Robustness in Toy Economic Data")
venue = st.text_input("Target venue style (optional)", "field-journal")
sections = st.multiselect(
    "Sections to include",
    ["Abstract","Introduction","Data","Methods","Results","Discussion","Limitations","Conclusion"],
    default=["Abstract","Introduction","Data","Methods","Results","Discussion","Conclusion"]
)

# ---- Tiny model selection & generation params ----
from utils.model_loader import get_local_gguf, model_choices
choices = model_choices()
labels = [c[0] for c in choices]
pick = st.selectbox("Local tiny model", labels, index=0)
_, model_id, model_file = choices[labels.index(pick)]

ctx = st.slider("Context length (tokens)", 1024, 8192, 4096, step=512)
temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
max_tokens = st.slider("Max new tokens", 128, 4096, 900, step=64)

st.markdown("---")

# ---- Prompt assembly (uses the CSV contents directly) ----
def csv_preview_text(file):
    if not file:
        return "(none)"
    try:
        df = pd.read_csv(file)
        # show a short preview inline in the prompt
        head_txt = df.head(10).to_csv(index=False)
        return f"(columns: {', '.join(df.columns)})\nPreview:\n{head_txt}"
    except Exception as e:
        return f"(could not parse CSV: {e})"

prompt = f"""
You are an assistant writing a short, coherent article that synthesizes four inputs:
(1) HARKING top-K results (univariate),
(2) HARKING top-K results with main regressor + controls,
(3) Regression diagnostics summary,
(4) Data visualization takeaways.

Title: {title}
Venue style: {venue}

Write concise academic prose (clear, neutral, reproducible).
Use subsection headings and short paragraphs. Avoid overclaiming.
When discussing HARKING, clearly flag multiple testing and exploratory nature.

Sections to include: {", ".join(sections)}

== UNIVARIATE HARKING (parsed) ==
{csv_preview_text(hark_csv)}

== MULTI-X HARKING (parsed) ==
{csv_preview_text(multix_csv)}

== REGRESSION NOTES ==
{reg_text}

== VIZ NOTES ==
{viz_text}
"""

with st.expander("🔍 View prompt", expanded=False):
    st.code(prompt, language="markdown")

# ---- Generate (local tiny LLM) ----
run = st.button("Generate Article Draft (local model)")

if run:
    with st.spinner("Downloading tiny model (first run) & generating…"):
        try:
            gguf_path = get_local_gguf(model_id=model_id, filename=model_file)
        except Exception as e:
            st.error(f"Could not fetch model from Hugging Face: {e}")
            st.stop()

        # llama-cpp in-process inference (no external binary)
        try:
            from llama_cpp import Llama
        except Exception as e:
            st.error(f"llama-cpp-python is not available: {e}")
            st.info("On Streamlit Cloud, this sometimes fails to build. You can run this page locally instead.")
            st.stop()

        try:
            llm = Llama(model_path=gguf_path, n_ctx=ctx, seed=123, n_threads=os.cpu_count())
            # Basic chat-style system/user prompt; adjust for the specific instruct format if needed
            full_prompt = f"<s>[INST]You are a careful academic writing assistant. {prompt}[/INST]"
            out = llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=0.95,
                repeat_penalty=1.1,
            )
            text = out["choices"][0]["text"]
            st.markdown("### Draft")
            st.write(text)
            st.download_button("💾 Download draft (txt)", data=text, file_name="draft.txt")
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.info("Fallback: run locally, reduce context length, or try an even smaller quant.")
