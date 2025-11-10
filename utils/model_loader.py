# utils/model_loader.py
import os
from huggingface_hub import hf_hub_download
import streamlit as st

# Choose a tiny instruct model. Swap to another GGUF if you prefer.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
# Pick a specific quant file present in that repo (browse the HF repo files):
DEFAULT_MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"  # small & decent; change if needed

@st.cache_resource(show_spinner=False)
def get_local_gguf(model_id: str = DEFAULT_MODEL_ID, filename: str = DEFAULT_MODEL_FILE) -> str:
    """
    Download a GGUF once and cache its local path. NOTHING is committed to GitHub.
    Returns local filesystem path to the GGUF.
    """
    local_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=os.path.join(os.path.expanduser("~"), ".cache", "tiny-gguf"),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    return local_path

def model_choices():
    # You can add more tiny models here if you want a dropdown in the UI.
    return [
        ("Qwen2.5-0.5B-Instruct (q4_k_m)", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        # Example alt:
        # ("Llama 3.2-1B Instruct (q4_k_m)", "bartowski/Meta-Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        # ("TinyLlama 1.1B Chat (q4_k_m)", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    ]
