import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import transformers

st.title("✅ Streamlit Environment Test for DeepCSAT")

st.write("This app verifies that all ML/DL dependencies install correctly on Streamlit Cloud.")

st.subheader("📦 Package Versions")
st.write({
    "streamlit": st.__version__,
    "tensorflow": tf.__version__,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__
})

st.success("🎉 All core libraries imported successfully!")
