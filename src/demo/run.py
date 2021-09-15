import numpy as np
import streamlit as st
import hydra
import torch
from librosa import power_to_db
from src.demo.utils import MODES, get_hydra_cfg, get_sample, get_tensor, get_csv_path

cfg = get_hydra_cfg()

# Basic UI elements.
mode = st.sidebar.selectbox(
    "Select which model to try.",
    (
        MODES["empty"],
        MODES["soundscapes"],
        MODES["birdcalls"],
        MODES["split"],
        MODES["joint"],
    ),
)
run = st.sidebar.button("Run")
file_uploader = st.sidebar.file_uploader("File upload")

# Get file path.
csv_path = get_csv_path(mode)
if not csv_path:
    st.stop()

# Get a random audio sample from the set.
sample = get_sample(csv_path=csv_path)
spectrogram, target = get_tensor(sample=sample, mode=mode)

spec = np.transpose(power_to_db(spectrogram[0].numpy()), axes=(1, 2, 0))
spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
st.image(spec)

# TODO: images are pretty bad, add audio https://discuss.streamlit.io/t/audio-display/7806
