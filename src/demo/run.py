import streamlit as st
import torch.nn
from omegaconf import DictConfig
from src.pl_modules.module import (
    JointClassification,
)
from src.demo.utils import (
    get_hydra_cfg,
    get_sample,
    get_tensor,
    draw_spectrogram,
    get_prediction,
    translate_detection,
)


@st.cache(allow_output_mutation=True)
def get_model(cfg: DictConfig) -> torch.nn.Module:
    """
    Load a pretrained model.
    Args:
        cfg: Hydra DictConfig.

    Returns:
        A list of loaded models for the corresponding model.
    """
    # Get models paths.
    checkpoint = cfg.demo.checkpoint

    model = JointClassification.load_from_checkpoint(checkpoint_path=checkpoint)
    model.eval()
    return model


cfg = get_hydra_cfg()

# Basic UI elements.
st.set_page_config(page_title="Birdcalls - Streamlit", page_icon=":bird:")
st.sidebar.title("Birdcalls :bird:")

# # Condense the layout
# padding = 0
# st.markdown(
#     f""" <style>
#     .reportview-container .main .block-container{{
#         padding-top: {padding}rem;
#         padding-right: {padding}rem;
#         padding-left: {padding}rem;
#         padding-bottom: {padding}rem;
#     }} </style> """,
#     unsafe_allow_html=True,
# )

# Disclaimer.
message = """
This simple demo randomly selects a 5 seconds ambient recording and if a bird is singing guesses its species.
The code is available <a href="https://github.com/edodema/Birdcalls">here</a>.
"""
st.sidebar.markdown(message, unsafe_allow_html=True)

# Get file path.
csv_path = cfg.demo.csv

# Get a random audio sample from the set.
sample = get_sample(csv_path=csv_path)
path, start_time, spectrogram, target = get_tensor(sample=sample)

# Get audio.
audio_file = open(path, "rb")
audio_bytes = audio_file.read()
# st.audio(audio_bytes, format="audio/ogg", start_time=start_time)

col1, col2 = st.beta_columns([1, 10])
with col2:
    st.audio(audio_bytes, format="audio/ogg", start_time=start_time)
with col1:
    st.button("Run")

# Draw spectrogram.
fig = draw_spectrogram(spectrogram)
st.plotly_chart(fig, use_container_width=True)

# Prediction.
model = get_model(cfg=cfg)

_, pred = model(spectrogram)
pred = get_prediction(pred)

# Display target and prediction class.
truth = translate_detection(target)
pred = translate_detection(pred)

col1, col2 = st.beta_columns(2)

with col1:
    st.header("Label")
    st.subheader(truth)

with col2:
    st.header("Prediction")
    st.subheader(pred)

col1, col2, col3 = st.beta_columns(3)

# Output.
with col1:
    st.write("")

with col2:
    if truth == pred:
        st.header(":white_check_mark:")
    else:
        st.header(":x:")


with col3:
    st.write("")
