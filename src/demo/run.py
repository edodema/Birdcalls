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
st.sidebar.title("Birdcalls")
run = st.button("Run")

# Disclaimer.
message = "Given an audio file the model considers a 5 seconds window and tries to guess the bird singing, if there is one."
st.sidebar.write(message)


# Get file path.
csv_path = cfg.demo.csv

# Get a random audio sample from the set.
sample = get_sample(csv_path=csv_path)
path, start_time, spectrogram, target = get_tensor(sample=sample)

# Get audio.
audio_file = open(path, "rb")
audio_bytes = audio_file.read()
st.audio(audio_bytes, format="audio/ogg", start_time=start_time)

# Draw spectrogram.
fig = draw_spectrogram(spectrogram)
st.pyplot(fig)

# Display target class.
st.write(f"Truth: {translate_detection(target)}")

# Prediction.
model = get_model(cfg=cfg)

_, pred = model(spectrogram)
pred = get_prediction(pred)

st.write(f"Prediction: {translate_detection(target)}")
