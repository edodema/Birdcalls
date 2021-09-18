from typing import List

import streamlit as st
import torch.nn
from omegaconf import DictConfig
from src.pl_modules.module import (
    SoundscapeDetection,
    BirdcallClassification,
    JointClassification,
)
from src.demo.utils import (
    MODES,
    get_hydra_cfg,
    get_sample,
    get_tensor,
    get_csv_path,
    draw_spectrogram,
    get_prediction,
    translate_detection,
)


@st.cache(allow_output_mutation=True)
def get_model(cfg: DictConfig, mode: str) -> List[torch.nn.Module]:
    """
    Load a pretrained model.
    Args:
        cfg: Hydra DictConfig.
        mode: The mode we have selected.

    Returns:
        A list of loaded models for the corresponding model.
    """
    # Get models paths.
    checkpoint_detection = cfg.demo.checkpoint.detection
    checkpoint_classification = cfg.demo.checkpoint.classification
    checkpoint_joint = cfg.demo.checkpoint.joint

    if mode == cfg.demo.mode.soundscapes:
        model = SoundscapeDetection.load_from_checkpoint(
            checkpoint_path=checkpoint_detection
        )
        model.eval()
        return [model]

    elif mode == cfg.demo.mode.birdcalls:
        model = BirdcallClassification.load_from_checkpoint(
            checkpoint_path=checkpoint_classification
        )
        model.eval()
        return [model]

    elif mode == cfg.demo.mode.split:
        # This time we have two models.
        detection = SoundscapeDetection.load_from_checkpoint(
            checkpoint_path=checkpoint_detection
        )
        classification = BirdcallClassification.load_from_checkpoint(
            checkpoint_path=checkpoint_classification
        )
        detection.eval()
        classification.eval()
        return [detection, classification]

    elif mode == cfg.demo.mode.joint:
        model = JointClassification.load_from_checkpoint(
            checkpoint_path=checkpoint_joint
        )
        model.eval()
        return [model]


cfg = get_hydra_cfg()

# Basic UI elements.
st.sidebar.title("Birdcalls")
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

# TODO: Add the disclaimer on the -

# Get file path.
csv_path = get_csv_path(mode)
if not csv_path:
    st.stop()

# Get a random audio sample from the set.
sample = get_sample(csv_path=csv_path)
path, start_time, spectrogram, target = get_tensor(sample=sample, mode=mode)

# Get audio.
audio_file = open(path, "rb")
audio_bytes = audio_file.read()
st.audio(audio_bytes, format="audio/ogg", start_time=start_time)

# Draw spectrogram.
fig = draw_spectrogram(spectrogram)
st.pyplot(fig)

# Display target class.
st.write(f"Truth: {translate_detection(target, mode=mode)}")

# Prediction.
models = get_model(cfg=cfg, mode=mode)

if mode == cfg.demo.mode.split:
    st.write("TODO")
else:
    model = models[0]
    _, pred = model(spectrogram)
    pred = get_prediction(pred, mode=mode)
    st.write(f"Prediction: {pred}")
