import sys
# sys.path.extend(["/home/edo/Documents/Code/Birdcalls"])

import streamlit as st
# from src.demo.utils import get_sample_joint

# # Constants.
# EMPTY_MODE = "-"
# SPLIT_MODE = "Split"
# JOINT_MODE = "Joint"
#
# # Basic UI elements.
# # Sidebar.
# mode = st.sidebar.selectbox(
#     "Select which model to try.", (EMPTY_MODE, SPLIT_MODE, JOINT_MODE)
# )
# run = st.sidebar.button("Run")
# file_uploader = st.sidebar.file_uploader("File upload")
#
# if mode == JOINT_MODE:
#     # Get random audio sample from validation set.
#     x = get_sample_joint()
#     st.write(x)

if __name__ == "__main__":
    print(sys.path)
