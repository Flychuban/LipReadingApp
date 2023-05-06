import streamlit as st
import os
import imageio
import tensorflow as tf

from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout="wide")

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipReader')
    st.info("LipReader is a deep learning model that can read lips from video frames. It is trained on the LRW dataset and can recognize evert human sound.")

options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Select a video', options)


col1, col2 = st.columns(2)
if options:
    with col1:
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
    with col2:
        st.text('column 2')