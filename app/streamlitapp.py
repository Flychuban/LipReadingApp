import streamlit as st
import os
import imageio
import tensorflow as tf

from utils import load_data, num_to_char
from modelutil import load_model
from newmodelutil import LipNet

st.set_page_config(layout="wide")

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipReader')
    st.info("LipReader is a deep learning model that can read lips from video frames. It is trained on the LRW dataset and can recognize evert human sound.")

st.title('LipReader App')
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Select a video', options)


col1, col2 = st.columns(2)
if options:
    with col1:
        st.info('This video below displays the convertd video in mp4 format.')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations, frames_shape = load_data(tf.convert_to_tensor(file_path))
        frames_n = frames_shape[0]
        width = frames_shape[1]
        height = frames_shape[2]
        video_channel = frames_shape[3]
        imageio.mimsave('animation_streamlit.gif', video, fps=10)
        st.image('animation_streamlit.gif', width=400)
        
        st.info('This is the output of the machine learning model as tokens')
        # model = load_model(width, height)
        model = LipNet(frames_n, width, height, video_channel)
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0]
        st.text(decoder)
        
        st.info('Decode the raw tokens into words')
        tranlated_text = num_to_char(decoder)
        tranlated_text = tf.strings.reduce_join(tranlated_text).numpy().decode('utf-8')
        st.text(tranlated_text)
    
    # For future improvements detect not only mpg but dynamic
    # Dynamic detection of lips