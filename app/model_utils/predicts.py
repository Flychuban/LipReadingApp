from .implementation_video import Video
from .visualization import show_video_subtitle
from .decoder import Decoder
from .translations import labels_to_text
from .spell import Spell
from .model import LipNet
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import sys
import os
import glob
import tensorflow as tf

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'grid.txt')


def predict(weight_path, video_path, absolute_max_string_len=32, output_size=28):
    print("Loading LipNet model in memory...")
    
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print("Data loaded")

    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape
    
    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                        postprocessors=[labels_to_text, spell.sentence])

    X_data       = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])

    y_pred         = lipnet.predict(X_data)
    result         = decoder.decode(y_pred, input_length)[0]

    show_video_subtitle(video.face, result)
    print(f"Result: {result}")

# def predicts(weight_path, video_path, absolute_max_string_len=32, output_size=28):
#     video = load(video_path)
#     predict(weight_path, video)

# def load(video_path):
#     print(f"\n[{video_path}]\nLoading data from disk...")
#     video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
#     if os.path.isfile(video_path):
#         video.from_video(video_path)
#     else:
#         video.from_frames(video_path)
#     print("Data loaded")
#     return video
