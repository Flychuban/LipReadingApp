import tensorflow as tf
from typing import List
import cv2
import os 
import numpy as np
import dlib
import skvideo.io
from keras import backend as K

face = None
mouth = None
face_predictor_path = r"C:\Users\sas\Desktop\LipReadingApp\app\model_utils\shape_predictor_68_face_landmarks.dat"


def get_frames_mouth(detector, predictor, frames):
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None
    mouth_frames = []
    for frame in frames:
        dets = detector(frame, 1)
        shape = None
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            i = -1
        if shape is None: # Detector doesn't detect face, just return as is
            return frames
        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48: # Only take mouth region
                continue
            mouth_points.append((part.x,part.y))
        np_mouth_points = np.array(mouth_points)

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        resized_img = cv2.resize(frame, new_img_shape)

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)
        # print(f"Mouth left: {mouth_l}")
        # print(f"Mouth right: {mouth_r}")
        # print(f"Mouth top: {mouth_t}")
        # print(f"Mouth bottom: {mouth_b}")
        
        mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

        mouth_frames.append(mouth_crop_image)
    return mouth_frames


def process_frames_face(frames, face_predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    mouth_frames = get_frames_mouth(detector, predictor, frames)
    face = np.array(frames)
    mouth = np.array(mouth_frames)
    set_data(mouth_frames)
    return mouth_frames
    
def set_data(frames):
    data_frames = []
    for frame in frames:
        frame = frame.swapaxes(0,1) # swap width and height to form format W x H x C
        if len(frame.shape) < 3:
            frame = np.array([frame]).swapaxes(0,2).swapaxes(0,1) # Add grayscale channel
        data_frames.append(frame)
    frames_n = len(data_frames)
    data_frames = np.array(data_frames) # T x W x H x C
    if K.image_data_format() == 'channels_first':
        data_frames = np.rollaxis(data_frames, 3) # C x T x W x H
    data = data_frames
    length = frames_n
    return data


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]: 
    videogen = skvideo.io.vreader(path)
    frames = np.array([frame for frame in videogen])
    frames_lips = np.array(process_frames_face(frames, face_predictor_path))
    
    # cap = cv2.VideoCapture(path)
    # frames = []
    # for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
    #     ret, frame = cap.read()
    #     frame = tf.image.rgb_to_grayscale(frame)
    #     frames.append(frame[190:236,80:220,:])
    # cap.release()
    # for frame in frames_lips:
    #     frame = tf.image.rgb_to_grayscale(frame)

    # print(f"Frames shape:{frames_lips.shape}")
    # print(f"Frames: {frames_lips}")
    mean = tf.math.reduce_mean(frames_lips)
    print(mean)
    std = tf.math.reduce_std(tf.cast(frames_lips, tf.float32))
    return tf.cast((frames_lips - mean), tf.float32) / std
    
def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path)
    print(f"Frames: {frames}")
    frames_shape = frames.shape
    print(f"Frames shape: {frames_shape}")
    alignments = load_alignments(alignment_path)
    
    return frames, alignments, frames_shape


# if __name__ == '__main__':
#     load_data(frames_path)