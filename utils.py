import torch
import numpy as np
import cv2
import tempfile, base64
import streamlit as st
import os


def readb64(uri):
    encoded_data = uri.split(',')[-1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def img2base64(img, extension="jpg"):
    _, img_encoded = cv2.imencode(f".{extension}", img)
    img_base64 = base64.b64encode(img_encoded)
    img_base64 = img_base64.decode('utf-8')
    return img_base64

def binary2video(video_binary):
    temp_ = tempfile.NamedTemporaryFile(suffix='.mp4')

    temp_.write(video_binary)
    video_capture = cv2.VideoCapture(temp_.name)
    ret, frame = video_capture.read()
    return video_capture

def extract_frames(data_path, interval=30, max_frames=50):
    """Method to extract frames"""
    cap = cv2.VideoCapture(data_path)
    frame_num = 0
    frames = list()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        if frame_num % interval == 0:
            frames.append(image)
        frame_num += 1
        if len(frames) > max_frames:
            break
    cap.release()
    return frames

def update_dir(key):
    choice = st.session_state[key]
    if os.path.isdir(os.path.join(st.session_state[key+'curr_dir'], choice)):
        st.session_state[key+'curr_dir'] = os.path.normpath(os.path.join(st.session_state[key+'curr_dir'], choice))
        files = sorted(os.listdir(st.session_state[key+'curr_dir']))
        if "images" in files:
          files.remove("images")
        st.session_state[key+'files'] = files

def st_file_selector(st_placeholder, path='.', label='Select a file/folder', key = 'selected'):
    if key+'curr_dir' not in st.session_state:
        base_path = '.' if path is None or path == '' else path
        base_path = base_path if os.path.isdir(base_path) else os.path.dirname(base_path)
        base_path = '.' if base_path is None or base_path == '' else base_path

        files = sorted(os.listdir(base_path))
        files.insert(0, 'Choose a file...')
        if "images" in files:
          files.remove("images")
        st.session_state[key+'files'] = files
        st.session_state[key+'curr_dir'] = base_path
    else:
        base_path = st.session_state[key+'curr_dir']

    selected_file = st_placeholder.selectbox(label=label, 
                                        options=st.session_state[key+'files'], 
                                        key=key, 
                                        on_change = lambda: update_dir(key))
    
    if selected_file == "Choose a file...":
        return None

    return selected_file