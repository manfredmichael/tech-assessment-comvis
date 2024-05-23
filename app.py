import base64
import json
import os, shutil
import re
import time
import uuid

import cv2

import numpy as np
import streamlit as st
from PIL import Image
# from extract_video import extract_method_single_video

from utils import st_file_selector, img2base64
from pipelines import ImagePipeline, VideoPipeline

import os

DEBUG = True

def main():
    st.markdown("###")    
    uploaded_file = st.file_uploader('Upload a picture', type=['mp4', 'jpg', 'jpeg', 'png'], accept_multiple_files=False)

    with st.spinner(f'Loading samples...'):
        while not os.path.isdir("sample_files"):
            time.sleep(1)
    st.markdown("### or")
    selected_file = st_file_selector(st, path='sample_files', key = 'selected', label = 'Choose a sample image/video')

    if uploaded_file: 
        random_id = uuid.uuid1()
        base_folder = "temps"
        filename = "{}.{}".format(random_id, uploaded_file.type.split("/")[-1])
        file_type = uploaded_file.type.split("/")[0]
        filepath = f"{base_folder}/{filename}"
        faces_folder = f"{base_folder}/images/{random_id}"
        st.write(filepath)
        if uploaded_file.type == 'video/mp4':
            with open(f"temps/{filename}", mode='wb') as f:
                f.write(uploaded_file.read())
            video_path = filepath
            st.video(uploaded_file)
        else:
            img = Image.open(uploaded_file).convert('RGB')
            ext = uploaded_file.type.split("/")[-1]
            st.image(img)
    elif selected_file:
        base_folder = "sample_files"
        file_type = selected_file.split(".")[-1]
        filename = selected_file.split("/")[-1]
        filepath = f"{base_folder}/{selected_file}"

        if file_type == 'mp4':
            video_file = open(filepath, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_path = filepath
        else:
            img = Image.open(filepath).convert('RGB')
            st.image(img)
    else:
        return


    
    annotated_img = None
    with st.spinner(f'Analyzing {file_type}...'):
        if file_type == 'video' or file_type == 'mp4':
            result = video_pipeline(video_path)
        else:
            result, annotated_img = image_pipeline({'images': [img2base64(np.array(img))]}, draw_bbox=True)

    if annotated_img is not None:
        st.image(annotated_img)

    if 'incorrectly' in result['message']:
        st.error(result['message'], icon="🚨")
    else:
        st.success(result['message'], icon="✅")
    
    st.divider()
    st.write('## Response JSON')
    st.write(result)


def setup():

    if not os.path.isdir("temps"):
        os.makedirs("temps")

if __name__ == "__main__":
    image_pipeline = ImagePipeline()
    video_pipeline = VideoPipeline()

    # with st.sidebar:

    st.title("Improper Mask Wearing Detection")
    setup()
    main()