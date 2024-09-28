import streamlit as st
import numpy as np
from PIL import Image
from InferencePipeline import *
import tensorflow as tf

img_buffer = st.file_uploader('Upload an image for inference')

if img_buffer is not None:
    img = Image.open(img_buffer)
    img = prepare_image(img)
    st.write(img.shape)
    model = tf.keras.models.load_model('./model_79.h5')
    pred = infer(model, img)
    st.write(pred)
