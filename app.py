import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras import models
import tensorflow as tf
import joblib

# {'cls_names': ['NORMAL', 'PNEUMONIA'], 'size': (200, 200)}
img_config = joblib.load('imgConfs/nm_config.pkl')

print(img_config)


@st.cache(allow_output_mutation=True)
def load_model_CORONA():
    model = tf.keras.models.load_model('CoronaCT_V1.h5')

    return model


model_CORONA = load_model_CORONA()

st.title(f"Prediction of Pneumonia")

# st.set_option('deprecation.showfileUploaderEncoding', False)
IMG_FILE = st.file_uploader(
    "Please Upload Image here....", type=['jpg', 'jpeg', 'png'])


if IMG_FILE:

    img = Image.open(IMG_FILE)

    img = np.array(img.convert("RGB").resize(img_config['size']))
    img = img / 255.0
    img = np.expand_dims(img, 0)

    pred = model_CORONA.predict(img)

    result = np.argmax(pred)
    score = round(np.max(pred) * 100, 2)

    result2 = img_config['cls_names'][result]

    mystr = f" {result2}  {score}% "

    if result2 == "PNEUMONIA":
        st.error(mystr)
        st.image(IMG_FILE, use_column_width=True, caption=mystr)
    else:
        st.success(mystr)
        st.image(IMG_FILE, use_column_width=True, caption=mystr)
