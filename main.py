import streamlit as st
import json
import requests
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import tensorflow
import pandas as pd
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os

#change title and favicon
st.set_page_config(page_title="Snap&Shop", page_icon="💄", layout="wide")
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #F7A862;">
  <a class="navbar-brand" href="#" target="_blank">Snap&Shop</a>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">About</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Contact</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)
# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
               .css-1d391kg {
                    padding-top: 0rem;
                    padding-right: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])


def load_lottieurl(url: str):
      r=requests.get(url)
      if r.status_code != 200:
          return None
      return r.json()
#--load assets---
lottie_shop=load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_o02kdakv.json")
lottie_dork=load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_o02kdakv.json")
#--Header section--
with st.container():
      left_column,right_column=st.columns(2)
      with left_column:
       st_lottie(lottie_shop,key="shop")
      with right_column:
        st_lottie(lottie_dork,key="dork")
     

def save_file(uploaded_file):
    try:
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0


def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)

    return result_normlized


def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distence, indices = neighbors.kneighbors([features])

    return indices

uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    if save_file(uploaded_file):
        # display image
        show_images = Image.open(uploaded_file)
        size = (400, 400)
        resized_im = show_images.resize(size)
        st.image(resized_im)
        # extract features of uploaded image
        features = extract_img_features(os.path.join("uploader", uploaded_file.name), model)
        #st.text(features)
        img_indicess = recommendd(features, features_list)
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.header("I")
            st.image(img_files_list[img_indicess[0][0]])

        with col2:
            st.header("II")
            st.image(img_files_list[img_indicess[0][1]])

        with col3:
            st.header("III")
            st.image(img_files_list[img_indicess[0][2]])

        with col4:
            st.header("IV")
            st.image(img_files_list[img_indicess[0][3]])

        with col5:
            st.header("V")
            st.image(img_files_list[img_indicess[0][4]])
    else:
        st.header("Some error occur")
