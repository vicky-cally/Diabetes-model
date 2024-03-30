# app.py
import streamlit as st
from predict_page1 import show_predict_page
#from explore_page1 import show_explore_page
import pandas as pd

#page= st.sidebar.selectbox("Explore or Predict",("Predict","Explore"))

#if page == "Predict":
show_predict_page()
#else:
    #show_explore_page()
