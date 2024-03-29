# app.py
import streamlit as st
from predict_page1 import show_predict_page
from explore_page1 import show_explore_page
import pandas as pd

page= st.selectbox("Explore or Predict",("Predict","Explore"))
# Load dataset
#dataset = pd.read_csv("diabetes2.csv")
#df = dataset.copy()

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
