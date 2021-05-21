from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('lr_deployment_20210521')
st.title('ML model tests')