import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def display_nn_model():
    st.title("Neural Network Model สำหรับการทำนายราคาหุ้น")
    uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])

    if uploaded_file is not None:
        # โหลดข้อมูลจากไฟล์ที่อัปโหลด
        df = pd.read_csv(uploaded_file)
        st.write("ข้อมูลในไฟล์ที่อัปโหลด:")
        st.dataframe(df)
        # โค้ดที่ใช้ในการทำนายด้วยโมเดล Neural Network
