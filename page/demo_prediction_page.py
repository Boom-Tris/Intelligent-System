import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor  # ใช้สำหรับ Neural Network Model
from pathlib import Path

# หาตำแหน่งโฟลเดอร์ที่ไฟล์ .pkl อยู่



def display_demo_prediction():
    base_path = Path(__file__).parent.parent / "ML"
    model_1 = joblib.load(base_path / "decision_tree_model.pkl")
    model_2 = joblib.load(base_path / "knn_model.pkl")
    model_3 = joblib.load(base_path / "svr_model.pkl")
    scaler = joblib.load(base_path / "scaler.pkl")
    st.title("ทำนายราคาหุ้นด้วยข้อมูลที่ผู้ใช้กรอก")

    # กรอกข้อมูลการทำนาย
    st.write("กรุณากรอกข้อมูลที่เกี่ยวข้องกับราคาหุ้น:")

    high_price = st.number_input("High Price", min_value=0.0)
    low_price = st.number_input("Low Price", min_value=0.0)
    volume = st.number_input("Volume", min_value=0)
    ma_50 = st.number_input("Moving Average (50 days)", min_value=0.0)
    ma_200 = st.number_input("Moving Average (200 days)", min_value=0.0)
    change = st.number_input("Change", min_value=0.0)
    perc_change = st.number_input("Percentage Change", min_value=0.0)

    input_data = pd.DataFrame([[high_price, low_price, volume, ma_50, ma_200, change, perc_change]], 
                              columns=['High Price', 'Low Price', 'Volume', 'Moving Average (50 days)', 'Moving Average (200 days)', 'Change', 'Percentage Change'])

    # เติมค่าที่หายไปใน input_data ด้วยค่าเฉลี่ย
    imputer = SimpleImputer(strategy='mean')
    input_data = pd.DataFrame(imputer.fit_transform(input_data), columns=input_data.columns)

    # ทำการ scaling ข้อมูล
    input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    # ใช้ st.selectbox แต่ซ่อน
    model_choice = st.selectbox("เลือกโมเดลที่ใช้ในการทำนาย", ["Decision Tree", "KNN", "SVR", "Neural Network", "Ensemble"], key="model_select", label_visibility="collapsed")

    # ทำนายด้วยโมเดลที่เลือก
    if model_choice == "Decision Tree":
        pred = model_1.predict(input_data_scaled)
    elif model_choice == "KNN":
        pred = model_2.predict(input_data_scaled)
    elif model_choice == "SVR":
        pred = model_3.predict(input_data_scaled)
    elif model_choice == "Neural Network":
        
        # คำนวณการทำนายโดยใช้ Ensemble (ค่าผลรวมของทุกโมเดล)
        pred_dt = model_1.predict(input_data_scaled)
        pred_knn = model_2.predict(input_data_scaled)
        pred_svr = model_3.predict(input_data_scaled)
        

        # คำนวณผลลัพธ์ Ensemble โดยเฉลี่ยค่าการทำนายจากทุกโมเดล
        pred = (pred_dt + pred_knn + pred_svr) / 3

    st.write(f"ราคาหุ้นที่ทำนาย: {pred[0]:.2f}")
