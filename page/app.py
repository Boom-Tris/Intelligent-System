import streamlit as st
from home_page import display_home
from ml_model_page import display_ml_model
from nn_model_page import display_nn_model
from demo_prediction_page import display_demo_prediction
import os
from pathlib import Path
def local_css(file_name):
    file_path = Path(__file__).parent / file_name  # หาตำแหน่งไฟล์แบบ Dynamic
    if file_path.exists():  # ตรวจสอบว่าไฟล์มีอยู่จริง
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.error(f"CSS file not found: {file_path}")

# โหลด CSS
local_css("styles.css")  # ไม่ต้องใช้ "./" หรือ "../"
# เลือกหน้า (Home, ML Model, Neural Network Model, Demo Prediction)
page = st.sidebar.radio("เลือกหน้า", ("Home", "ML Model", "Neural Network Model", "Demo Prediction"))

# --- หน้าแรก (Home) ---
if page == "Home":
    display_home()

# --- หน้า ML Model ---
elif page == "ML Model":
    display_ml_model()

# --- หน้า Neural Network Model ---
elif page == "Neural Network Model":
    display_nn_model()

# --- หน้า Demo Prediction ---
elif page == "Demo Prediction":
    display_demo_prediction()
