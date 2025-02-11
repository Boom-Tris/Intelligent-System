import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor  # ใช้สำหรับ Neural Network Model
# อ่านไฟล์ CSS

# ฟังก์ชันคำนวณเมตริกต่าง ๆ
def calculate_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    return mae, mse, r2, mape, accuracy

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# นำเข้าไฟล์ CSS
local_css("styles.css")
# โหลดโมเดลที่เทรนไว้
model_1 = joblib.load('./page/decision_tree_model.pkl')
model_2 = joblib.load('./page/knn_model.pkl')
model_3 = joblib.load('./page/svr_model.pkl')
scaler = joblib.load('./page/scaler.pkl')
#nn_model = joblib.load('neural_network_model.pkl')  # โมเดล Neural Network ที่เทรนแล้ว

# เลือกหน้า (Home, ML Model, Neural Network Model, Demo Prediction)
page = st.sidebar.radio("เลือกหน้า", ("Home", "ML Model", "Neural Network Model", "Demo Prediction"))

# --- หน้าแรก (Home) ---
if page == "Home":
    
    st.title("สวัสดีครับนี้คือ Final Project ในรายวิชา Intelligent System ")
    st.markdown('<p class="big-font">โดย: ธีระพัฒน์ จ่อนตะมะ รหัสนักศึกษา 6604062620131</p>', unsafe_allow_html=True)
    st.markdown('<p class="normal-text">  ในโปรเจคนี้ เราจะสำรวจและนำเทคนิคการเรียนรู้ของเครื่อง (Machine Learning) หลายตัวที่สำคัญมาใช้ เช่น</p>', unsafe_allow_html=True)

# --- หน้า ML Model ---
if page == "ML Model":
    st.title("เลือกโมเดลที่ต้องการดู")
    
    # ให้ผู้ใช้เลือกดูระหว่าง ML Model กับ NN Model
    option = st.radio("", ["Machine Learning Model", "nn_model"])

    if option == "Machine Learning Model":
        st.header("Machine Learning Model")
        st.write("รายละเอียดเกี่ยวกับ Machine Learning Model ที่ใช้...")
        # ใส่โค้ดสำหรับ Machine Learning Model ที่นี่
        uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])

    if uploaded_file is not None:
        # โหลดข้อมูลจากไฟล์ที่อัปโหลด
        df = pd.read_csv(uploaded_file)
        
        # ลบช่องว่างด้านหน้า-หลังของชื่อคอลัมน์ทั้งหมด
        df.columns = df.columns.str.strip()

        # รีเซ็ตดัชนีและตั้งค่าให้เริ่มที่ 1
        df.reset_index(drop=True, inplace=True)
        df.index += 1  # เลขแถวเริ่มที่ 1

        # แสดงข้อมูลในไฟล์
        st.write("ข้อมูลในไฟล์ที่อัปโหลด:")
        st.dataframe(df)  # แสดงข้อมูลทั้งหมดและสามารถเลื่อนดูได้

        # แยก Features (X) และ Target (y)
        required_columns = ['High Price', 'Low Price', 'Volume', 'Moving Average (50 days)', 'Moving Average (200 days)', 'Change', 'Percentage Change']

        # ตรวจสอบว่าแต่ละคอลัมน์ใน required_columns มีอยู่ใน df หรือไม่
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.write(f"คอลัมน์ที่ขาดหายไป: {missing_columns}")
        else:
            # แยก Features (X) และ Target (y)
            X_test_new = df[required_columns].copy()
            y_test = df['Close Price'].ffill()  # เติมค่าหายไปใน target ด้วย forward fill

            # เติมค่าที่หายไปใน X_test_new ด้วยค่าเฉลี่ย
            imputer = SimpleImputer(strategy='mean')
            X_test_new = pd.DataFrame(imputer.fit_transform(X_test_new), columns=required_columns)

            # ใช้ scaler ที่เคย fit กับ training data แล้ว
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_new), columns=required_columns)

            # ทำนายด้วยโมเดล
            pred_1 = model_1.predict(X_test_scaled)
            pred_2 = model_2.predict(X_test_scaled)
            pred_3 = model_3.predict(X_test_scaled)

            # คำนวณผลรวมของการทำนาย
            y_pred_ensemble = (pred_1 + pred_2 + pred_3) / 3

            # คำนวณค่า MAE, MSE, R2, MAPE
            mae_1, mse_1, r2_1, mape_1, accuracy_1 = calculate_metrics(y_test, pred_1)
            mae_2, mse_2, r2_2, mape_2, accuracy_2 = calculate_metrics(y_test, pred_2)
            mae_3, mse_3, r2_3, mape_3, accuracy_3 = calculate_metrics(y_test, pred_3)
            mae_ensemble, mse_ensemble, r2_ensemble, mape_ensemble, accuracy_ensemble = calculate_metrics(y_test, y_pred_ensemble)

            # สร้าง DataFrame เพื่อแสดงผลลัพธ์ในตาราง
            result_data = {
                'Model': ['Decision Tree', 'KNN', 'SVR', 'Ensemble'],
                'MAE': [mae_1, mae_2, mae_3, mae_ensemble],
                'MSE': [mse_1, mse_2, mse_3, mse_ensemble],
                'R2': [r2_1, r2_2, r2_3, r2_ensemble],
                'MAPE (%)': [mape_1, mape_2, mape_3, mape_ensemble],
                'Prediction Accuracy (%)': [accuracy_1, accuracy_2, accuracy_3, accuracy_ensemble]
            }

            results_df = pd.DataFrame(result_data)

            # รีเซ็ตดัชนีและตั้งค่าให้เริ่มที่ 1
            results_df.reset_index(drop=True, inplace=True)
            results_df.index += 1  # เลขรันเริ่มที่ 1

            # แสดงผลลัพธ์ในรูปแบบตาราง
            st.write("Performance Metrics for Each Model:")
            st.dataframe(results_df.style.format({'MAE': '{:.4f}', 'MSE': '{:.4f}', 'R2': '{:.4f}', 'MAPE (%)': '{:.4f}', 'Prediction Accuracy (%)': '{:.4f}'}))

            # --- สร้าง Density Plot --- #
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.kdeplot(y_test, label='Actual', ax=ax, color='black', linestyle='--')
            sns.kdeplot(pred_1, label='Decision Tree', ax=ax, color='blue')
            sns.kdeplot(pred_2, label='KNN', ax=ax, color='green')
            sns.kdeplot(pred_3, label='SVR', ax=ax, color='red')
            sns.kdeplot(y_pred_ensemble, label='Ensemble', ax=ax, color='orange', linestyle='-.')

            ax.set_title('Density Plot of Model Predictions vs Actual Values')
            ax.set_xlabel('Close Price')
            ax.set_ylabel('Density')
            ax.legend()

            # แสดง Density Plot
            st.pyplot(fig)
    elif option == "nn_model":
        st.header("Neural Network Model สำหรับการทำนายราคาหุ้น")
        st.write("รายละเอียดเกี่ยวกับ Neural Network Model ที่ใช้...")
        # ใส่โค้ดสำหรับ Neural Network Model ที่นี่
    # ให้ผู้ใช้สามารถอัปโหลดไฟล์
    

# --- หน้า Neural Network Model ---
if page == "Neural Network Model":
    st.title("Neural Network Model สำหรับการทำนายราคาหุ้น")
    
    # ให้ผู้ใช้สามารถอัปโหลดไฟล์
    uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])

    if uploaded_file is not None:
        # โหลดข้อมูลจากไฟล์ที่อัปโหลด
        df = pd.read_csv(uploaded_file)
        
        # ลบช่องว่างด้านหน้า-หลังของชื่อคอลัมน์ทั้งหมด
        df.columns = df.columns.str.strip()

        # รีเซ็ตดัชนีและตั้งค่าให้เริ่มที่ 1
        df.reset_index(drop=True, inplace=True)
        df.index += 1  # เลขแถวเริ่มที่ 1

        # แสดงข้อมูลในไฟล์
        st.write("ข้อมูลในไฟล์ที่อัปโหลด:")
        st.dataframe(df)  # แสดงข้อมูลทั้งหมดและสามารถเลื่อนดูได้

        # แยก Features (X) และ Target (y)
        required_columns = ['High Price', 'Low Price', 'Volume', 'Moving Average (50 days)', 'Moving Average (200 days)', 'Change', 'Percentage Change']

        # ตรวจสอบว่าแต่ละคอลัมน์ใน required_columns มีอยู่ใน df หรือไม่
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.write(f"คอลัมน์ที่ขาดหายไป: {missing_columns}")
        else:
            # แยก Features (X) และ Target (y)
            X_test_new = df[required_columns].copy()
            y_test = df['Close Price'].ffill()  # เติมค่าหายไปใน target ด้วย forward fill

            # เติมค่าที่หายไปใน X_test_new ด้วยค่าเฉลี่ย
            imputer = SimpleImputer(strategy='mean')
            X_test_new = pd.DataFrame(imputer.fit_transform(X_test_new), columns=required_columns)

            # ใช้ scaler ที่เคย fit กับ training data แล้ว
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_new), columns=required_columns)

            # ทำนายด้วยโมเดล Neural Network
            pred_nn = nn_model.predict(X_test_scaled)

            # คำนวณค่า MAE, MSE, R2, MAPE
            mae_nn, mse_nn, r2_nn, mape_nn, accuracy_nn = calculate_metrics(y_test, pred_nn)

            # สร้าง DataFrame เพื่อแสดงผลลัพธ์ในตาราง
            result_data = {
                'Model': ['Neural Network'],
                'MAE': [mae_nn],
                'MSE': [mse_nn],
                'R2': [r2_nn],
                'MAPE (%)': [mape_nn],
                'Prediction Accuracy (%)': [accuracy_nn]
            }

            results_df = pd.DataFrame(result_data)

            # รีเซ็ตดัชนีและตั้งค่าให้เริ่มที่ 1
            results_df.reset_index(drop=True, inplace=True)
            results_df.index += 1  # เลขรันเริ่มที่ 1

            # แสดงผลลัพธ์ในรูปแบบตาราง
            st.write("Performance Metrics for Neural Network Model:")
            st.dataframe(results_df.style.format({'MAE': '{:.4f}', 'MSE': '{:.4f}', 'R2': '{:.4f}', 'MAPE (%)': '{:.4f}', 'Prediction Accuracy (%)': '{:.4f}'})) 

# --- หน้า Demo Prediction --- 
if page == "Demo Prediction":
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
        pred = nn_model.predict(input_data_scaled)
    elif model_choice == "Ensemble":
        # คำนวณการทำนายโดยใช้ Ensemble (ค่าผลรวมของทุกโมเดล)
        pred_dt = model_1.predict(input_data_scaled)
        pred_knn = model_2.predict(input_data_scaled)
        pred_svr = model_3.predict(input_data_scaled)
        

        # คำนวณผลลัพธ์ Ensemble โดยเฉลี่ยค่าการทำนายจากทุกโมเดล
        pred = (pred_dt + pred_knn + pred_svr) / 3

    st.write(f"ราคาหุ้นที่ทำนาย: {pred[0]:.2f}")
