import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor  # ใช้สำหรับ Neural Network Model


# ฟังก์ชันคำนวณเมตริกต่าง ๆ
def calculate_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    return mae, mse, r2, mape, accuracy

# โหลดโมเดลที่บันทึกไว้
model_1 = joblib.load('../decision_tree_model.pkl')
model_2 = joblib.load('../knn_model.pkl')
model_3 = joblib.load('../svr_model.pkl')
scaler = joblib.load('../scaler.pkl')

def display_ml_model():
    st.title("การทำนายราคาหุ้น")
    
    
   
    st.markdown('<p class="big-font">การเตรียมข้อมูล (Data Preprocessing)</p>', unsafe_allow_html=True)
    st.markdown('<p class="highlight">แหล่งที่มาของข้อมูล (Data Preprocessing)</p>', unsafe_allow_html=True)
    st.markdown('<p class="text_indent">การวิเคราะห์และการสร้างโมเดลที่เกี่ยวข้องกับข้อมูลตลาดหุ้นต้องอาศัยข้อมูลที่ถูกต้องและครบถ้วน แต่ในบางกรณี การเตรียมข้อมูล (Data Preparation) มีความสำคัญเทียบเท่ากับการพัฒนาโมเดล เนื่องจากข้อมูลที่ไม่สมบูรณ์สามารถสร้างความท้าทายและโอกาสในการปรับปรุงคุณภาพของแบบจำลองได้ บทความนี้จึงขอนำเสนอแนวคิดและที่มาของการสร้าง Dataset โดยใช้เทคโนโลยี AI อย่าง GPT เพื่อจำลองข้อมูลตลาดหุ้นของประเทศไทยที่มีความไม่สมบูรณ์ (Data Preprocessing)</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="highlight">การสร้าง Dataset โดยใช้ GPT</p>', unsafe_allow_html=True)
    st.markdown('<p class="text_indent">ในการสร้างข้อมูลจำลองจำนวน 10,000 ข้อมูลเกี่ยวกับดัชนีตลาดหุ้นของประเทศไทย เราได้ใช้เทคโนโลยี GPT ในการสร้างข้อมูลโดยกำหนดเงื่อนไขให้ข้อมูลมีลักษณะที่ไม่สมบูรณ์ (Incomplete Data) เพื่อให้สามารถนำไปใช้ในการฝึกและปรับปรุงขั้นตอนการเตรียมข้อมูลได้ ข้อมูลที่สร้างขึ้นประกอบด้วยคุณลักษณะสำคัญดังต่อไปนี้<br><br>'
                '1. วันที่ (Date): วันที่ที่เกี่ยวข้องกับข้อมูลดัชนีตลาดหุ้น ข้อมูลนี้บางส่วนอาจมีการขาดหาย (Missing Data) เพื่อจำลองสถานการณ์ที่ข้อมูลบางช่วงเวลาไม่สามารถบันทึกได้<br><br>'
                '2. ดัชนีตลาดหุ้น (Stock Market Index): ข้อมูลดัชนี SET Index ซึ่งเป็นดัชนีตลาดหุ้นหลักของประเทศไทย ข้อมูลนี้อาจมีความผิดพลาดบางส่วนหรือไม่ครบถ้วน<br><br>'
                '3. ราคาปิด (Close Price): ราคาปิดของดัชนีในแต่ละวัน ข้อมูลนี้มีความสำคัญในการวิเคราะห์แนวโน้มและการคำนวณผลตอบแทน<br><br>'
                '4. ราคาสูงสุด (High Price): ราคาสูงสุดในช่วงวัน ข้อมูลนี้อาจมีการขาดหายบางส่วน<br><br>'
                '5. ราคาต่ำสุด (Low Price): ราคาต่ำสุดในช่วงวัน ข้อมูลบางจุดอาจมีค่าที่ผิดปกติ (Outliers)<br><br>'
                '6. ปริมาณการซื้อขาย (Volume): จำนวนหุ้นที่ซื้อขายในวันนั้น ข้อมูลนี้อาจมีค่าที่ขาดหายหรือผิดปกติ เช่น ค่าที่สูงเกินจริง<br><br>'
                '7. ค่าเฉลี่ยเคลื่อนที่ (Moving Averages): การคำนวณค่าเฉลี่ยเคลื่อนที่ เช่น ค่าเฉลี่ย 50 วัน และ 200 วัน ซึ่งมีความสำคัญในการวิเคราะห์แนวโน้มของตลาด ข้อมูลบางชุดอาจมีค่าไม่สมเหตุสมผลเนื่องจากความไม่สมบูรณ์ของข้อมูลดิบ<br><br>'
                '8. การเปลี่ยนแปลง (Change): การเปลี่ยนแปลงของดัชนีจากวันก่อนหน้า ข้อมูลบางส่วนอาจมีค่าที่ผิดปกติเนื่องจากข้อมูลวันที่ขาดหาย<br><br>'

                '9. เปอร์เซ็นต์การเปลี่ยนแปลง (Percentage Change): การเปลี่ยนแปลงในรูปของเปอร์เซ็นต์จากราคาปิดวันก่อนหน้า ข้อมูลบางจุดอาจมีความไม่ถูกต้องหากข้อมูลราคาปิดไม่ครบถ้วน</p>', unsafe_allow_html=True)
             
    st.markdown('<p class="highlight">ประโยชน์ของการสร้างข้อมูลที่ไม่สมบูรณ์</p>', unsafe_allow_html=True)
    st.markdown('<p class="normal-text">• การฝึกฝนการเตรียมข้อมูล (Data Preparation): ผู้พัฒนาสามารถฝึกฝนการเติมค่าข้อมูลที่ขาดหาย การตรวจจับค่าผิดปกติ และการทำความสะอาดข้อมูล  <br><br>'
        '• การทดสอบโมเดลที่มีความยืดหยุ่น: โมเดลที่สามารถจัดการข้อมูลที่ไม่สมบูรณ์ได้มีแนวโน้มที่จะมีความยืดหยุ่นและประสิทธิภาพที่ดีขึ้นในสถานการณ์จริง  <br><br>'
        '• การปรับปรุงคุณภาพข้อมูล: การวิเคราะห์ข้อมูลที่ไม่สมบูรณ์ช่วยให้สามารถพัฒนาเทคนิคในการจัดการข้อมูลที่ขาดหายได้ดียิ่งขึ้น </p>', unsafe_allow_html=True)
    # กำหนด path ของไฟล์ CSV โดยตรง
    file_path = '../Thailand_Stock_Market_Data.csv'

    # ตรวจสอบว่าไฟล์ CSV มีอยู่หรือไม่
    try:
        df = pd.read_csv(file_path)
        
        # ลบช่องว่างด้านหน้า-หลังของชื่อคอลัมน์ทั้งหมด
        df.columns = df.columns.str.strip()

        # รีเซ็ตดัชนีและตั้งค่าให้เริ่มที่ 1
        df.reset_index(drop=True, inplace=True)
        df.index += 1  # เลขแถวเริ่มที่ 1

        # แสดงข้อมูลในไฟล์
        st.markdown('<p class="highlight">ข้อมูลในไฟล์ที่อัปโหลด:</p>', unsafe_allow_html=True)
        st.dataframe(df)  # แสดงข้อมูลทั้งหมดและสามารถเลื่อนดูได้

        st.markdown('<p class="text_indent">ในการเตรียมข้อมูลสำหรับการวิเคราะห์ข้อมูลทางการเงิน ควรทำการจัดการกับค่าที่ขาดหายไปในตารางที่มีคอลัมน์สำคัญดังนี้: วันที่ (Date), ราคาปิด (Close Price), ราคาสูงสุด (High Price), ราคาต่ำสุด (Low Price), ปริมาณการซื้อขาย (Volume), ค่าเฉลี่ยเคลื่อนที่ 50 วัน (Moving Average - 50 days), ค่าเฉลี่ยเคลื่อนที่ 200 วัน (Moving Average - 200 days), การเปลี่ยนแปลง (Change), และการเปลี่ยนแปลงเปอร์เซ็นต์ (Percentage Change) </p>', unsafe_allow_html=True)
        st.markdown('<p class="normal-text">การจัดการค่าที่ขาดหายไปในตารางข้อมูลทางการเงินเป็นขั้นตอนที่สำคัญในการเตรียมข้อมูลให้พร้อมสำหรับการวิเคราะห์ วิธีการที่ใช้ในการจัดการกับข้อมูลที่หายไปจะมีผลต่อความแม่นยำของผลลัพธ์ในภายหลัง ดังนั้นจึงมีวิธีการที่สามารถนำไปใช้ได้หลากหลาย ขึ้นอยู่กับลักษณะของข้อมูลและวัตถุประสงค์ในการวิเคราะห์ โดยวิธีที่นิยมใช้ได้แก่: </p>', unsafe_allow_html=True)
        
        st.markdown('<p class="text_indent">การเติมค่าที่หายไปด้วยค่าเฉลี่ย (Mean Imputation)การเติมค่าที่หายไปด้วยค่าเฉลี่ยของแต่ละคอลัมน์ (เช่น ค่าเฉลี่ยของราคาปิด) เป็นวิธีที่ง่ายและนิยมใช้กันมากที่สุด เนื่องจากช่วยให้ข้อมูลสมบูรณ์โดยไม่ต้องตัดข้อมูลออกจากการพิจารณา </p>', unsafe_allow_html=True)
        st.markdown('<p class="text_indent">การใช้ค่าที่ใกล้เคียง (Forward Fill หรือ Backward Fill)บางครั้งค่าที่ขาดหายไปอาจเกิดขึ้นเนื่องจากการเก็บข้อมูลไม่สมบูรณ์ในบางช่วงเวลา ในกรณีนี้การใช้ค่าที่มีอยู่แล้วในแถวก่อนหน้าหรือหลัง (เช่น ใช้ค่าก่อนหน้าหรือหลังเติมค่าที่หายไป) อาจเป็นวิธีที่ช่วยรักษาความต่อเนื่องของข้อมูลได้ </p>', unsafe_allow_html=True)
        st.markdown('<p class="text_indent">การตัดข้อมูลที่ขาดหายไป (Dropping Missing Data)ในบางกรณี การตัดแถวที่มีค่าหายไปออกจาก DataFrame อาจเป็นวิธีที่เหมาะสม โดยเฉพาะถ้าข้อมูลที่หายไปมีจำนวนไม่มากและไม่ส่งผลกระทบต่อการวิเคราะห์</p>', unsafe_allow_html=True)
        
        
        st.markdown('<p class="big-font">วิธีการเตรียมข้อมูลเบื้องต้น</p>', unsafe_allow_html=True)
        st.markdown('<p class="normal-text">หลังจากที่เราทำความเข้าใจเกี่ยวกับวิธีการจัดการค่าที่หายไปแล้ว ต่อไปนี้คือโค้ดเบื้องต้นสำหรับการตรวจสอบและจัดการกับค่าที่ขาดหายไปในตารางข้อมูลทางการเงิน </p>', unsafe_allow_html=True)
        
        #ตรวจสอบคอลัมน์ที่จำเป็น
        st.markdown('<p class="highlight">1.ตรวจสอบคอลัมน์ที่จำเป็น</p>', unsafe_allow_html=True)
        code = '''
        # ตรวจสอบว่าคอลัมน์ที่ต้องการมีอยู่จริงหรือไม่
required_columns = ['Date', 'Close Price', 'High Price', 'Low Price', 'Volume', 'Change', 'Moving Average (50 days)', 'Moving Average (200 days)', 'Percentage Change']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")
        '''
        st.code(code, language='python')
        st.markdown('<p class="text_indent">ขั้นตอนแรกในการเตรียมข้อมูลคือการตรวจสอบว่า DataFrame ของเรามีคอลัมน์ที่จำเป็นทั้งหมดหรือไม่ เพื่อให้มั่นใจว่าข้อมูลที่ใช้มีครบถ้วนก่อนการประมวลผล โดยโค้ดนี้จะตรวจสอบว่าทุกคอลัมน์ใน required_columns มีอยู่ใน DataFrame data หรือไม่ หากคอลัมน์ใดขาดหายไป จะมีการยกข้อผิดพลาด ValueError ขึ้นมา พร้อมกับชื่อคอลัมน์ที่ขาด</p>', unsafe_allow_html=True)

        # แยก Features (X) และ Target (y)
        st.markdown('<p class="highlight"><br>2.แยก Features (X) และ Target (y) </p>', unsafe_allow_html=True)
        code = '''
# แยก X และ y
X = data.drop(['Date', 'Close Price'], axis=1)
y = data['Close Price'].ffill()  # เติมค่าที่หายไปใน target
        ''' 
        st.code(code, language='python')
        #
        st.markdown('<p class="text_indent">การแยกข้อมูลเป็น Features (X) และ Target (y) คือขั้นตอนที่สำคัญในกระบวนการเรียนรู้ของโมเดล <br>'
        '• X คือ Features ที่ใช้ในการทำนาย ซึ่งได้จากการลบคอลัมน์ Date และ Close Price ออกไป <br>'
        '• y คือ Target หรือคอลัมน์ Close Price ที่เราจะทำนาย โดยใช้ ffill() เพื่อเติมค่าที่หายไป (missing values) ด้วยค่าก่อนหน้า </p>', unsafe_allow_html=True)
        
        
        # เติมค่า missing values ใน Features (X)
        st.markdown('<p class="highlight"><br>3.เติมค่า missing values ใน Features (X)', unsafe_allow_html=True)
        
       
        code = '''
# เติมค่า missing values ใน X
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        '''
        st.code(code, language='python')
        st.markdown('<p class="text_indent">ในบางครั้งข้อมูลอาจมีค่าที่หายไป (missing values) การเติมค่าเหล่านี้เป็นขั้นตอนที่สำคัญเพื่อให้ข้อมูลสมบูรณ์<br>'
                    '• SimpleImputer เป็นเครื่องมือจาก sklearn ที่ใช้เติมค่าที่หายไป <br>'
                    '• เราเลือกใช้ strategy=mean เพื่อเติมค่าที่หายไปด้วยค่าเฉลี่ยของแต่ละคอลัมน์ <br>'
                    '• ผลลัพธ์จะถูกแปลงกลับเป็น DataFrame พร้อมเก็บชื่อคอลัมน์เดิม</p>', unsafe_allow_html=True)
        # scaling ข้อมูล
        st.markdown('<p class="highlight"><br>4. Scaling ข้อมูล</p>', unsafe_allow_html=True)
        code = '''
# Scaling ข้อมูล
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
        '''
        st.code(code, language='python')
        st.markdown('<p class="text_indent">การ Scaling ข้อมูลช่วยให้ฟีเจอร์ทั้งหมดมีค่าเฉลี่ย (mean) เท่ากับ 0 และส่วนเบี่ยงเบนมาตรฐาน (standard deviation) เท่ากับ 1 ซึ่งช่วยให้โมเดลเรียนรู้ได้ดีขึ้น <br>'
                    '• StandardScaler จะปรับสเกลของข้อมูลทั้งหมดให้มี mean = 0 และ standard deviation = 1<br>'
                    '• การใช้ fit_transform() เพื่อปรับสเกลข้อมูล</p>', unsafe_allow_html=True)
       # แบ่งข้อมูลเป็น Train และ Test
        st.markdown('<p class="highlight"><br>5.แบ่งข้อมูลเป็น Train และ Test</p>', unsafe_allow_html=True)
        code = '''
# Split ข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42
        # ตรวจสอบว่าคอลัมน์ที่ต้องการมีอยู่จริงหรือไม่
        '''
        st.code(code, language='python')
        st.markdown('<p class="text_indent">เมื่อเตรียมข้อมูลเสร็จแล้ว เราจะทำการแบ่งข้อมูลออกเป็นชุดฝึก (train) และชุดทดสอบ (test) เพื่อประเมินผลการทำงานของโมเดล <br>'
                    '• test_size=0.2 หมายถึงแบ่งข้อมูล 20% สำหรับชุดทดสอบ และ 80% สำหรับชุดฝึก<br>'
                    '• random_state=42 เป็นการตั้งค่า seed เพื่อให้ผลลัพธ์ออกมาเหมือนเดิมทุกครั้งที่รันโค้ด</p>', unsafe_allow_html=True)
       
        st.markdown('<p class="big-font ">โมเดลที่นำมาใช้</p>', unsafe_allow_html=True)
        st.markdown('<p class="highlight">1.K-Nearest Neighbors (KNN)</p>', unsafe_allow_html=True)
        st.markdown('<p class="normal-text">• วิธีการทำงาน: KNN ใช้แนวคิดที่ว่า "สิ่งที่คล้ายกันจะอยู่ใกล้กัน" ในการทำนายราคาหุ้นโดยการหาความสัมพันธ์ระหว่างข้อมูลการซื้อขายในอดีตและผลลัพธ์ที่ต้องการ โดยดูจากข้อมูลที่ใกล้เคียงที่สุด (neighbors) กับข้อมูลใหม่ที่ต้องการทำนาย <br>'
                    '• การใช้ในโปรเจคนี้: KNN จะทำการทำนายราคาปิดของหุ้นโดยอ้างอิงข้อมูลจากราคาปิดในช่วงเวลาใกล้เคียงและใช้ค่าเฉลี่ยจากเพื่อนบ้านที่มีลักษณะคล้ายกัน เช่น ราคาสูงสุด, ราคาต่ำสุด, หรือค่าเฉลี่ยเคลื่อนที่<br></p>', unsafe_allow_html=True)
       
        st.markdown('<p class="highlight">2.Decision Tree (D3)</p>', unsafe_allow_html=True)
        st.markdown('<p class="normal-text">• วิธีการทำงาน: Decision Tree สร้างโมเดลที่เป็นโครงสร้างต้นไม้เพื่อการตัดสินใจ โดยการแบ่งข้อมูลออกเป็นกลุ่มๆ ตามคุณสมบัติต่างๆ เช่น ราคาสูงสุด, ราคาต่ำสุด, หรือการเปลี่ยนแปลงของราคาหุ้น <br>'
                    '• การใช้ในโปรเจคนี้: โมเดลนี้จะใช้ในการแบ่งข้อมูลทางการเงินและการทำนายราคาหุ้นตามลักษณะต่างๆ เช่น การใช้ค่าเฉลี่ยเคลื่อนที่ 50 และ 200 วัน เพื่อให้สามารถตัดสินใจได้ว่าแนวโน้มราคาหุ้นจะเป็นอย่างไรในอนาคต<br></p>', unsafe_allow_html=True)
       
       
        st.markdown('<p class="highlight">3.Support Vector Regression (SVR)</p>', unsafe_allow_html=True)
        st.markdown('<p class="normal-text">• วิธีการทำงาน: SVR เป็นการนำ Support Vector Machine (SVM) มาประยุกต์ใช้ในการทำนายค่าตัวเลข (regression) โดยพยายามหาฟังก์ชันที่ดีที่สุดในการทำนายราคาหุ้นจากข้อมูลทางการเงิน <br>'
                    '• การใช้ในโปรเจคนี้: SVR จะใช้ข้อมูลเชิงลึกของราคาหุ้นและค่าต่างๆ เช่น ราคาสูงสุดและราคาต่ำสุด เพื่อคาดการณ์ราคาปิดในอนาคต โดยคำนึงถึงความสัมพันธ์ระหว่างตัวแปรต่างๆ และไม่พยายามจับคู่ข้อมูลเกินไป (ไม่เกิด overfitting)<br></p>', unsafe_allow_html=True)
       
        st.markdown('<p class="highlight">4.Ensemble Method (Stacking)</p>', unsafe_allow_html=True)
        st.markdown('<p class="normal-text">• วิธีการทำงาน: Ensemble Method รวมหลายโมเดลเข้าด้วยกันเพื่อเพิ่มความแม่นยำในการทำนาย โดยการใช้โมเดลต่างๆ เช่น KNN, Decision Tree, และ SVR มารวมกันเพื่อให้ผลลัพธ์ที่มีความแม่นยำสูงสุด<br>'
                    '• การใช้ในโปรเจคนี้: ในการทำนายราคาหุ้น SET สามารถนำโมเดลหลายตัวมารวมกันและใช้เทคนิคการ stacking เพื่อให้ผลลัพธ์การทำนายมีความเสถียรและแม่นยำมากขึ้น<br></p>', unsafe_allow_html=True)
       
        datat = {
    'Model': ['K-Nearest Neighbors (KNN)', 'Decision Tree (D3)', 'Support Vector Regression (SVR)', 'Ensemble Method (Stacking)'],
    'ข้อดี': [
        'ใช้งานง่าย\nเหมาะสำหรับข้อมูลจำนวนน้อย\nทำงานได้ดีเมื่อมีรูปแบบที่ชัดเจน',
        'เข้าใจง่าย\nสามารถใช้กับทั้ง Regression และ Classification\nตีความได้ง่าย',
        'จัดการกับข้อมูลที่ไม่เป็นเส้นตรงได้ดี\nทำนายตัวเลขได้แม่นยำ',
        'รวมหลายโมเดลเพื่อเพิ่มความแม่นยำ\nลดความผิดพลาดจากโมเดลเดี่ยว'
    ],
    'ข้อเสีย': [
        'ใช้พลังในการคำนวณสูงเมื่อข้อมูลมาก\nไม่เหมาะกับข้อมูลที่มีมิติสูง',
        'อาจเกิด overfitting ถ้าโมเดลซับซ้อนเกินไป\nไม่เหมาะกับข้อมูลที่มีลักษณะซับซ้อนมาก',
        'การปรับพารามิเตอร์ซับซ้อน\nต้องการการคำนวณที่สูง',
        'ใช้เวลาในการฝึกและคำนวณมากขึ้น\nอาจยากต่อการตีความผลลัพธ์'
    ]
}

# สร้าง DataFrame จากข้อมูล
        dft = pd.DataFrame(datat)
        dft.index += 1

# แสดงตารางใน Streamlit
        st.markdown('<p class="big-font ">ข้อดีและข้อเสียของแต่ละโมเดล</p>', unsafe_allow_html=True)
        st.table(dft)
       
        
       
        
        
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

    except FileNotFoundError:
        st.write("ไม่พบไฟล์ CSV ในโฟลเดอร์")

