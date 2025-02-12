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


# ฟังก์ชันคำนวณเมตริกต่าง ๆ
def calculate_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    return mae, mse, r2, mape, accuracy

# โหลดโมเดลที่บันทึกไว้

def display_ml_model():
    
    base_path = Path(__file__).parent.parent / "ML"
    model_1 = joblib.load(base_path / "decision_tree_model.pkl")
    model_2 = joblib.load(base_path / "knn_model.pkl")
    model_3 = joblib.load(base_path / "svr_model.pkl")
    scaler = joblib.load(base_path / "scaler.pkl")
    file_path = Path(__file__).parent.parent / "data"
    file_csv =  file_path / "Thailand_Stock_Market_Data.csv"
    st.title("การทำนายราคาหุ้น")
   
    st.markdown('<div class="big-font">การเตรียมข้อมูล (Data Preprocessing)</div>', unsafe_allow_html=True)
    st.markdown('<div class="highlight">แหล่งที่มาของข้อมูล (Data Preprocessing)</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การวิเคราะห์และการสร้างโมเดลที่เกี่ยวข้องกับข้อมูลตลาดหุ้นต้องอาศัยข้อมูลจำนวนมาก ซึ่งต้องการความถูกต้องและความครบถ้วนของข้อมูล แต่ในบางครั้ง การเตรียมข้อมูล (Data Preparation) มีความสำคัญไม่น้อยไปกว่าการพัฒนาโมเดล เพราะข้อมูลที่ไม่สมบูรณ์อาจสร้างความท้าทายในการทำงานและส่งผลต่อคุณภาพของโมเดลได้ บทความนี้จึงจะนำเสนอแนวคิดและที่มาของการสร้าง Dataset ที่ใช้ในโปรเจคนี้ โดยใช้เทคโนโลยี AI อย่าง ChatGPT ในการจำลองข้อมูลตลาดหุ้นของประเทศไทยที่มีความไม่สมบูรณ์ เพื่อเสริมสร้างความเข้าใจและการจัดการกับข้อมูลที่มีข้อบกพร่องอย่างมีประสิทธิภาพ.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="highlight"><br>การสร้าง Dataset โดยใช้ ChatGPT</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ในการสร้างข้อมูลจำลองในโปรเจคนี้ เราได้ใช้ ChatGPT ในการสร้างข้อมูลจำลองจำนวน 10,000 รายการเกี่ยวกับดัชนีตลาดหุ้นของประเทศไทย โดยกำหนดเงื่อนไขให้ข้อมูลมีลักษณะที่ไม่สมบูรณ์ (Incomplete Data) เพื่อให้สามารถนำไปใช้ในการฝึกและปรับปรุงขั้นตอนการเตรียมข้อมูลได้ ข้อมูลที่สร้างขึ้นประกอบด้วยคอลัมน์สำคัญดังนี้<br><br>'
                '1. Date: วันที่ที่เกี่ยวข้องกับข้อมูลดัชนีตลาดหุ้น ข้อมูลนี้บางส่วนอาจมีการขาดหาย (Missing Data) เพื่อจำลองสถานการณ์ที่ข้อมูลบางช่วงเวลาไม่สามารถบันทึกได้<br><br>'
                '2. Close Price: ราคาปิดของดัชนีในแต่ละวัน ข้อมูลนี้มีความสำคัญในการวิเคราะห์แนวโน้มและการคำนวณผลตอบแทน<br><br>'
                '3. High Price: ราคาสูงสุดในช่วงวัน ข้อมูลนี้อาจมีการขาดหายบางส่วน<br><br>'
                '4. Low Price: ราคาต่ำสุดในช่วงวัน ข้อมูลบางจุดอาจมีค่าที่ผิดปกติ (Outliers)<br><br>'
                '5. Volume: จำนวนหุ้นที่ซื้อขายในวันนั้น ข้อมูลนี้อาจมีค่าที่ขาดหายหรือผิดปกติ เช่น ค่าที่สูงเกินจริง<br><br>'
                '6. Moving Averages: การคำนวณค่าเฉลี่ยเคลื่อนที่ เช่น ค่าเฉลี่ย 50 วัน และ 200 วัน ซึ่งมีความสำคัญในการวิเคราะห์แนวโน้มของตลาด ข้อมูลบางชุดอาจมีค่าไม่สมเหตุสมผลเนื่องจากความไม่สมบูรณ์ของข้อมูลดิบ<br><br>'
                '7. Change: การเปลี่ยนแปลงของดัชนีจากวันก่อนหน้า ข้อมูลบางส่วนอาจมีค่าที่ผิดปกติเนื่องจากข้อมูลวันที่ขาดหาย<br><br>'

                '8. Percentage Change: การเปลี่ยนแปลงในรูปของเปอร์เซ็นต์จากราคาปิดวันก่อนหน้า ข้อมูลบางจุดอาจมีความไม่ถูกต้องหากข้อมูลราคาปิดไม่ครบถ้วน</div>', unsafe_allow_html=True)
             
    st.markdown('<div class="highlight"><br>ประโยชน์ของการสร้างข้อมูลที่ไม่สมบูรณ์</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• การฝึกฝนการเตรียมข้อมูล (Data Preparation): ผู้พัฒนาสามารถฝึกฝนการเติมค่าข้อมูลที่ขาดหาย การตรวจจับค่าผิดปกติ และการทำความสะอาดข้อมูล  <br><br>'
        '• การทดสอบโมเดลที่มีความยืดหยุ่น: โมเดลที่สามารถจัดการข้อมูลที่ไม่สมบูรณ์ได้มีแนวโน้มที่จะมีความยืดหยุ่นและประสิทธิภาพที่ดีขึ้นในสถานการณ์จริง  <br><br>'
        '• การปรับปรุงคุณภาพข้อมูล: การวิเคราะห์ข้อมูลที่ไม่สมบูรณ์ช่วยให้สามารถพัฒนาเทคนิคในการจัดการข้อมูลที่ขาดหายได้ดียิ่งขึ้น </div>', unsafe_allow_html=True)
    # กำหนด path ของไฟล์ CSV โดยตรง
   
    

    # ตรวจสอบว่าไฟล์ CSV มีอยู่หรือไม่
    try:
        df = pd.read_csv(file_csv)
        
        # ลบช่องว่างด้านหน้า-หลังของชื่อคอลัมน์ทั้งหมด
        df.columns = df.columns.str.strip()

        # รีเซ็ตดัชนีและตั้งค่าให้เริ่มที่ 1
        df.reset_index(drop=True, inplace=True)
        df.index += 1  # เลขแถวเริ่มที่ 1

        # แสดงข้อมูลในไฟล์
        st.markdown('<div class="highlight"><br>ข้อมูลในไฟล์ที่อัปโหลด</div>', unsafe_allow_html=True)
        st.dataframe(df)  # แสดงข้อมูลทั้งหมดและสามารถเลื่อนดูได้

        st.markdown('<div class="text_indent">ในการเตรียมข้อมูลสำหรับการวิเคราะห์ข้อมูลทางการเงิน ควรทำการจัดการกับค่าที่ขาดหายไปในตารางที่มีคอลัมน์สำคัญดังนี้ Date, Close Price, High Price, Low Price, Volume, Moving Average - 50 days, Moving Average - 200 days, Change, Percentage Change</div>', unsafe_allow_html=True)
        st.markdown('<br><div class="text_indent">การจัดการค่าที่ขาดหายไปในตารางข้อมูลทางการเงินเป็นขั้นตอนที่สำคัญในการเตรียมข้อมูลให้พร้อมสำหรับการวิเคราะห์ วิธีการที่ใช้ในการจัดการกับข้อมูลที่หายไปจะมีผลต่อความแม่นยำของผลลัพธ์ในภายหลัง ดังนั้นจึงมีวิธีการที่สามารถนำไปใช้ได้หลากหลาย ขึ้นอยู่กับลักษณะของข้อมูลและวัตถุประสงค์ในการวิเคราะห์ โดยวิธีที่นิยมใช้ได้แก่</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="text_indent">การเติมค่าที่หายไปด้วยค่าเฉลี่ย (Mean Imputation)การเติมค่าที่หายไปด้วยค่าเฉลี่ยของแต่ละคอลัมน์ (เช่น ค่าเฉลี่ยของราคาปิด) เป็นวิธีที่ง่ายและนิยมใช้กันมากที่สุด เนื่องจากช่วยให้ข้อมูลสมบูรณ์โดยไม่ต้องตัดข้อมูลออกจากการพิจารณา </div> <br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">การใช้ค่าที่ใกล้เคียง (Forward Fill หรือ Backward Fill)บางครั้งค่าที่ขาดหายไปอาจเกิดขึ้นเนื่องจากการเก็บข้อมูลไม่สมบูรณ์ในบางช่วงเวลา ในกรณีนี้การใช้ค่าที่มีอยู่แล้วในแถวก่อนหน้าหรือหลัง (เช่น ใช้ค่าก่อนหน้าหรือหลังเติมค่าที่หายไป) อาจเป็นวิธีที่ช่วยรักษาความต่อเนื่องของข้อมูลได้ </div> <br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">การตัดข้อมูลที่ขาดหายไป (Dropping Missing Data)ในบางกรณี การตัดแถวที่มีค่าหายไปออกจาก DataFrame อาจเป็นวิธีที่เหมาะสม โดยเฉพาะถ้าข้อมูลที่หายไปมีจำนวนไม่มากและไม่ส่งผลกระทบต่อการวิเคราะห์</div>', unsafe_allow_html=True)
        
        
        st.markdown('<div class="big-font"><br>วิธีการเตรียมข้อมูลเบื้องต้น</div>', unsafe_allow_html=True)
        st.markdown('<div class="normal-text">หลังจากที่เราทำความเข้าใจเกี่ยวกับวิธีการจัดการค่าที่หายไปแล้ว ต่อไปนี้คือโค้ดเบื้องต้นสำหรับการตรวจสอบและจัดการกับค่าที่ขาดหายไปในตารางข้อมูลทางการเงิน </div>', unsafe_allow_html=True)
        
        #ตรวจสอบคอลัมน์ที่จำเป็น
        st.markdown('<div class="highlight"><br>1.ตรวจสอบคอลัมน์ที่จำเป็น</div>', unsafe_allow_html=True)
        code = '''
        # ตรวจสอบว่าคอลัมน์ที่ต้องการมีอยู่จริงหรือไม่
required_columns = ['Date', 'Close Price', 'High Price', 'Low Price', 'Volume', 'Change', 'Moving Average (50 days)', 'Moving Average (200 days)', 'Percentage Change']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")
        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">ขั้นตอนแรกในการเตรียมข้อมูลคือการตรวจสอบว่า DataFrame ของเรามีคอลัมน์ที่จำเป็นทั้งหมดหรือไม่ เพื่อให้มั่นใจว่าข้อมูลที่ใช้มีครบถ้วนก่อนการประมวลผล โดยโค้ดนี้จะตรวจสอบว่าทุกคอลัมน์ใน required_columns มีอยู่ใน DataFrame data หรือไม่ หากคอลัมน์ใดขาดหายไป จะมีการยกข้อผิดพลาด ValueError ขึ้นมา พร้อมกับชื่อคอลัมน์ที่ขาด</div>', unsafe_allow_html=True)

        # แยก Features (X) และ Target (y)
        st.markdown('<div class="highlight"><br>2.แยก Features (X) และ Target (y) </div>', unsafe_allow_html=True)
        code = '''
# แยก X และ y
X = data.drop(['Date', 'Close Price'], axis=1)
y = data['Close Price'].ffill()  # เติมค่าที่หายไปใน target
        ''' 
        st.code(code, language='python')
        #
        st.markdown('<div class="text_indent">การแยกข้อมูลเป็น Features (X) และ Target (y) คือขั้นตอนที่สำคัญในกระบวนการเรียนรู้ของโมเดล <br>'
        '• X คือ Features ที่ใช้ในการทำนาย ซึ่งได้จากการลบคอลัมน์ Date และ Close Price ออกไป <br><br>'
        '• y คือ Target หรือคอลัมน์ Close Price ที่เราจะทำนาย โดยใช้ ffill() เพื่อเติมค่าที่หายไป (missing values) ด้วยค่าก่อนหน้า </div>', unsafe_allow_html=True)
        
        
        # เติมค่า missing values ใน Features (X)
        st.markdown('<div class="highlight"><br>3.เติมค่า missing values ใน Features (X)', unsafe_allow_html=True)
        
       
        code = '''
# เติมค่า missing values ใน X
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">ในบางครั้งข้อมูลอาจมีค่าที่หายไป (missing values) การเติมค่าเหล่านี้เป็นขั้นตอนที่สำคัญเพื่อให้ข้อมูลสมบูรณ์<br>'
                    '• SimpleImputer เป็นเครื่องมือจาก sklearn ที่ใช้เติมค่าที่หายไป <br><br>'
                    '• เราเลือกใช้ strategy=mean เพื่อเติมค่าที่หายไปด้วยค่าเฉลี่ยของแต่ละคอลัมน์ <br><br>'
                    '• ผลลัพธ์จะถูกแปลงกลับเป็น DataFrame พร้อมเก็บชื่อคอลัมน์เดิม</div>', unsafe_allow_html=True)
        # scaling ข้อมูล
        st.markdown('<div class="highlight"><br>4. Scaling ข้อมูล</div>', unsafe_allow_html=True)
        code = '''
# Scaling ข้อมูล
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">การ Scaling ข้อมูลช่วยให้ฟีเจอร์ทั้งหมดมีค่าเฉลี่ย (mean) เท่ากับ 0 และส่วนเบี่ยงเบนมาตรฐาน (standard deviation) เท่ากับ 1 ซึ่งช่วยให้โมเดลเรียนรู้ได้ดีขึ้น <br>'
                    '• StandardScaler จะปรับสเกลของข้อมูลทั้งหมดให้มี mean = 0 และ standard deviation = 1<br><br>'
                    '• การใช้ fit_transform() เพื่อปรับสเกลข้อมูล</div>', unsafe_allow_html=True)
       # แบ่งข้อมูลเป็น Train และ Test
        st.markdown('<div class="highlight"><br>5.แบ่งข้อมูลเป็น Train และ Test</div>', unsafe_allow_html=True)
        code = '''
# Split ข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42
        # ตรวจสอบว่าคอลัมน์ที่ต้องการมีอยู่จริงหรือไม่
        '''
        st.code(code, language='python')
        
        st.markdown('<div class="text_indent">เมื่อเตรียมข้อมูลเสร็จแล้ว เราจะทำการแบ่งข้อมูลออกเป็นชุดฝึก (train) และชุดทดสอบ (test) เพื่อประเมินผลการทำงานของโมเดล <br>'
                    '• test_size=0.2 หมายถึงแบ่งข้อมูล 20% สำหรับชุดทดสอบ และ 80% สำหรับชุดฝึก<br><br>'
                    '• random_state=42 เป็นการตั้งค่า seed เพื่อให้ผลลัพธ์ออกมาเหมือนเดิมทุกครั้งที่รันโค้ด</div>', unsafe_allow_html=True)
       # สร้างโมเดลที่ใช้
        st.markdown('<div class="big-font "><br>โมเดลที่นำมาใช้</div>', unsafe_allow_html=True)
       # knn
        st.markdown('<div class="highlight">1.K-Nearest Neighbors (KNN)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">เมื่อพูดถึงการทำนายราคาหุ้น แน่นอนว่าเรามักจะต้องใช้เทคนิคต่าง ๆ ในการวิเคราะห์ข้อมูลเพื่อให้ได้ผลลัพธ์ที่แม่นยำ และหนึ่งในเทคนิคที่ได้รับความนิยมคือ K-Nearest Neighbors (KNN) ซึ่งเป็นวิธีการที่ทำงานบนแนวคิดง่าย ๆ ว่า "สิ่งที่คล้ายกันจะอยู่ใกล้กัน" นั่นหมายความว่า เมื่อเราพยายามทำนายราคาหุ้นในวันถัดไปหรือในช่วงเวลาหนึ่ง เราจะพิจารณาจากข้อมูลที่มีลักษณะคล้ายคลึงกับข้อมูลปัจจุบันที่เรากำลังต้องการทำนาย เพื่อให้ได้ผลลัพธ์ที่ดีที่สุด </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">KNN ใช้หลักการที่เรียกว่า Similarity-based approach โดยพิจารณาความสัมพันธ์ระหว่างข้อมูลปัจจุบันและข้อมูลในอดีต ซึ่งจะคำนวณระยะห่างระหว่างข้อมูลทั้งสอง ผ่านฟังก์ชันการวัดระยะห่าง เช่น Euclidean distance หรือ Manhattan distance ที่ช่วยให้เรารู้ว่าข้อมูลไหนใกล้เคียงที่สุดกับข้อมูลที่เรากำลังจะทำนาย สำหรับตลาดหุ้น KNN จะมองหาความสัมพันธ์ระหว่างราคาหุ้นในอดีต (เช่น ราคาปิด, ราคาสูงสุด, ราคาต่ำสุด) เพื่อช่วยทำนายราคาหุ้นในอนาคต</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ทฤษฎีเบื้องหลัง  K-Nearest Neighbors (KNN) เป็นหนึ่งในอัลกอริธึมการเรียนรู้ที่ง่ายที่สุด แต่มีประสิทธิภาพสูง โดยที่ KNN ไม่มีการสร้างโมเดลที่ชัดเจนและไม่ต้องการการฝึกฝนล่วงหน้า (non-parametric) ซึ่งหมายความว่า KNN จะใช้ข้อมูลทั้งหมดในฐานข้อมูลเพื่อทำนายผลลัพธ์โดยตรงแทนการสร้างฟังก์ชันเชิงเส้นหรือเชิงซับซ้อนใด ๆ</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">แนวคิดของ KNN คือการหาความคล้ายคลึงระหว่างข้อมูล ซึ่งอาศัย distance metrics เช่น Euclidean distance หรือ Manhattan distance เพื่อตัดสินใจว่า ข้อมูลใดที่ใกล้เคียงกับข้อมูลปัจจุบันมากที่สุด จากนั้นจึงคำนวณผลลัพธ์จากการคำนวณเฉลี่ยหรือค่าผลลัพธ์ที่ได้จาก K neighbors ที่ใกล้เคียงที่สุด การเลือก K value เป็นปัจจัยที่สำคัญมากในการทำงานของ KNN ถ้า K เล็กเกินไป โมเดลอาจมีความเสี่ยงที่จะเกิด overfitting หรือถ้า K ใหญ่เกินไป อาจทำให้การทำนายไม่แม่นยำและอ่อนไหวต่อข้อมูลที่ผิดปกติ </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">การใช้ KNN ในโปรเจคนี้ ในการทำนายราคาปิดของหุ้นในโปรเจคนี้ เราใช้ KNN เพื่อคำนวณราคาปิดของหุ้นในอนาคต โดยอ้างอิงจากราคาปิดในช่วงเวลาที่ใกล้เคียงกัน เมื่อเรานำข้อมูลต่าง ๆ เช่น ราคาสูงสุด, ราคาต่ำสุด, หรือค่าเฉลี่ยเคลื่อนที่ (Moving Average) มาใช้ร่วมกัน KNN จะช่วยให้เราได้การทำนายที่มีความแม่นยำยิ่งขึ้น โดยการคำนวณจากเพื่อนบ้านที่ใกล้เคียงและมีลักษณะคล้ายกันมากที่สุด </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">สิ่งสำคัญในการใช้ KNN คือการเลือกค่า K ที่เหมาะสม ซึ่งเป็นจำนวนของเพื่อนบ้านที่ใช้ในการคำนวณ ผลลัพธ์ที่ได้จากการทำนายจะมีความแม่นยำขึ้นหากเราเลือก K ที่เหมาะสมกับลักษณะของข้อมูลในตลาดหุ้น  <br></div><br>', unsafe_allow_html=True)
        
        st.markdown('<div class="highlight">2.Decision Tree (D3)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">เมื่อพูดถึงการทำนายราคาหุ้นและการตัดสินใจในการลงทุน แนวทางที่ได้รับความนิยมอีกหนึ่งเทคนิคคือ Decision Tree (D3) ซึ่งเป็นโมเดลที่ช่วยให้เราเข้าใจและจัดกลุ่มข้อมูลที่มีลักษณะเฉพาะ ด้วยการใช้โครงสร้างที่คล้ายกับต้นไม้ เพื่อให้สามารถตัดสินใจได้อย่างชัดเจนและมีประสิทธิภาพ</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">Decision Tree คือโมเดลที่สร้างเป็นโครงสร้างต้นไม้ ซึ่งมีลักษณะคล้ายกับกระบวนการตัดสินใจของมนุษย์ที่แบ่งข้อมูลออกเป็นกลุ่ม ๆ ตามคุณสมบัติหรือคุณลักษณะเฉพาะของข้อมูล เช่น ราคาสูงสุด, ราคาต่ำสุด, หรือการเปลี่ยนแปลงของราคาหุ้น การตัดสินใจในแต่ละโหนด (Node) ของต้นไม้จะพิจารณาจากการเลือกคุณสมบัติที่ดีที่สุดที่จะช่วยให้การแบ่งข้อมูลเป็นไปได้อย่างมีประสิทธิภาพที่สุด การเลือกคุณสมบัติจะพิจารณาจาก Entropy หรือ Gini Impurity ซึ่งเป็นมาตรวัดความบริสุทธิ์ของข้อมูลที่มีการแบ่งออกเป็นกลุ่มต่าง ๆ โดยที่ข้อมูลในแต่ละกลุ่มจะมีลักษณะคล้ายกันมากที่สุด</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ทฤษฎีเบื้องหลัง Decision TreeDecision Tree มีพื้นฐานมาจากการใช้หลักการของการแบ่งข้อมูลออกเป็นกลุ่ม (splitting) ตามคุณสมบัติที่สำคัญที่สุดในการทำนายผลลัพธ์ ซึ่งทำให้สามารถเข้าใจและตีความได้ง่าย โดยกระบวนการนี้จะถูกดำเนินการผ่าน Recursion (การทำงานซ้ำ) โดยเริ่มจากการแบ่งข้อมูลที่มีลักษณะไม่แน่นอนให้กลายเป็นข้อมูลที่มีความบริสุทธิ์มากขึ้นจนกว่าจะได้กลุ่มข้อมูลที่มีความเป็นเอกลักษณ์ (pure) นั่นเองการใช้ Entropy และ Gini Impurity จะช่วยให้การตัดสินใจในการแบ่งกลุ่มข้อมูลนั้นมีประสิทธิภาพสูงสุด ซึ่งจะทำให้โมเดลมีความแม่นยำในการทำนายราคาหุ้นมากขึ้น</div><br>', unsafe_allow_html=True)
        
        st.markdown('<div class="text_indent">กระบวนการนี้ช่วยให้ Decision Tree สามารถตัดสินใจได้ในรูปแบบที่เข้าใจง่าย และมีความโปร่งใส ซึ่งเหมาะสมอย่างยิ่งกับการทำนายราคาหุ้นที่มีข้อมูลและปัจจัยหลากหลาย</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ในโปรเจคนี้ Decision Tree ถูกนำมาใช้ในการแบ่งกลุ่มข้อมูลทางการเงินและทำนายราคาหุ้น โดยเฉพาะการใช้ข้อมูลที่สำคัญ เช่น ค่าเฉลี่ยเคลื่อนที่ 50 (50-Day Moving Average) และ ค่าเฉลี่ยเคลื่อนที่ 200 วัน (200-Day Moving Average) ซึ่งช่วยในการวิเคราะห์แนวโน้มของตลาดหุ้นในระยะยาวและระยะสั้น การใช้ค่าเฉลี่ยเคลื่อนที่ช่วยให้เรามองเห็นภาพรวมของราคาหุ้นที่มีความผันผวนได้อย่างชัดเจน โดยการตัดสินใจว่าจะซื้อหรือขายหุ้นจะขึ้นอยู่กับการวิเคราะห์ข้อมูลเหล่านี้ รวมถึงปัจจัยอื่น ๆ ที่อาจส่งผลต่อราคาหุ้น</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">Decision Tree ช่วยให้สามารถตัดสินใจได้ว่าแนวโน้มของราคาหุ้นในอนาคตจะเป็นไปในทิศทางใด โดยการแบ่งข้อมูลออกเป็นกลุ่ม ๆ และใช้กฎการตัดสินใจที่ได้จากการวิเคราะห์ข้อมูลในอดีต ซึ่งสามารถช่วยให้ผู้ลงทุนมีเครื่องมือในการตัดสินใจที่ชัดเจนและมีพื้นฐานจากข้อมูลที่มีความสมเหตุสมผล</div><br>', unsafe_allow_html=True)
       
       
        st.markdown('<div class="highlight">3.Support Vector Regression (SVR)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">เมื่อพูดถึงการทำนายราคาหุ้นในตลาดที่มีความผันผวนและไม่แน่นอน การเลือกเครื่องมือที่เหมาะสมในการวิเคราะห์ข้อมูลเป็นสิ่งสำคัญ หนึ่งในเทคนิคที่ได้รับความนิยมคือ Support Vector Regression (SVR) ซึ่งเป็นการนำแนวคิดจาก Support Vector Machine (SVM) มาประยุกต์ใช้ในการทำนายค่าตัวเลขหรือ regression เช่น ราคาหุ้นในอนาคต</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">หลักการทำงานของ SVR คือ การหาฟังก์ชันที่ดีที่สุดที่สามารถทำนายค่าตัวแปรเป้าหมาย (ในที่นี้คือราคาหุ้น) จากชุดข้อมูลที่มีอยู่ โดยพยายามหาฟังก์ชันที่มีความคลาดเคลื่อนน้อยที่สุดจากข้อมูลจริงในขอบเขตที่กำหนด (ใน SVR นี้จะถูกควบคุมโดยค่า epsilon ซึ่งเป็นค่าที่อนุญาตให้มีการคลาดเคลื่อนในระดับหนึ่ง) และยังสามารถทำนายค่าหมายเลขออกมาได้แม่นยำทั้งในกรณีที่ข้อมูลมีความไม่แน่นอนหรือมีข้อผิดพลาด</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ทฤษฎีเบื้องหลัง SVRSVR เป็นการขยายแนวคิดจาก Support Vector Machine (SVM) ที่มีการเลือกเส้นที่ดีที่สุดในการแบ่งข้อมูลในกรณีของ classification แต่ใน regression นั้นการหาฟังก์ชันที่ดีที่สุดจะใช้ epsilon-insensitive loss function ซึ่งหมายความว่า การทำนายจะมีความคลาดเคลื่อนน้อยที่สุดภายในขอบเขตที่กำหนดโดย epsilon แต่จะไม่พยายามหาฟังก์ชันที่เหมาะสมกับข้อมูลมากเกินไป ดังนั้นจึงลดโอกาสของการเกิด overfitting ซึ่งทำให้ SVR สามารถทำนายผลได้แม่นยำแม้ในกรณีที่ข้อมูลมีความไม่สมบูรณ์หรือมีเสียงรบกวนอีกหนึ่งข้อดีของ SVR คือการใช้ kernel trick ซึ่งช่วยให้สามารถจับความสัมพันธ์ที่ซับซ้อนในข้อมูลได้ โดยการแปลงข้อมูลจากมิติเดิมไปยังมิติที่สูงขึ้น ทำให้สามารถหาความสัมพันธ์ที่ซ่อนอยู่ได้อย่างมีประสิทธิภาพ</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">สิ่งที่ทำให้ SVR แตกต่างจากการทำนายแบบ regression ทั่วไปคือการใช้ kernel trick ซึ่งช่วยให้สามารถจับความสัมพันธ์ที่ซับซ้อนได้โดยการเปลี่ยนข้อมูลต้นฉบับไปในมิติที่สูงขึ้น เพื่อให้การหาฟังก์ชันที่ดีที่สุดทำได้ง่ายขึ้น โดยไม่ต้องกังวลเกี่ยวกับการสร้างฟังก์ชันที่ซับซ้อนเกินไป</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ในโปรเจคนี้ SVR ถูกนำมาใช้ในการทำนายราคาหุ้น โดยมุ่งเน้นที่การทำนายราคาปิดในอนาคตโดยอ้างอิงจากข้อมูลเชิงลึกต่าง ๆ เช่น ราคาสูงสุด, ราคาต่ำสุด, หรือการเปลี่ยนแปลงของราคาหุ้นในช่วงเวลาต่าง ๆ SVR จะพยายามหาความสัมพันธ์ที่ดีที่สุดระหว่างตัวแปรเหล่านี้เพื่อทำนายราคาปิดในอนาคตอย่างแม่นยำ</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">สิ่งที่สำคัญคือ SVR ไม่พยายามจับคู่ข้อมูลเกินไปหรือเกิดการ overfitting ซึ่งเป็นข้อดีของโมเดลนี้ เมื่อเทียบกับโมเดล regression อื่น ๆ ที่อาจจะปรับตัวให้เหมาะสมกับข้อมูลมากเกินไป จนทำให้โมเดลมีประสิทธิภาพต่ำเมื่อเจอข้อมูลใหม่ที่ไม่เคยเห็นมาก่อน</div><br>', unsafe_allow_html=True)
        
       
        st.markdown('<div class="highlight">4.Ensemble Method (Stacking)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">เมื่อพูดถึงการทำนายราคาหุ้นในตลาดที่มีความผันผวนสูง ความแม่นยำของโมเดลในการทำนายสามารถทำให้เกิดการตัดสินใจที่ดีขึ้นในการลงทุน ซึ่งการใช้ Ensemble Methods ถือเป็นหนึ่งในวิธีที่ได้รับความนิยมในการเพิ่มความแม่นยำให้กับโมเดลทำนาย โดยเฉพาะในเทคนิค Stacking ที่สามารถนำหลายโมเดลมารวมกันเพื่อให้ผลลัพธ์ที่ดีที่สุด</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">Ensemble Method เป็นวิธีการที่ใช้หลายโมเดลในการทำนายผลลัพธ์และนำผลลัพธ์จากแต่ละโมเดลมารวมกันเพื่อเพิ่มความแม่นยำของการทำนาย โดยการใช้โมเดลหลายตัว เช่น K-Nearest Neighbors (KNN), Decision Tree, และ Support Vector Regression (SVR) มารวมกันในลักษณะที่เสริมกันและลดข้อผิดพลาดของโมเดลแต่ละตัว โดยการทำนายของแต่ละโมเดลจะถูกนำมารวมกันเพื่อให้ผลลัพธ์สุดท้ายที่มีความแม่นยำสูงขึ้น</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ใน Stacking, โมเดลแรก (ที่เรียกว่า base models) จะทำการทำนายค่าผลลัพธ์ตามข้อมูลที่มี โดยจะใช้ meta-model (หรือเรียกว่า stacking model) ซึ่งเป็นโมเดลที่ใช้ผลลัพธ์จาก base models เป็นข้อมูลนำเข้าในการทำการทำนายสุดท้าย เพื่อหาค่าที่ดีที่สุดที่สามารถอธิบายได้จากทุกโมเดลที่ใช้การรวมโมเดลเหล่านี้ช่วยให้ระบบทำนายมีความเสถียรและทนทานต่อความผิดพลาดหรือข้อมูลที่ไม่สมบูรณ์ในกรณีที่โมเดลตัวใดตัวหนึ่งเกิดข้อผิดพลาดหรือไม่สามารถจับความสัมพันธ์ได้ดี</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ทฤษฎีเบื้องหลัง Ensemble Method (Stacking)Ensemble Learning อาศัยหลักการที่ว่า การรวมหลาย ๆ โมเดลที่มีความแตกต่างกันสามารถช่วยให้ผลลัพธ์ที่ได้มีความแม่นยำมากขึ้น โดยการใช้หลายโมเดลเพื่อทำนายผลลัพธ์จะทำให้สามารถแก้ไขข้อผิดพลาดจากโมเดลใดโมเดลหนึ่งได้ เนื่องจากโมเดลแต่ละตัวอาจจะมีจุดแข็งและจุดอ่อนที่ต่างกัน Stacking เป็นหนึ่งในเทคนิค ensemble ที่ได้รับความนิยม โดยที่ base models หลายตัวจะทำการทำนายผลลัพธ์ และ meta-model จะนำผลลัพธ์จากโมเดลเหล่านั้นมาเรียนรู้เพื่อทำการทำนายที่แม่นยำยิ่งขึ้น การเรียนรู้แบบนี้ช่วยให้ระบบลดความเสี่ยงจากการใช้โมเดลเดียวที่อาจจะมีข้อผิดพลาด หรืออาจจะไม่สามารถจับลักษณะบางอย่างในข้อมูลได้</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ในโปรเจคนี้ Ensemble Method โดยเฉพาะ Stacking ถูกนำมาใช้เพื่อทำนายราคาหุ้นในตลาด SET โดยการรวมหลายโมเดลที่มีคุณสมบัติในการทำนายที่แตกต่างกัน เช่น KNN, Decision Tree, และ SVR เพื่อให้ผลลัพธ์ที่เสถียรและแม่นยำมากขึ้น</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ในแต่ละโมเดลนั้นจะให้ผลลัพธ์ที่แตกต่างกัน แต่โดยการใช้ stacking, เราสามารถนำผลลัพธ์จากโมเดลแต่ละตัวมารวมกัน ซึ่งจะช่วยให้การทำนายราคาหุ้นในอนาคตมีความแม่นยำและเสถียรยิ่งขึ้น การใช้ meta-model จะช่วยตัดสินใจว่าค่าผลลัพธ์ที่ได้จากโมเดลแต่ละตัวนั้นควรจะถูกคำนวณอย่างไร เพื่อให้ได้ผลลัพธ์ที่ดีที่สุดในการทำนายราคาหุ้นในตลาด</div><br>', unsafe_allow_html=True)
       
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
        st.markdown('<div class="highlight">ข้อดีและข้อเสียของแต่ละโมเดล</div>', unsafe_allow_html=True)
        st.table(dft)
       
        
       
        
        
        # แยก Features (X) และ Target (y)
        required_columns = ['High Price', 'Low Price', 'Volume', 'Moving Average (50 days)', 'Moving Average (200 days)', 'Change', 'Percentage Change']

        # ตรวจสอบว่าแต่ละคอลัมน์ใน required_columns มีอยู่ใน df หรือไม่
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.write(f"คอลัมน์ที่ขาดหายไป {missing_columns}")
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
            st.markdown('<div class="big-font">Metrics วัดประสิทธิภาพของโมเดล Machine Learning </div>', unsafe_allow_html=True)
            st.markdown('<div class="text_indent">การวัดประสิทธิภาพของโมเดล Machine Learning เป็นขั้นตอนสำคัญที่ช่วยให้เราประเมินความแม่นยำและความน่าเชื่อถือของการทำนายได้ ในโปรเจกต์นี้ เราได้นำ Metrics หลายตัว มาใช้ในการประเมินโมเดลแต่ละตัว ดังนี้</div>', unsafe_allow_html=True)
            st.markdown('<br><div class="highlight">RMSE (Root Mean Squared Error)</div>', unsafe_allow_html=True)
            st.markdown('<div class="text_indent">RMSE คือ ค่ารากที่สองของค่าเฉลี่ยของความคลาดเคลื่อนที่ยกกำลังสอง ใช้เพื่อวัดว่าค่าที่โมเดลทำนายแตกต่างจากค่าจริงมากน้อยเพียงใด </div>', unsafe_allow_html=True)
            
            # RMSE
           
            st.markdown('<div class="text_indent">ถ้าเราบวกลบค่าคลาดเคลื่อนตรง ๆ อาจมีบางค่าหักล้างกันได้ ดังนั้นเราจึงยกกำลังสองก่อนเฉลี่ย แต่พอยกกำลังสอง หน่วยของค่าจะเปลี่ยนไปจากเดิม (เช่น จาก "บาท" เป็น "บาท²") ดังนั้นเราจึง ถอดรูทกลับ เพื่อให้หน่วยตรงกับค่าจริง</div><br>', unsafe_allow_html=True)
            st.latex(r'''RMSE = \sqrt{\frac{1}{n} \times \sum  (prediction - actual)^2}''')
            
            
            # MAE 
            st.markdown('<br><div class="highlight">MAE (Mean Absolute Error)</div>', unsafe_allow_html=True)
            st.markdown('<div class="text_indent">MAE คือ ค่าความคลาดเคลื่อนเฉลี่ยของการทำนาย โดยไม่สนใจเครื่องหมายบวกหรือลบ </div>', unsafe_allow_html=True)
          
            st.markdown('<div class="text_indent">บางครั้งเราไม่อยากให้ค่าคลาดเคลื่อนที่สูง ๆ มีน้ำหนักมากเกินไป (เหมือน RMSE) ดังนั้นเราจึงใช้ ค่าเฉลี่ยของค่าความผิดพลาดแบบสัมบูรณ์ แทน</div><br>', unsafe_allow_html=True)
            st.latex(r'''MAE = \frac{1}{n} \sum \left| prediction_i - actual_i \right|''')
            
            # R² (R-Squared)
            st.markdown('<br><div class="highlight">R² (R-Squared)</div>', unsafe_allow_html=True)
            st.markdown('<div class="text_indent">R² หรือ Coefficient of Determination เป็นค่าที่บอกว่าโมเดลอธิบายความแปรปรวนของข้อมูลได้ดีแค่ไหน </div>', unsafe_allow_html=True)
          
            st.markdown('<div class="text_indent">บางครั้งเราต้องการดูว่าโมเดลของเรา ดีกว่าการเดาแบบสุ่มไหม หรืออธิบายความสัมพันธ์ของตัวแปรได้ดีแค่ไหน</div><br>', unsafe_allow_html=True)
            st.latex(r'''R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}''')
            
            # MAPE (Mean Absolute Percentage Error)
            st.markdown('<br><div class="highlight">MAPE (Mean Absolute Percentage Error)</div>', unsafe_allow_html=True)
            st.markdown('<div class="text_indent">MAPE คือ ค่าคลาดเคลื่อนโดยคิดเป็นเปอร์เซ็นต์ของค่าจริง ทำให้สามารถเปรียบเทียบโมเดลที่มีหน่วยต่างกันได้</div>', unsafe_allow_html=True)
          
            st.markdown('<div class="text_indent">บางครั้งเราต้องการดูว่าโมเดลพยากรณ์พลาดไปกี่เปอร์เซ็นต์ของค่าจริง แทนที่จะดูเป็นหน่วยเลขเฉย ๆ</div><br>', unsafe_allow_html=True)
            st.latex(r'''MAPE = \frac{1}{n} \sum \left| \frac{Actual - Forecast}{Actual} \right| \times 100''')
            
            # 5. Accuracy (%)
            st.markdown('<br><div class="highlight"> 5. Accuracy (%)</div>', unsafe_allow_html=True)
            st.markdown('<div class="text_indent">Accuracy เป็นตัวชี้วัดที่คำนวณจากค่า MAPE เพื่อแปลงเป็นค่าความแม่นยำที่เข้าใจง่าย<br> Accuracy=100−MAPE</div><br>', unsafe_allow_html=True)
            st.latex(r'''Accuracy = 100 - MAPE''')
           
           
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
            st.markdown('<br><div class="big-font">ผลการประเมินประสิทธิภาพของโมเดล</div>', unsafe_allow_html=True)
            st.dataframe(results_df.style.format({'MAE': '{:.4f}', 'MSE': '{:.4f}', 'R2': '{:.4f}', 'MAPE (%)': '{:.4f}', 'Prediction Accuracy (%)': '{:.4f}'}))

            st.markdown('<div class="highlight">📊 ความหมายของแต่ละตัวชี้วัด</div>', unsafe_allow_html=True)
            st.markdown('<div class="highlight">RMSE (Root Mean Squared Error)', unsafe_allow_html=True)
            st.markdown('<div class="normal-text">•ค่าเฉลี่ยรากที่สองของข้อผิดพลาด ยิ่งค่าต่ำยิ่งดี แสดงว่าค่าทำนายใกล้เคียงค่าจริง<br><br>'
                        '•ค่าในตาราง: SVR ให้ RMSE ต่ำสุด (36.39) แปลว่าทำนายได้แม่นยำที่สุดในแง่ความคลาดเคลื่อน</div>', unsafe_allow_html=True)
            
            st.markdown('<br><div class="highlight">RMSE MAE (Mean Absolute Error)', unsafe_allow_html=True)
            st.markdown('<div class="normal-text">•ค่าความผิดพลาดเฉลี่ยในเชิงบวก ยิ่งน้อยยิ่งดี<br><br>'
                        '•ค่าในตาราง: Ensemble ให้ MAE ต่ำกว่า SVR เล็กน้อยที่ (8720.90) </div>', unsafe_allow_html=True)
            
            st.markdown('<br><div class="highlight">R² (Coefficient of Determination)', unsafe_allow_html=True)
            st.markdown('<div class="normal-text">•ค่าที่แสดงประสิทธิภาพของโมเดลในการอธิบายข้อมูล อยู่ระหว่าง 0 ถึง 1 ยิ่งใกล้ 1 ยิ่งดี<br><br>'
                        '•ค่าในตาราง: Ensemble ให้ R² สูงสุดที่ 0.8951 ซึ่งบ่งบอกว่าอธิบายข้อมูลได้ดีที่สุด</div>', unsafe_allow_html=True)
           
          
            st.markdown('<br><div class="highlight">MAPE (Mean Absolute Percentage Error)', unsafe_allow_html=True)
            st.markdown('<div class="normal-text">•ค่าความคลาดเคลื่อนเฉลี่ยในเชิงเปอร์เซ็นต์ ยิ่งต่ำยิ่งดี<br><br>'
                        '•ค่าในตาราง: SVR มี MAPE ต่ำสุด (2.57%)</div>', unsafe_allow_html=True)
            
            st.markdown('<br><div class="highlight">Accuracy (%)', unsafe_allow_html=True)
            st.markdown('<div class="normal-text">•ความแม่นยำของโมเดล ยิ่งใกล้ 100% ยิ่งดี<br><br>'
                        '•ค่าในตาราง: SVR ให้ความแม่นยำสูงสุดที่ 97.43%', unsafe_allow_html=True)
           
          
           
            
            
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

