import streamlit as st

# ฟังก์ชันโหลด CSS


# ฟังก์ชันสำหรับหน้า Home
def display_home():
    st.title("สวัสดีครับนี้คือ Final Project ในรายวิชา Intelligent System ")
    st.markdown('<div class="highlight">โดย: ธีระพัฒน์ จ่อนตะมะ รหัสนักศึกษา 6604062620131</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ในโปรเจคนี้ เราจะนำเทคนิคการเรียนรู้ในเรื่อง Machine Learning หลายตัวที่สำคัญมาใช้ เช่น Decision Tree, K-Nearest Neighbors (KNN), Support Vector Regression (SVR), Ensemble Method (Stacking) ผลลัพธ์จะถูกประเมินด้วยตัวชี้วัดทางสถิติเพื่อหาว่าโมเดลใดให้ผลลัพธ์ที่ดีที่สุด และ Neural Networks โดยมีเป้าหมายในการนำโมเดลเหล่านี้มาประยุกต์ใช้ในการแก้ปัญหาจริงและเปรียบเทียบประสิทธิภาพของแต่ละโมเดล</div>', unsafe_allow_html=True)
    #โมเดลการเรียนรู้
    st.markdown('<div class="big-font"> โมเดลการเรียนรู้ </div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">1. Decision Tree – โมเดลที่ไม่เป็นเชิงเส้น (Non-linear) ที่ทำการแบ่งข้อมูลเป็นกลุ่มย่อยๆ ตามค่าของคุณลักษณะต่างๆ เพื่อให้สามารถตัดสินใจได้ โมเดลนี้สามารถอธิบายการตัดสินใจได้ง่ายและเหมาะสำหรับการทำความเข้าใจวิธีการที่โมเดลตัดสินใจจากข้อมูลที่ป้อนเข้าไป </div>', unsafe_allow_html=True)
    st.image("https://mlpills.dev/wp-content/uploads/2023/11/image-27.png", caption="รูปภาพจาก https://mlpills.dev/machine-learning/introduction-to-decision-trees/")
    
    st.markdown('<div class="text_indent">2. K-Nearest Neighbors (KNN) – โมเดลที่ง่ายแต่มีประสิทธิภาพ โดยการจัดประเภทข้อมูลใหม่ๆ ขึ้นอยู่กับความใกล้เคียงของข้อมูลนั้นกับข้อมูลในชุดฝึกฝน โมเดลนี้เชื่อมโยงสมมติฐานที่ว่า สิ่งที่คล้ายกันมักจะอยู่ใกล้เคียงกัน </div>', unsafe_allow_html=True)
    st.image("https://intuitivetutorial.com/wp-content/uploads/2023/04/knn-1.png", caption="รูปภาพจาก https://intuitivetutorial.com/2023/04/07/k-nearest-neighbors-algorithm/")

    st.markdown('<div class="text_indent">3. Support Vector Regression (SVR) – เทคนิคการถดถอย (Regression) ที่มุ่งหาค่าที่ดีที่สุดในขณะที่อนุญาตให้มีขอบเขตของข้อผิดพลาด โดย SVR เหมาะสมกับข้อมูลในมิติสูงและเมื่อมีความสัมพันธ์ที่ไม่เป็นเชิงเส้นระหว่างข้อมูล </div>', unsafe_allow_html=True)
    st.image("https://i0.wp.com/spotintelligence.com/wp-content/uploads/2024/05/support-vector-regression-svr.jpg?fit=1200%2C675&ssl=1", caption="รูปภาพจาก https://spotintelligence.com/2024/05/08/support-vector-regression-svr/")
    
    st.markdown('<div class="text_indent">4. Ensemble Method (Stacking) คือเทคนิคที่ใช้การรวมหลายๆ โมเดลเข้าด้วยกันเพื่อให้ผลลัพธ์ที่แม่นยำยิ่งขึ้น โดยจะนำผลลัพธ์จากโมเดลต่างๆ มารวมกันแล้วใช้โมเดลใหม่ในการทำนายผลสุดท้าย กระบวนการนี้ช่วยลดความผิดพลาดของโมเดลเดียว และสามารถทำให้ผลลัพธ์มีความแม่นยำมากขึ้น</div>', unsafe_allow_html=True)
    st.image("https://journals.sagepub.com/cms/10.1177/00131644221117193/asset/images/large/10.1177_00131644221117193-fig3.jpeg", caption="รูปภาพจาก https://journals.sagepub.com/doi/10.1177/00131644221117193")
    
    st.markdown('<div class="text_indent">5. Neural Networks – โมเดลที่ได้รับแรงบันดาลใจจากสมองมนุษย์ ใช้สำหรับการจดจำลวดลายและการทำนายผล Neural Networks เป็นโมเดลที่มีความยืดหยุ่นสูงและสามารถจับความสัมพันธ์ที่ซับซ้อนได้จากข้อมูล </div>', unsafe_allow_html=True)
    st.image("https://miro.medium.com/v2/resize:fit:2000/1*cuTSPlTq0a_327iTPJyD-Q.png", caption="https://medium.com/towards-data-science/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464")
    
    #การประยุก
    st.markdown('<div class="big-font">การนำมาประยุกต์ใช้ในการทำโปรเจค</div>', unsafe_allow_html=True)
    st.markdown('<div class="highlight">การทำนายราคาปิดของหุ้น SET ในประเทศไทย</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">โปรเจคนี้มุ่งเน้นการทำนายราคาหุ้นโดยใช้ข้อมูลดัชนีตลาดหุ้น เช่น ราคาปิด, ราคาสูงสุด, ราคาต่ำสุด, ปริมาณการซื้อขาย และค่าเฉลี่ยเคลื่อนที่ 50 และ 200 วัน ข้อมูลเหล่านี้จะถูกเตรียมและปรับขนาดเพื่อให้เหมาะสมกับการทำนาย โดยใช้โมเดล  K-Nearest Neighbors (KNN), Decision Tree (D3), Support Vector Regression (SVR) , และ Ensemble Method (Stacking) ผลลัพธ์จะถูกประเมินด้วยตัวชี้วัดทางสถิติเพื่อหาว่าโมเดลใดให้ผลลัพธ์ที่ดีที่สุดในการทำนายราคาหุ้น</div>', unsafe_allow_html=True)