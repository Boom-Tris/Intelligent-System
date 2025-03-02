import streamlit as st

# ฟังก์ชันโหลด CSS


# ฟังก์ชันสำหรับหน้า Home
def display_home():
    st.title("ปัญญาประดิษฐ์ (Artificial Intelligence - AI)")
    
    
    # ส่วนของเนื้อหา
    # เมื่อเทคโนโลยีคิดได้เหมือนมนุษย์
    st.markdown('<div class="normal-text">✨ เมื่อเทคโนโลยีคิดได้เหมือนมนุษย์ ✨</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">ลองจินตนาการว่าโลกที่คุณตื่นขึ้นมาในทุกวันมีผู้ช่วยอัจฉริยะที่สามารถจัดการตารางชีวิต คาดการณ์สภาพอากาศ หรือแม้กระทั่งแนะนำภาพยนตร์ที่คุณจะต้องชอบ — สิ่งเหล่านี้เป็นไปได้ด้วย "ปัญญาประดิษฐ์" หรือ AI</div>', unsafe_allow_html=True)
    
    # AI (Artificial Intelligence) คืออะไร?
    st.markdown('<br><div class="big-font">🤖 AI (Artificial Intelligence) คืออะไร?</div>', unsafe_allow_html=True)
    st.image("https://lntedutech.com/wp-content/uploads/2024/04/Artificial-Intelligence-AI-scaled-1.jpg", caption="รูปภาพจาก https://lntedutech.com/blogs/harnessing-the-power-of-ai-in-the-education-and-professional-worlds/")
    st.markdown('<div class="text_indent">AI หรือ Artificial Intelligence (ปัญญาประดิษฐ์) คือเทคโนโลยีที่ทำให้ เครื่องจักรหรือระบบคอมพิวเตอร์สามารถเลียนแบบความคิด การตัดสินใจ และการเรียนรู้ได้เหมือนมนุษย์ มันเปรียบเสมือนสมองดิจิทัลที่ถูกออกแบบมาเพื่อช่วยให้ชีวิตของเราสะดวกสบายและฉลาดขึ้น</div>', unsafe_allow_html=True)
    st.markdown('<br><div class="normal-text">💡 ตัวอย่างง่าย ๆ ที่เราใช้งาน AI กันทุกวัน: <br> •ระบบแนะนำภาพยนตร์บน Netflix'
                '<br>•ผู้ช่วยเสียงอัจฉริยะอย่าง Siri และ Google Assistant'
                '<br> •ระบบค้นหาสินค้าบนแพลตฟอ</div>', unsafe_allow_html=True)

    # เบื้องหลังความฉลาดของ AI
    st.markdown('<div class="big-font">🧠 เบื้องหลังความฉลาดของ AI</div>', unsafe_allow_html=True)
    st.image("https://media.live-platforms.com/livepltf/Images/2023/Oct/article284-1.jpg", caption="รูปภาพจาก https://www.live-platforms.com/th/education/article/275-how-to-use-ai-for-business/")
    st.markdown('<div class="text_indent">เบื้องหลัง AI นั้นไม่ใช่เวทมนตร์ แต่เป็นกระบวนการที่เรียกว่า Machine Learning (ML) และ Deep Learning (DL) ซึ่งเปรียบเสมือนการฝึกสมองของคอมพิวเตอร์ให้รู้จักแยกแยะข้อมูล</div>', unsafe_allow_html=True)
    st.markdown('<br><div class="normal-text">📚 Machine Learning (ML): การให้คอมพิวเตอร์เรียนรู้จากข้อมูลจำนวนมาก โดยไม่ต้องตั้งโปรแกรมแบบละเอียด <br> ตัวอย่าง: การทำนายราคาหุ้น หรือการกรองสแปมในอีเมล'
                '<br><br>🌌 Deep Learning (DL): ระบบที่ซับซ้อนมากขึ้นโดยใช้โครงข่ายประสาทเทียม (Neural Networks)'
                '<br> ตัวอย่าง: AI ที่สามารถสร้างงานศิลปะหรือแปลงภาพให้เป็นแบบต่าง ๆค้นหาสินค้าบนแพลตฟอ</div>', unsafe_allow_html=True)
    
    #แยกรถ
    st.markdown('<div class="big-font">AI ก็เหมือนมนุษย์... แยกรถกับคนได้ยังไง? 🚗🤖</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">ลองนึกภาพว่าคุณเดินอยู่บนถนน แล้วต้องแยกแยะว่าอะไรคือ รถ และอะไรคือ คน การทำแบบนี้ดูเหมือนเป็นเรื่องง่ายใช่ไหม? เพราะเราคุ้นเคยกับรูปร่างของรถและคนจากประสบการณ์ชีวิตที่ผ่านมา <br>แต่ทำไมเราถึงแยกแยะได้ล่ะ? คำตอบง่าย ๆ คือ เราเคยเห็นมาก่อน! เพราะในสมองของเรามีข้อมูลจำนวนมากเกี่ยวกับรูปร่าง ลักษณะ และความแตกต่างระหว่างรถกับคน</div>', unsafe_allow_html=True)
    
    #ต้องเรียนรู้
    st.markdown('<br><div class="big-font">🧠 เมื่อ AI ต้องเรียนรู้แบบมนุษย์</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">AI ก็ต้องทำงานเหมือนกันครับ! ถ้าเราต้องการให้ AI แยกรถออกจากคนได้ มันจำเป็นต้อง เรียนรู้จากข้อมูล (Data) เช่น รูปภาพหรือข้อมูลลักษณะของรถและคน <br><br> 📸 ยิ่ง AI ได้เห็นข้อมูลหลากหลายรูปแบบ เช่น รถสีแดง รถบรรทุก หรือแม้แต่คนในชุดต่าง ๆ มันก็จะยิ่งฉลาดขึ้นและแยกแยะได้แม่นยำมากขึ้น</div>', unsafe_allow_html=True)
    
    #แต่ถ้าเจอ "รถแปลก
    st.markdown('<br><div class="big-font">🔍 แต่ถ้าเจอ "รถแปลก ๆ" ล่ะ?</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">สมมติว่าคุณเจอรถที่มีรูปร่างประหลาด หรือเป็นดีไซน์ใหม่ที่ไม่เคยเห็นมาก่อน คุณอาจจะสับสนและไม่มั่นใจว่าจะเรียกมันว่ารถหรือเปล่า <br><br> AI ก็เป็นแบบเดียวกันครับ! ถ้าข้อมูลที่ AI เคยเรียนรู้น้อยเกินไปหรือขาดความหลากหลาย มันจะมีโอกาส แยกแยะผิดพลาด เช่น ทายว่ารถแปลก ๆ เป็นอย่างอื่น</div>', unsafe_allow_html=True)
    
    #🚀 ทำไม Data ถึงสำคัญต่อ AI?
    st.markdown('<br><div class="big-font">🚀 แล้วทำไม Data ถึงสำคัญต่อ AI?</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">ถ้าเราต้องการสร้าง AI ที่ฉลาดและแม่นยำ เราจำเป็นต้อง<br>'
                '1. มีข้อมูล (Data) มากพอ: ให้ AI ได้เห็นตัวอย่างที่หลากหลาย<br>'
                '2. ข้อมูลที่มีคุณภาพ: ต้องชัดเจนและถูกต้อง เพื่อให้ AI ไม่เรียนรู้ผิด<br>'
                '3. การฝึกฝน (Training): ให้ AI ค่อย ๆ เข้าใจและสร้าง Model ที่สามารถแยกแยะข้อมูลได้ดี</div>', unsafe_allow_html=True)
    
    #📈 ยิ่งข้อมูลดี ยิ่ง AI แม่นยำ
    st.markdown('<br><div class="big-font">📈 ยิ่งข้อมูลดี ยิ่ง AI แม่นยำ</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">เมื่อ AI มีข้อมูลที่หลากหลายและมากพอ มันก็จะทำงานได้แม่นยำมากขึ้น เช่น<br>'
                '•แยกแยะสิ่งของในภาพได้แบบไม่มีข้อผิดพลาด<br>'
                '•ทำนายพฤติกรรมลูกค้าได้ตรงเป๊ะ<br>'
                '•ช่วยวินิจฉัยโรคได้อย่างแม่นยำ</div>', unsafe_allow_html=True)
    
    # ในโปรเจคนี้
    st.markdown('<br><div class="big-font">Machine Learning: เมื่อ AI เรียนรู้จากข้อมูลจริงเพื่อแก้ปัญหาในโลกจริง</div>', unsafe_allow_html=True)
    st.markdown('<br><div class="text_indent">AI จะฉลาดได้ต้องมีข้อมูลที่หลากหลายและมากพอให้เรียนรู้ จากนั้น AI จะค่อย ๆ พัฒนาความสามารถผ่านการสร้างโมเดลที่เหมาะสมในการแก้ปัญหา เช่น การแยกแยะภาพ การทำนายข้อมูล หรือการตัดสินใจที่ซับซ้อนในโปรเจกต์นี้ เราจะประยุกต์ใช้ เทคนิคการเรียนรู้ของ Machine Learning ที่หลากหลายและสำคัญ เช่น<br><br>'
                '•Decision Tree: เทคนิคที่สร้างการตัดสินใจเป็นลำดับขั้นตอน โดยใช้โครงสร้างแบบต้นไม้<br><br>'
                '•K-Nearest Neighbors (KNN): วิธีการที่ค้นหาข้อมูลที่คล้ายกันมากที่สุดรอบตัว เพื่อคาดการณ์ผลลัพธ์<br><br>'
                '•Support Vector Regression (SVR): เทคนิคที่ช่วยในการทำนายข้อมูลต่อเนื่อง<br><br>'
                '•Ensemble Method (Stacking): การรวมโมเดลหลายตัวเข้าด้วยกันเพื่อเพิ่มประสิทธิภาพในการทำนาย<br><br>'
                        ' Neural Networks: โครงข่ายประสาทเทียมที่สามารถเรียนรู้ข้อมูลเชิงลึกได้เหมือนสมองมนุษย์</div> <br>', unsafe_allow_html=True)
    
    st.markdown('<br><div class="big-font">🎯 การประเมินผลและเป้าหมายของโปรเจกต์</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ผลลัพธ์จากโมเดลเหล่านี้จะถูกวิเคราะห์และประเมินด้วย ตัวชี้วัดทางสถิติ เพื่อค้นหาโมเดลที่มีประสิทธิภาพดีที่สุด โดยเป้าหมายหลักคือการนำโมเดลมาแก้ปัญหาในสถานการณ์จริง พร้อมเปรียบเทียบประสิทธิภาพระหว่างแต่ละโมเดล เพื่อหาแนวทางการประยุกต์ใช้ที่เหมาะสมที่สุด</div>', unsafe_allow_html=True)
    #โมเดลการเรียนรู้
    st.markdown('<br><div class="big-font"> โมเดลการเรียนรู้ </div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">1. Decision Tree – โมเดลที่ไม่เป็นเชิงเส้น (Non-linear) ที่ทำการแบ่งข้อมูลเป็นกลุ่มย่อยๆ ตามค่าของคุณลักษณะต่างๆ เพื่อให้สามารถตัดสินใจได้ โมเดลนี้สามารถอธิบายการตัดสินใจได้ง่ายและเหมาะสำหรับการทำความเข้าใจวิธีการที่โมเดลตัดสินใจจากข้อมูลที่ป้อนเข้าไป </div><br>', unsafe_allow_html=True)
    st.image("https://mlpills.dev/wp-content/uploads/2023/11/image-27.png", caption="รูปภาพจาก https://mlpills.dev/machine-learning/introduction-to-decision-trees/")
    
    st.markdown('<div class="text_indent">2. K-Nearest Neighbors (KNN) – โมเดลที่ง่ายแต่มีประสิทธิภาพ โดยการจัดประเภทข้อมูลใหม่ๆ ขึ้นอยู่กับความใกล้เคียงของข้อมูลนั้นกับข้อมูลในชุดฝึกฝน โมเดลนี้เชื่อมโยงสมมติฐานที่ว่า สิ่งที่คล้ายกันมักจะอยู่ใกล้เคียงกัน </div><br>', unsafe_allow_html=True)
    st.image("https://intuitivetutorial.com/wp-content/uploads/2023/04/knn-1.png", caption="รูปภาพจาก https://intuitivetutorial.com/2023/04/07/k-nearest-neighbors-algorithm/")

    st.markdown('<div class="text_indent">3. Support Vector Regression (SVR) – เทคนิคการถดถอย (Regression) ที่มุ่งหาค่าที่ดีที่สุดในขณะที่อนุญาตให้มีขอบเขตของข้อผิดพลาด โดย SVR เหมาะสมกับข้อมูลในมิติสูงและเมื่อมีความสัมพันธ์ที่ไม่เป็นเชิงเส้นระหว่างข้อมูล </div><br>', unsafe_allow_html=True)
    st.image("https://i0.wp.com/spotintelligence.com/wp-content/uploads/2024/05/support-vector-regression-svr.jpg?fit=1200%2C675&ssl=1", caption="รูปภาพจาก https://spotintelligence.com/2024/05/08/support-vector-regression-svr/")
    
    st.markdown('<div class="text_indent">4. Ensemble Method (Stacking) คือเทคนิคที่ใช้การรวมหลายๆ โมเดลเข้าด้วยกันเพื่อให้ผลลัพธ์ที่แม่นยำยิ่งขึ้น โดยจะนำผลลัพธ์จากโมเดลต่างๆ มารวมกันแล้วใช้โมเดลใหม่ในการทำนายผลสุดท้าย กระบวนการนี้ช่วยลดความผิดพลาดของโมเดลเดียว และสามารถทำให้ผลลัพธ์มีความแม่นยำมากขึ้น </div><br>', unsafe_allow_html=True)
    st.image("https://journals.sagepub.com/cms/10.1177/00131644221117193/asset/images/large/10.1177_00131644221117193-fig3.jpeg", caption="รูปภาพจาก https://journals.sagepub.com/doi/10.1177/00131644221117193")
    
    st.markdown('<div class="text_indent">5. Neural Networks – โมเดลที่ได้รับแรงบันดาลใจจากสมองมนุษย์ ใช้สำหรับการจดจำลวดลายและการทำนายผล Neural Networks เป็นโมเดลที่มีความยืดหยุ่นสูงและสามารถจับความสัมพันธ์ที่ซับซ้อนได้จากข้อมูล </div>', unsafe_allow_html=True)
    st.image("https://miro.medium.com/v2/resize:fit:2000/1*cuTSPlTq0a_327iTPJyD-Q.png", caption="รูปภาพจาก https://medium.com/towards-data-science/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464")
    
    #การประยุก
    st.markdown('<div class="big-font">การนำมาประยุกต์ใช้ในการทำโปรเจค</div>', unsafe_allow_html=True)
    st.markdown('<div class="highlight">การทำนายราคาปิดของหุ้น SET ในประเทศไทย</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ในโลกการลงทุน การทำนายราคาหุ้นที่แม่นยำถือเป็นกุญแจสำคัญในการตัดสินใจที่ชาญฉลาด โปรเจกต์นี้จึงมุ่งเน้นการ พัฒนาระบบทำนายราคาหุ้น โดยใช้ ข้อมูลดัชนีตลาดหุ้น ที่สำคัญ เช่น<br>'
                '📈 ราคาปิด (Closing Price) <br>'
                '📊 ราคาสูงสุด-ต่ำสุด (High & Low Prices)<br>'
                '💸 ปริมาณการซื้อขาย (Trading Volume)<br>'
                '📅 ค่าเฉลี่ยเคลื่อนที่ 50 และ 200 วัน (Moving Averages) /div>', unsafe_allow_html=True)
    
    st.markdown('<br><div class="highlight">⚙️ กระบวนการและเทคนิคการทำนายที่ใช้</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ข้อมูลเหล่านี้จะถูก เตรียมและปรับขนาด (Preprocessing) เพื่อให้เหมาะสมกับการวิเคราะห์และทำนาย จากนั้นเราจะนำ โมเดล Machine Learning ชั้นนำมาประยุกต์ใช้ ได้แก่:<br><br>'
                '🔍 K-Nearest Neighbors (KNN): หาความคล้ายคลึงของข้อมูลเพื่อนบ้านใกล้เคียงเพื่อทำนายราคา<br><br>'
                '🌳 Decision Tree (D3): ตัดสินใจแบบลำดับขั้นตอนจากข้อมูลที่ป้อนเข้า<br><br>'
                '📐 Support Vector Regression (SVR): ช่วยทำนายข้อมูลที่มีความต่อเนื่องด้วยเส้นที่เหมาะสมที่สุด<br><br>'
                '🧩 Ensemble Method (Stacking): รวมโมเดลหลายตัวเข้าด้วยกันเพื่อเพิ่มประสิทธิภาพการทำนาย <br><br></div>', unsafe_allow_html=True)
   
    st.markdown('<div class="highlight">แอปวิเคราะห์เสียง Speech และ Music</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ในยุคที่เทคโนโลยีด้านเสียงและภาษาเป็นที่นิยมมากขึ้น การใช้ Convolutional Neural Networks (CNN) ในการจำแนกเสียงกลายเป็นเครื่องมือสำคัญที่ช่วยให้ระบบสามารถเข้าใจและแยกแยะเสียงต่างๆ ได้อย่างมีประสิทธิภาพ CNN ซึ่งเดิมทีถูกใช้ในการประมวลผลภาพ ถูกปรับใช้กับการวิเคราะห์เสียงด้วยการแปลงสัญญาณเสียงให้อยู่ในรูปแบบของ Spectrogram หรือ MFCCs (Mel-Frequency Cepstral Coefficients) ซึ่งคล้ายกับภาพ 2 มิติ<br></div>', unsafe_allow_html=True)
    st.markdown('<br><div class="highlight">⚙️ กระบวนการและเทคนิคการทำนายที่ใช้</div>', unsafe_allow_html=True) 
    st.markdown('<div class="text_indent">ในการวิเคราะห์และจำแนกเสียงด้วย Convolutional Neural Networks (CNN) มีกระบวนการและเทคนิคสำคัญที่ใช้เพื่อให้ได้ผลลัพธ์ที่มีประสิทธิภาพ ดังนี้:<br><br>'
                '🔍การแปลงสัญญาณเสียงเป็น Spectrogram หรือ MFCCs<br><br>'
                '🧠การออกแบบโครงสร้าง CNN<br><br>'
                '📊การฝึกและประเมินโมเดล<br><br>'
                '🚀การปรับปรุงโมเดล<br><br></div>', unsafe_allow_html=True)           