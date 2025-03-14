import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import os
from pathlib import Path
import tempfile

# กำหนดลิงก์ดาวน์โหลดไฟล์จาก Google Drive
MODEL_URL = 'https://drive.google.com/uc?id=1acfRIXq7Ldee-Z2gCLjqWMtaCWKxptne'

# กำหนดที่เก็บไฟล์โมเดล
base_path = Path(__file__).parent.parent / "NL"
model_path = base_path / "model.h5"

# ดาวน์โหลดไฟล์โมเดลจาก Google Drive หากยังไม่มีในระบบ
if not model_path.exists():
    os.makedirs(base_path, exist_ok=True)
    gdown.download(MODEL_URL, str(model_path), quiet=False)

# กำหนดไฟล์เสียงที่ต้องการใช้
file_path = Path(__file__).parent.parent / "data"
file_speech = file_path / "Speech.wav"
file_music = file_path / "COCKTAIL.wav"

# ฟังก์ชันดึง features จากไฟล์เสียง
def extract_features(audio_path):
    if not os.path.exists(audio_path):
        st.error(f"ไม่พบไฟล์เสียง: {audio_path}")
        return None

    y, sr = librosa.load(audio_path, sr=16000)  # ลดตัวอย่างเสียงให้เร็วขึ้น
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # คำนวณ Mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # เปลี่ยนค่า Mel spectrogram เป็น dB
    return log_mel_spec

# โหลดโมเดลครั้งแรกและเก็บไว้ในตัวแปร
model = load_model(model_path, compile=False)

# ฟังก์ชันหลัก
def nn_modele():
    
    st.write("กรุณาเลือกประเภทเสียงที่ต้องการทดสอบ:")

    # ให้ผู้ใช้เลือกประเภทของเสียง
    audio_option = st.radio(
        "เลือกประเภทเสียง:",
        ["Speech", "Music", "เลือกไฟล์ของคุณเอง"]
    )

    audio_path = None  # กำหนดค่าเริ่มต้นให้กับ audio_path
    temp_file = None  # เพิ่มตัวแปร temp_file ที่เป็น None เริ่มต้น

    if audio_option == "Speech":
        audio_path = str(file_speech)
    elif audio_option == "Music":
        audio_path = str(file_music)
    elif audio_option == "เลือกไฟล์ของคุณเอง":
        uploaded_file = st.file_uploader("อัพโหลดไฟล์เสียงของคุณเอง", type=["wav", "mp3", "mp4"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file.close()
            audio_path = temp_file.name

    if not audio_path or not os.path.exists(audio_path):
        st.warning("กรุณาเลือกประเภทเสียงหรืออัพโหลดไฟล์เสียงให้ถูกต้อง")
        return

    # ดึง features จากไฟล์เสียง
    mel_spec = extract_features(audio_path)
    if mel_spec is None:
        return  # หยุดถ้ามีข้อผิดพลาดในการโหลดไฟล์เสียง

    # ปรับขนาดของ Mel Spectrogram
    max_len = 1320  # ขนาดที่โมเดลคาดหวัง
    if mel_spec.shape[1] < max_len:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])))
    else:
        mel_spec = mel_spec[:, :max_len]  # ครอปให้ได้ขนาดที่ต้องการ

    mel_spec = mel_spec[..., np.newaxis]  # เพิ่มมิติให้เหมาะกับโมเดล

    # ตรวจสอบขนาดของข้อมูลที่ป้อนเข้าโมเดล
    if mel_spec.shape != (128, 1320, 1):
        st.error(f"ขนาดของข้อมูลที่ป้อนเข้าโมเดลไม่ถูกต้อง: {mel_spec.shape}")
        return

    # ทำนายเสียง
    try:
        prediction = model.predict(np.expand_dims(mel_spec, axis=0))[0][0]
        speech_prob = prediction * 100
        music_prob = (1 - prediction) * 100

        st.write(f"Speech Probability: {speech_prob:.2f}%")
        st.progress(int(speech_prob))
        st.write(f"Music Probability: {music_prob:.2f}%")
        st.progress(int(music_prob))
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนายเสียง: {str(e)}")

    # ลบไฟล์ชั่วคราวหลังใช้งานเสร็จ
    if temp_file and os.path.exists(temp_file.name):
        os.unlink(temp_file.name)
    if "audio_path" in locals() and audio_path.startswith("/tmp") and os.path.exists(audio_path):
        os.unlink(audio_path)







def display_nn_model():
    st.title("แอปวิเคราะห์เสียง Speech และ Music")
    st.markdown('<div class="text_indent">ในยุคที่เทคโนโลยีด้านเสียงและภาษาเป็นที่นิยมมากขึ้น การใช้ Convolutional Neural Networks (CNN) ในการจำแนกเสียงกลายเป็นเครื่องมือสำคัญที่ช่วยให้ระบบสามารถเข้าใจและแยกแยะเสียงต่างๆ ได้อย่างมีประสิทธิภาพ CNN ซึ่งเดิมทีถูกใช้ในการประมวลผลภาพ ถูกปรับใช้กับการวิเคราะห์เสียงด้วยการแปลงสัญญาณเสียงให้อยู่ในรูปแบบของ Spectrogram หรือ MFCCs (Mel-Frequency Cepstral Coefficients) ซึ่งคล้ายกับภาพ 2 มิติ</div><br>', unsafe_allow_html=True)
    st.image("https://librosa.org/doc/main/_images/librosa-feature-mfcc-1_00.png", caption="รูปภาพจาก https://librosa.org/doc/main/_images/librosa-feature-mfcc-1_00.png")
    st.markdown('<div class="highlight">ทำไมต้องใช้ CNN กับการจำแนกเสียง?)</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การใช้ CNN (Convolutional Neural Network) กับการจำแนกเสียงเป็นเรื่องที่มีประโยชน์เพราะ CNN สามารถจับลักษณะเฉพาะของข้อมูลที่มีรูปแบบซับซ้อนได้ดี โดยเสียงสามารถแปลงเป็นสเปกโตรแกรม (spectrogram) ซึ่งเป็นภาพที่แสดงลักษณะความถี่ของเสียงในช่วงเวลาต่าง ๆ ได้ การใช้ CNN กับสเปกโตรแกรมช่วยให้โมเดลสามารถเรียนรู้ลักษณะของเสียงในแต่ละช่วงเวลาและจำแนกประเภทเสียงต่าง ๆ ได้อย่างมีประสิทธิภาพ โดยที่ CNN จะเรียนรู้ลักษณะการเปลี่ยนแปลงของความถี่และรูปแบบของเสียงที่มีความสำคัญในกระบวนการจำแนก.</div><br><br>', unsafe_allow_html=True)
    
    st.markdown('<div class="highlight">หลักการของการใช้ CNN (Convolutional Neural Network) ในการจำแนกเสียงสามารถ?)</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">แปลงเสียงเป็นสเปกโตรแกรม:เสียงที่เป็นสัญญาณแบบคลื่น (waveform) จะถูกแปลงเป็นสเปกโตรแกรม (spectrogram) ซึ่งเป็นภาพที่แสดงการกระจายของความถี่ของเสียงในช่วงเวลาต่าง ๆ โดยใช้เทคนิคเช่น Short-Time Fourier Transform (STFT). สิ่งนี้ทำให้เสียงกลายเป็นข้อมูลที่มีลักษณะเป็นภาพ 2D ซึ่ง CNN สามารถประมวลผลได้ดี</div><br><br>', unsafe_allow_html=True)
    st.image("https://librosa.org/doc/0.9.2/_images/sphx_glr_plot_hprss_002.png", caption="รูปภาพจาก https://librosa.org/doc/0.9.2/_images/sphx_glr_plot_hprss_002.png")
    st.markdown('<div class="text_indent">การทำงานของ CNN:CNN ใช้เลเยอร์การ convolutions เพื่อดึงคุณลักษณะ (features) จากภาพ โดยการใช้ฟิลเตอร์ (filters) ที่ผ่านการฝึกฝนจากข้อมูล เพื่อจับลักษณะเฉพาะของสเปกโตรแกรม เช่น รูปแบบคลื่นเสียงที่มีความถี่เฉพาะในแต่ละช่วงเวลา. การใช้ฟิลเตอร์นี้ช่วยให้ CNN สามารถจับลักษณะของเสียงได้อย่างแม่นยำ</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การเรียนรู้และการจำแนก:เมื่อ CNN ได้เรียนรู้ลักษณะเฉพาะจากสเปกโตรแกรมของเสียงแล้ว โมเดลสามารถจำแนกเสียงออกเป็นประเภทต่าง ๆ ได้ เช่น การจำแนกเสียงของสัตว์, ดนตรี, หรือเสียงของคนพูด. เลเยอร์ที่มีการ pooling จะช่วยลดขนาดข้อมูลและเพิ่มประสิทธิภาพในการประมวลผล.</div><br><br>', unsafe_allow_html=True)
    st.image("https://www.asiatest.co.th/img/leak4.png", caption="รูปภาพจาก https://www.asiatest.co.th/img/leak4.png")
    
    st.markdown('<div class="big-font">การเตรียมข้อมูล (Data Preprocessing)</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ชุดข้อมูลที่ใช้ในการศึกษาคือ GTZAN ซึ่งสามารถดาวน์โหลดได้จากเว็บไซต์ Kaggle โดยชุดข้อมูลนี้แบ่งออกเป็น 2 คลาส คือ Music และ Speech ในแต่ละคลาสจะมีไฟล์เสียงทั้งหมด 389 ไฟล์สำหรับ Music และ 423 ไฟล์สำหรับ Speech โดยแต่ละไฟล์มีความยาวที่หลากหลาย ซึ่งข้อมูลทั้งหมดได้ถูกอัปโหลดลงใน Google Drive สำหรับการเข้าถึงและใช้งาน</div><br><br>', unsafe_allow_html=True)
   
    st.markdown('<div class="highlight"><br>Libraries Used</div>', unsafe_allow_html=True)
    code = '''
       import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
        '''
    st.code(code, language='python')
    st.markdown('<div class="text_indent">os:ไลบรารีนี้ใช้สำหรับการจัดการกับไฟล์และโฟลเดอร์ในระบบ เช่น การเข้าถึงไฟล์เสียงในโฟลเดอร์ และการดำเนินการเกี่ยวกับพาธของไฟล์ต่าง ๆ ในโปรเจค</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">librosa:ไลบรารีที่ใช้สำหรับการประมวลผลสัญญาณเสียง มันสามารถใช้ในการโหลดไฟล์เสียง, การแปลงข้อมูลเสียงเป็นสเปกโตรแกรม, การแยกคุณลักษณะของเสียง (features) เช่น Mel-frequency cepstral coefficients (MFCCs) ซึ่งเป็นข้อมูลที่ใช้ในการจำแนกเสียงได้ดี</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">numpy:ไลบรารีที่ใช้ในการคำนวณทางคณิตศาสตร์และการจัดการกับอาร์เรย์ (array) ซึ่งมีประโยชน์ในการจัดเก็บข้อมูลที่ได้จากการประมวลผลเสียงในรูปแบบของตัวเลข</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">tensorflow:เป็นเฟรมเวิร์กที่ใช้ในการสร้างและฝึกสอนโมเดลของปัญหาการเรียนรู้เชิงลึก (deep learning) โดยในกรณีนี้เราใช้ TensorFlow เพื่อสร้างโมเดลที่สามารถจำแนกเสียงจากข้อมูลที่เรามี</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">sklearn.model_selection.train_test_split:ใช้สำหรับการแบ่งชุดข้อมูลออกเป็นชุดฝึก (training) และชุดทดสอบ (testing) โดยการแบ่งข้อมูลออกเป็น 2 ส่วนเพื่อการฝึกสอนโมเดลและทดสอบประสิทธิภาพของโมเดล</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">matplotlib.pyplot:ใช้สำหรับการสร้างกราฟและการแสดงผลข้อมูลต่าง ๆ เช่น การแสดงภาพสเปกโตรแกรมของเสียงหรือการแสดงผลการฝึกสอนโมเดลในรูปแบบกราฟ</div><br><br>', unsafe_allow_html=True)

    st.markdown('<div class="highlight"><br>ฟังก์ชันการแปลงเสียงเป็น Mel Spectrogram</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ฟังก์ชัน extract_features ในตัวอย่างนี้ทำหน้าที่ในการโหลดไฟล์เสียงและแปลงมันเป็น Mel Spectrogram ซึ่งสามารถใช้ในการทำงานกับการรู้จำเสียง (Audio Recognition) หรือการวิเคราะห์เสียงในด้านต่างๆ ต่อไปนี้คือลำดับขั้นตอนการทำงานของฟังก์ชันนี้</div><br><br>', unsafe_allow_html=True)
    code = '''
# ฟังก์ชันโหลดเสียงและแปลงเป็น Mel Spectrogram
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        if y is None or len(y) == 0:
            raise ValueError(f"ไฟล์ {audio_file} ไม่มีข้อมูลเสียง")

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดไฟล์ {audio_file}: {e}")
        return None
 
        '''
    st.code(code, language='python')
    st.markdown('<div class="text_indent">โหลดไฟล์เสียงฟังก์ชันเริ่มต้นด้วยการใช้ librosa.load() เพื่อโหลดไฟล์เสียงที่ระบุในพารามิเตอร์ audio_file โดยจะคืนค่าตัวแปร y ซึ่งเก็บข้อมูลเสียงในรูปแบบของอาเรย์ และ sr ซึ่งเป็นอัตราตัวอย่าง (sample rate) ของไฟล์เสียงนั้นๆ</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การตรวจสอบข้อมูลเสียงฟังก์ชันตรวจสอบว่าไฟล์เสียงที่โหลดมีข้อมูลหรือไม่ โดยใช้ if y is None or len(y) == 0 หากไม่มีข้อมูลเสียง ฟังก์ชันจะขว้างข้อผิดพลาด (error) และแสดงข้อความที่ระบุว่าไฟล์เสียงไม่มีข้อมูล</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การแปลงเสียงเป็น Mel Spectrogramหากไฟล์เสียงโหลดได้สำเร็จ ฟังก์ชันจะใช้ librosa.feature.melspectrogram() เพื่อแปลงข้อมูลเสียงเป็น Mel Spectrogram ซึ่งจะช่วยให้เราเห็นพลังงานที่กระจายอยู่ในช่วงความถี่ต่างๆ โดยการกำหนดจำนวนเมลแบนด์ (n_mels) เป็น 128 และขีดจำกัดสูงสุดของความถี่ (fmax) ที่ 8000 Hz ซึ่งเป็นค่าที่เหมาะสมสำหรับการวิเคราะห์เสียงที่มีช่วงความถี่สูง</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การแปลง Mel Spectrogram เป็น Decibel (dB)การแปลง Mel Spectrogram เป็น Decibel (librosa.power_to_db()) จะช่วยให้เราเห็นรายละเอียดได้ดีขึ้น เพราะในรูปแบบนี้ค่าพลังงานที่สูงจะมีค่าในระดับสูงสุดในกราฟ ซึ่งเหมาะสำหรับการวิเคราะห์เสียงในด้านต่างๆ</div><br><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การจัดการข้อผิดพลาดหากเกิดข้อผิดพลาดระหว่างการโหลดไฟล์เสียงหรือแปลงข้อมูล ฟังก์ชันจะจับข้อผิดพลาดเหล่านั้นและแสดงข้อความที่บอกให้ทราบว่าเกิดปัญหาที่ไฟล์เสียงใด</div><br><br>', unsafe_allow_html=True)

    st.markdown('<div class="highlight"><br>เชื่อมต่อกับ Google Drive </div>', unsafe_allow_html=True)
    code = '''
from google.colab import drive
drive.mount('/content/drive')

 
        '''
    st.code(code, language='python')


    st.markdown('<div class="highlight"><br>การตรวจสอบและโหลดไฟล์เสียงจากไดเรกทอรีที่กำหนด พร้อมทั้งแปลงไฟล์เสียงเป็น Mel Spectrogram และแท็ก </div>', unsafe_allow_html=True)
    code = '''
import os

music_dir = "/content/drive/MyDrive/music_wav"
speech_dir = "/content/drive/MyDrive/speech_wav"

# ตรวจสอบว่าไดเรกทอรีมีอยู่
if os.path.exists(music_dir):
    print(f"Music directory exists: {music_dir}")
else:
    print(f"Music directory does not exist: {music_dir}")

if os.path.exists(speech_dir):
    print(f"Speech directory exists: {speech_dir}")
else:
    print(f"Speech directory does not exist: {speech_dir}")

# อ่านไฟล์ทั้งหมดจากไดเรกทอรี
music_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir) if f.endswith('.wav')]
speech_files = [os.path.join(speech_dir, f) for f in os.listdir(speech_dir) if f.endswith('.wav')]

# รวมไฟล์เสียงและแท็ก
features, labels = [], []

# สำหรับไฟล์เสียงเพลง
for file in music_files:
    mel_spec = extract_features(file)
    if mel_spec is not None:
        features.append(mel_spec)
        labels.append(0)  # 0 = music

# สำหรับไฟล์เสียงพูด
for file in speech_files:
    mel_spec = extract_features(file)
    if mel_spec is not None:
        features.append(mel_spec)
        labels.append(1)  # 1 = speech

if not features:
    raise ValueError("ไม่สามารถโหลดไฟล์เสียงได้ ตรวจสอบไฟล์ต้นฉบับอีกครั้ง")


 
        '''
    st.code(code, language='python')
    st.markdown('<div class="text_indent">ทำหน้าที่ในการโหลดไฟล์เสียงจากไดเรกทอรีที่กำหนด (music_wav และ speech_wav), แปลงไฟล์เสียงเป็น Mel Spectrogram, และแท็กประเภทเสียงเป็นเพลงหรือเสียงพูด ก่อนที่จะจัดเก็บข้อมูลในตัวแปร features และ labels เพื่อใช้ในการฝึกโมเดล</div><br><br>', unsafe_allow_html=True)

    st.markdown('<div class="highlight"><br>จัดการข้อมูลเสียงเพื่อเตรียมฝึกโมเดล</div>', unsafe_allow_html=True)
    code = '''
import numpy as np

# หาขนาดสูงสุดของมิติที่สองใน features
max_length = max([feature.shape[1] for feature in features])

# ทำการ padding ให้ขนาดในมิติที่สองเท่ากัน
features_padded = [np.pad(feature, ((0, 0), (0, max_length - feature.shape[1])), mode='constant') for feature in features]

# แปลง features_padded เป็น NumPy array
X = np.array(features_padded)
y = np.array(labels)

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# เปลี่ยนรูปร่างของ X_train และ X_test ให้มีมิติใหม่
X_train, X_test = X_train[..., np.newaxis], X_test[..., np.newaxis]


 
        '''
    st.code(code, language='python')


    st.markdown('<div class="highlight"><br>สร้าง CNN โมเดล</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">หลังจากที่เราได้ทำการเตรียมข้อมูลและตรวจสอบข้อมูลที่จะนำไปใช้เรียบร้อยแล้ว ขั้นตอนต่อไปคือการนำข้อมูลไปเทรนโมเดลกันนะครับ</div><br><br>', unsafe_allow_html=True)
    code = '''
model = models.Sequential([
    layers.InputLayer(shape=(X_train.shape[1], X_train.shape[2], 1)),  # ใช้ 'shape' แทน 'input_shape'

    # เพิ่ม Conv2D layer และ MaxPooling2D layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # เพิ่ม Conv2D layer และ MaxPooling2D layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # เพิ่ม Conv2D layer และ MaxPooling2D layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten ข้อมูลจาก 2D ให้เป็น 1D
    layers.Flatten(),

    # เพิ่ม Dense layer
    layers.Dense(128, activation='relu'),

    # เพิ่ม Dense layer สำหรับการทำนายผล
    layers.Dense(64, activation='relu'),  # เพิ่ม Dense layer ใหม่
    layers.Dense(1, activation='sigmoid')  # Layer output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

 
        '''
    st.code(code, language='python')
    st.markdown('<div class="highlight"><br>การฝึกโมเดล</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การฝึกโมเดลนี้จะทำให้โมเดลเรียนรู้จากข้อมูล X_train และ y_train เป็นจำนวน 100 รอบ (epochs) โดยอัปเดตน้ำหนักของโมเดลหลังจากการประมวลผล 32 ตัวอย่างในแต่ละรอบ พร้อมทั้งตรวจสอบผลการทำนายโดยใช้ข้อมูลทดสอบ X_test และ y_test ในระหว่างการฝึกเพื่อป้องกันการ overfitting</div><br><br>', unsafe_allow_html=True)
    code = '''
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
 
        '''
    

    st.code(code, language='python')
    
    st.markdown('<div class="highlight"><br>บันทึกโมเดล</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> บันทึกโมเดลที่ฝึกแล้ว ลงเป็นไฟล์ .h5 เพื่อให้นำไปใช้งานในภายหลังได้</div><br><br>', unsafe_allow_html=True)
    code = '''
model.save("model.h5")
print("โมเดลถูกบันทึกเรียบร้อยแล้ว 🎉")


 
        '''
    st.code(code, language='python')
    st.markdown('<div class="highlight"><br>ตัวอย่างการทำงานโมเดล</div>', unsafe_allow_html=True)
    nn_modele()