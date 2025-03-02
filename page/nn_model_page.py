import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import os
from pathlib import Path

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
    y, sr = librosa.load(audio_path, sr=16000)  # ลดตัวอย่างเสียงให้เร็วขึ้น
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # คำนวณ Mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # เปลี่ยนค่า Mel spectrogram เป็น dB
    return log_mel_spec

# โหลดโมเดลครั้งแรกและเก็บไว้ในตัวแปร
model = load_model(model_path, compile=False)

# ฟังก์ชันดึง features และทำนายเสียง
def display_nn_model():
    st.write("กำลังประมวลผล...กรุณารอ")  # แสดงข้อความระหว่างการประมวลผล

    # ให้ผู้ใช้เลือกประเภทของเสียงก่อน
    audio_option = st.radio("เลือกประเภทเสียงที่ต้องการทดสอบ:", ["Speech", "Music", "เลือกไฟล์ของคุณเอง"], key="audio_option")

    # ถ้าเลือก Speech หรือ Music ให้แสดงตัวเลือกในการอัพโหลดไฟล์
    uploaded_file = None
    if audio_option == "Speech":
        audio_path = file_speech
    elif audio_option == "Music":
        audio_path = file_music
    elif audio_option == "เลือกไฟล์ของคุณเอง":
        uploaded_file = st.file_uploader("อัพโหลดไฟล์เสียงของคุณเอง", type=["wav", "mp3"])

   
   

    # ดึง features จากไฟล์เสียง
    mel_spec = extract_features(audio_path)

    # ปรับขนาดของ Mel Spectrogram
    max_len = 1320  # ขนาดที่โมเดลคาดหวัง
    if mel_spec.shape[1] < max_len:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])))
    elif mel_spec.shape[1] > max_len:
        mel_spec = mel_spec[:, :max_len]

    mel_spec = mel_spec[..., np.newaxis]  # เพิ่มมิติให้เหมาะกับโมเดล

    # ทำนายเสียง
    prediction = model.predict(np.expand_dims(mel_spec, axis=0))[0][0]
    speech_prob = prediction * 100
    music_prob = (1 - prediction) * 100

    st.write(f"Speech Probability: {speech_prob:.2f}%")
    progress_bar_speech = st.progress(int(speech_prob))  # ใช้ค่า speech_prob ตรงๆ
    st.write(f"Music Probability: {music_prob:.2f}%")
   
    progress_bar_music = st.progress(int(music_prob))    # ใช้ค่า music_prob ตรงๆ


