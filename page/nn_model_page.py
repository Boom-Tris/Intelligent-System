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

    # เลือกไฟล์เสียงจาก Streamlit UI
    audio_option = st.radio("เลือกไฟล์เสียงที่ต้องการทดสอบ:", ["Speech", "Music"], key="audio_option")

    # กำหนดไฟล์เสียงตามตัวเลือก
    if audio_option == "Speech":
        audio_path = file_speech
    else:
        audio_path = file_music

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
    st.write("กำลังทำนาย...กรุณารอ")  # บอกสถานะการทำนาย
    prediction = model.predict(np.expand_dims(mel_spec, axis=0))[0][0]
    speech_prob = prediction * 100
    music_prob = (1 - prediction) * 100

    # ตรวจสอบค่าไม่ให้เกินขอบเขต 0-100
    speech_prob = max(0, min(100, speech_prob))
    music_prob = max(0, min(100, music_prob))

    # แสดงผลเปอร์เซ็นต์ในรูปแบบหลอด
    st.progress(speech_prob)  # แสดงเปอร์เซ็นต์ของ Speech
    st.write(f"Speech Probability: {speech_prob:.2f}%")

    st.progress(music_prob)  # แสดงเปอร์เซ็นต์ของ Music
    st.write(f"Music Probability: {music_prob:.2f}%")

# เรียกใช้งานฟังก์ชัน

