import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import os

# กำหนดไฟล์เสียงที่ต้องการใช้
file_path =  Path(__file__).parent.parent / "data"

file_speech = file_path / "Speech.wav"
file_music = file_path / "COCKTAIL.wav"

# ฟังก์ชันดึง features จากไฟล์เสียง
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)  # โหลดไฟล์เสียง
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # คำนวณ Mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # เปลี่ยนค่า Mel spectrogram เป็น dB
    return log_mel_spec

# ฟังก์ชันดึง features จากไฟล์เสียง
def display_nn_model():
    # เลือกไฟล์เสียงจาก Streamlit UI
    audio_option = st.radio("เลือกไฟล์เสียงที่ต้องการทดสอบ:", ["Speech", "Music"], key="audio_option")

    # กำหนดไฟล์เสียงตามตัวเลือก
    if audio_option == "Speech":
        audio_path = file_speech
    else:
        audio_path = file_music

    # โหลดโมเดล
    base_path = Path(__file__).parent.parent / "NL"
    model = load_model(base_path / "model.h5", compile=False)

    # ดึง features จากไฟล์เสียง
    mel_spec = extract_features(audio_path)
    print(f"Mel spectrogram shape: {mel_spec.shape}")  # เพิ่ม print เพื่อตรวจสอบขนาด Mel Spectrogram

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

    return speech_prob, music_prob
