import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import os
from pathlib import Path
import yt_dlp as youtube_dl
from pydub import AudioSegment
import tempfile

MODEL_URL = 'https://drive.google.com/uc?id=1acfRIXq7Ldee-Z2gCLjqWMtaCWKxptne'

# ที่เก็บโมเดล
base_path = Path(__file__).parent.parent / "NL"
model_path = base_path / "model.h5"

# ดาวน์โหลดโมเดลหากยังไม่มี
if not model_path.exists():
    os.makedirs(base_path, exist_ok=True)
    gdown.download(MODEL_URL, str(model_path), quiet=False)

# โหลดโมเดล
model = load_model(model_path, compile=False)

# ฟังก์ชันดึง features จากไฟล์เสียง
def extract_features(audio_path):
    if not os.path.exists(audio_path):
        st.error(f"ไม่พบไฟล์เสียง: {audio_path}")
        return None
    
    y, sr = librosa.load(audio_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

# ฟังก์ชันแปลงไฟล์เสียงเป็น MP3
def convert_to_mp3(input_path):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()  # ปิดไฟล์เพื่อป้องกันปัญหาการเข้าถึง
        output_path = temp_file.name
        
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="mp3")
        return output_path
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการแปลงไฟล์: {str(e)}")
        return None

# ฟังก์ชันดาวน์โหลดและแปลง YouTube เป็น MP3
def download_youtube_audio(url):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        output_path = temp_file.name
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'outtmpl': output_path
        }
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        return output_path
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการดาวน์โหลด YouTube: {str(e)}")
        return None

# ฟังก์ชันโหลดและทำนายเสียง
def display_nn_model():
    st.write("กำลังประมวลผล...กรุณารอ")

    audio_option = st.radio("เลือกประเภทเสียง:", ["Speech", "Music", "เลือกไฟล์เอง", "ลิ้งค์ YouTube"], key="audio_option")

    audio_path = None

    if audio_option == "เลือกไฟล์เอง":
        uploaded_file = st.file_uploader("อัพโหลดไฟล์เสียง", type=["wav", "mp3"])
        if uploaded_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(uploaded_file.read())
            temp_file.close()
            audio_path = temp_file.name
    elif audio_option == "ลิ้งค์ YouTube":
        youtube_url = st.text_input("ใส่ลิ้งค์ YouTube:")
        if youtube_url:
            audio_path = download_youtube_audio(youtube_url)

    if not audio_path or not os.path.exists(audio_path):
        st.warning("กรุณาเลือกไฟล์หรือใส่ลิ้งค์ที่ถูกต้อง")
        return

    mel_spec = extract_features(audio_path)
    if mel_spec is None:
        return

    max_len = 1320
    if mel_spec.shape[1] < max_len:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])))
    elif mel_spec.shape[1] > max_len:
        mel_spec = mel_spec[:, :max_len]

    mel_spec = mel_spec[..., np.newaxis]

    prediction = model.predict(np.expand_dims(mel_spec, axis=0))[0][0]
    speech_prob = prediction * 100
    music_prob = (1 - prediction) * 100

    st.write(f"Speech Probability: {speech_prob:.2f}%")
    st.progress(int(speech_prob))
    st.write(f"Music Probability: {music_prob:.2f}%")
    st.progress(int(music_prob))

