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
import subprocess

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

# ฟังก์ชันแปลงไฟล์ .mp4 เป็น .wav
def convert_mp4_to_wav(input_file, output_file):
    try:
        # ใช้ ffmpeg แปลงไฟล์ .mp4 เป็น .wav
        command = ["ffmpeg", "-v", "error", "-i", input_file, output_file]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"เกิดข้อผิดพลาดในการแปลงไฟล์ .mp4 เป็น .wav: {str(e)}")
        st.error(f"FFmpeg output: {e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}")
        return False

# ฟังก์ชันดาวน์โหลดและแปลง YouTube เป็นไฟล์ WAV
def download_youtube_audio(url):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_file.name,
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # แปลงเป็น WAV
        wav_path = f"{temp_file.name}.wav"
        if convert_mp4_to_wav(temp_file.name, wav_path):
            os.unlink(temp_file.name)  # ลบไฟล์ MP4 หลังแปลง
            return wav_path
        else:
            os.unlink(temp_file.name)
            return None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการดาวน์โหลด YouTube: {str(e)}")
        return None

# ฟังก์ชันหลัก
def display_nn_model():
    st.title("แอปวิเคราะห์เสียง Speech และ Music")
    st.write("กรุณาเลือกประเภทเสียงที่ต้องการทดสอบ:")

    # ให้ผู้ใช้เลือกประเภทของเสียง
    audio_option = st.radio(
        "เลือกประเภทเสียง:",
        ["Speech", "Music", "เลือกไฟล์ของคุณเอง", "ลิ้งค์ YouTube"]
    )

    audio_path = None  # กำหนดค่าเริ่มต้นให้กับ audio_path

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
    elif audio_option == "ลิ้งค์ YouTube":
        youtube_url = st.text_input("กรุณากรอกลิงก์ YouTube:")
        if youtube_url:
            audio_path = download_youtube_audio(youtube_url)

    if not audio_path or not os.path.exists(audio_path):
        st.warning("กรุณาเลือกประเภทเสียงหรืออัพโหลดไฟล์เสียงให้ถูกต้อง")
        return

    # ถ้าไฟล์เป็น .mp4 ให้แปลงเป็น .wav ก่อน
    if audio_path.endswith(".mp4"):
        wav_path = f"{audio_path}.wav"
        if convert_mp4_to_wav(audio_path, wav_path):
            audio_path = wav_path
        else:
            return

    # ดึง features จากไฟล์เสียง
    mel_spec = extract_features(audio_path)
    if mel_spec is None:
        return  # หยุดถ้ามีข้อผิดพลาดในการโหลดไฟล์เสียง

    # ปรับขนาดของ Mel Spectrogram
    max_len = 1320  # ขนาดที่โมเดลคาดหวัง
    if mel_spec.shape[1] < max_len:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])))

    mel_spec = mel_spec[..., np.newaxis]  # เพิ่มมิติให้เหมาะกับโมเดล

    # ทำนายเสียง
    prediction = model.predict(np.expand_dims(mel_spec, axis=0))[0][0]
    speech_prob = prediction * 100
    music_prob = (1 - prediction) * 100

    st.write(f"Speech Probability: {speech_prob:.2f}%")
    st.progress(int(speech_prob))
    st.write(f"Music Probability: {music_prob:.2f}%")
    st.progress(int(music_prob))

    # ลบไฟล์ชั่วคราวหลังใช้งานเสร็จ
    if "temp_file" in locals() and os.path.exists(temp_file.name):
        os.unlink(temp_file.name)
    if "audio_path" in locals() and audio_path.startswith("/tmp") and os.path.exists(audio_path):
        os.unlink(audio_path)

