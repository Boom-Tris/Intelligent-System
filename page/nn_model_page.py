import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os
from pathlib import Path
import tempfile

# กำหนดที่เก็บไฟล์โมเดล
base_path = Path(__file__).parent.parent / "NL"
model_path = base_path / "model.h5"

# โหลดโมเดล
model = load_model(model_path, compile=False)

# ฟังก์ชันดึง features จากไฟล์เสียง
def extract_features(audio_path):
    if not os.path.exists(audio_path):
        st.error(f"ไม่พบไฟล์เสียง: {audio_path}")
        return None

    y, sr = librosa.load(audio_path, sr=16000)  # ลดตัวอย่างเสียงให้เร็วขึ้น
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # คำนวณ Mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # เปลี่ยนค่า Mel spectrogram เป็น dB
    return log_mel_spec

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
        audio_path = str(Path(__file__).parent.parent / "data" / "Speech.wav")
    elif audio_option == "Music":
        audio_path = str(Path(__file__).parent.parent / "data" / "COCKTAIL.wav")
    elif audio_option == "เลือกไฟล์ของคุณเอง":
        uploaded_file = st.file_uploader("อัพโหลดไฟล์เสียงของคุณเอง", type=["wav", "mp3", "mp4"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.write(uploaded_file.read())
            temp_file.close()
            audio_path = temp_file.name
    elif audio_option == "ลิ้งค์ YouTube":
        st.write("กรุณาใช้เว็บไซต์หรือแอปพลิเคชันแปลง YouTube เป็นไฟล์เสียง (เช่น ytmp3.cc) แล้วอัปโหลดไฟล์เสียงด้านล่าง")
        uploaded_file = st.file_uploader("อัพโหลดไฟล์เสียงจาก YouTube", type=["wav", "mp3", "mp4"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
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
    if "temp_file" in locals() and os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

# เรียกใช้งานฟังก์ชันหลัก
