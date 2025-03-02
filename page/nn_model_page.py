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
import shutil
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
def convert_webm_to_wav(input_file, temp_wav_file):
    command = [
        "ffmpeg",
        "-i", input_file,
        temp_wav_file
    ]
    subprocess.run(command, check=True)

def convert_wav_to_mp3(wav_file, output_file):
    command = [
        "ffmpeg",
        "-i", wav_file,
        "-acodec", "libmp3lame",
        output_file
    ]
    subprocess.run(command, check=True)

st.title("WebM to MP3 Converter")

uploaded_file = st.file_uploader("Upload a webm file", type="webm")

if uploaded_file is not None:
    input_path = f"/tmp/{uploaded_file.name}"
    wav_path = f"/tmp/{uploaded_file.name}.wav"
    mp3_path = f"/tmp/{uploaded_file.name}.mp3"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Convert to MP3"):
        try:
            convert_webm_to_wav(input_path, wav_path)  # แปลงเป็น wav ก่อน
            convert_wav_to_mp3(wav_path, mp3_path)  # แปลงเป็น mp3
            st.success(f"Conversion successful! MP3 saved to {mp3_path}")
            st.audio(mp3_path)
        except Exception as e:
            st.error(f"Error during conversion: {e}")
def convert_to_mp3(input_path):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio = AudioSegment.from_file(input_path)
        audio.export(temp_file.name, format="mp3")
        return temp_file.name
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการแปลงไฟล์เสียง: {str(e)}")
        return None

# ฟังก์ชันดาวน์โหลดและแปลง YouTube เป็นไฟล์ MP3
def download_youtube_audio(url):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_file.name,
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # ตรวจสอบไฟล์ที่ดาวน์โหลดมา (เช็คว่า ffmpeg สามารถเปิดได้หรือไม่)
        try:
            subprocess.run(['ffmpeg', '-v', 'error', '-i', temp_file.name], check=True)
        except subprocess.CalledProcessError:
            st.error(f"ไม่สามารถแปลงไฟล์ {temp_file.name} ได้ เนื่องจากเป็นไฟล์ที่ไม่รองรับ")
            os.unlink(temp_file.name)
            return None
        
        # แปลงเป็น MP3
        mp3_path = convert_to_mp3(temp_file.name)
        os.unlink(temp_file.name)  # ลบไฟล์ webm หลังแปลงเสร็จ
        return mp3_path
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการดาวน์โหลด YouTube: {str(e)}")
        return None


# ฟังก์ชันหลัก
def display_nn_model():
    st.write("กำลังประมวลผล...กรุณารอ")

    # ให้ผู้ใช้เลือกประเภทของเสียงก่อน
    audio_option = st.radio("เลือกประเภทเสียงที่ต้องการทดสอบ:", ["Speech", "Music", "เลือกไฟล์ของคุณเอง", "ลิ้งค์ YouTube"])

    audio_path = None  # กำหนดค่าเริ่มต้นให้กับ audio_path

    if audio_option == "Speech":
        audio_path = str(file_speech)
    elif audio_option == "Music":
        audio_path = str(file_music)
    elif audio_option == "เลือกไฟล์ของคุณเอง":
        uploaded_file = st.file_uploader("อัพโหลดไฟล์เสียงของคุณเอง", type=["wav", "mp3"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
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

    # ดึง features จากไฟล์เสียง
    mel_spec = extract_features(audio_path)
    if mel_spec is None:
        return  # หยุดถ้ามีข้อผิดพลาดในการโหลดไฟล์เสียง

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
    st.progress(int(speech_prob))
    st.write(f"Music Probability: {music_prob:.2f}%")
    st.progress(int(music_prob))

    # ลบไฟล์ชั่วคราวหลังใช้งานเสร็จ
    if "temp_file" in locals() and os.path.exists(temp_file.name):
        os.unlink(temp_file.name)
    if "audio_path" in locals() and audio_path.startswith("/tmp") and os.path.exists(audio_path):
        os.unlink(audio_path)
