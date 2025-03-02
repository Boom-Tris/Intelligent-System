def display_demo_prediction_and_audio_model():
    st.title("ทำนายราคาหุ้นและวิเคราะห์เสียง")

    # ส่วนของการทำนายราคาหุ้น
    st.subheader("ทำนายราคาหุ้นจากข้อมูลที่ผู้ใช้กรอก")
    high_price = st.number_input("High Price", min_value=0.0)
    low_price = st.number_input("Low Price", min_value=0.0)
    volume = st.number_input("Volume", min_value=0)
    ma_50 = st.number_input("Moving Average (50 days)", min_value=0.0)
    ma_200 = st.number_input("Moving Average (200 days)", min_value=0.0)
    change = st.number_input("Change", min_value=0.0)
    perc_change = st.number_input("Percentage Change", min_value=0.0)

    input_data = pd.DataFrame([[high_price, low_price, volume, ma_50, ma_200, change, perc_change]], 
                              columns=['High Price', 'Low Price', 'Volume', 'Moving Average (50 days)', 'Moving Average (200 days)', 'Change', 'Percentage Change'])

    imputer = SimpleImputer(strategy='mean')
    input_data = pd.DataFrame(imputer.fit_transform(input_data), columns=input_data.columns)

    scaler = joblib.load(Path(__file__).parent.parent / "ML" / "scaler.pkl")
    input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    model_1 = joblib.load(Path(__file__).parent.parent / "ML" / "decision_tree_model.pkl")
    model_2 = joblib.load(Path(__file__).parent.parent / "ML" / "knn_model.pkl")
    model_3 = joblib.load(Path(__file__).parent.parent / "ML" / "svr_model.pkl")

    model_choice = st.selectbox("เลือกโมเดลที่ใช้ในการทำนาย", ["Decision Tree", "KNN", "SVR", "Ensemble"])

    if model_choice == "Decision Tree":
        pred = model_1.predict(input_data_scaled)
    elif model_choice == "KNN":
        pred = model_2.predict(input_data_scaled)
    elif model_choice == "SVR":
        pred = model_3.predict(input_data_scaled)
    elif model_choice == "Ensemble":
        pred_dt = model_1.predict(input_data_scaled)
        pred_knn = model_2.predict(input_data_scaled)
        pred_svr = model_3.predict(input_data_scaled)
        pred = (pred_dt + pred_knn + pred_svr) / 3

    st.write(f"ราคาหุ้นที่ทำนาย: {pred[0]:.2f}")

    # ส่วนของการวิเคราะห์เสียง
    st.subheader("วิเคราะห์เสียงจากไฟล์ที่เลือก")
    audio_option = st.radio(
        "เลือกประเภทเสียง:",
        ["Speech", "Music", "เลือกไฟล์ของคุณเอง", "ลิ้งค์ YouTube"]
    )

    audio_path = None
    if audio_option == "Speech":
        audio_path = "data/Speech.wav"
    elif audio_option == "Music":
        audio_path = "data/COCKTAIL.wav"
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

    if audio_path:
        mel_spec = extract_features(audio_path)
        if mel_spec is None:
            return

        max_len = 1320
        if mel_spec.shape[1] < max_len:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])))
        else:
            mel_spec = mel_spec[:, :max_len]

        mel_spec = mel_spec[..., np.newaxis]
        if mel_spec.shape != (128, 1320, 1):
            st.error(f"ขนาดของข้อมูลที่ป้อนเข้าโมเดลไม่ถูกต้อง: {mel_spec.shape}")
            return

        model = load_model(Path(__file__).parent.parent / "NL" / "model.h5", compile=False)
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

    if "temp_file" in locals() and os.path.exists(temp_file):
        os.unlink(temp_file)
    if "audio_path" in locals() and audio_path.startswith("/tmp") and os.path.exists(audio_path):
        os.unlink(audio_path)
