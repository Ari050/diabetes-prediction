import streamlit as st
import numpy as np
import joblib

# Load the trained model and other components
trained_model = joblib.load('adaboost_model.pkl')  # Pastikan file model ada di direktori
label_encoders = joblib.load('label_encoders.pkl')
features_columns = joblib.load('features_columns.pkl')

# Define the Streamlit app
st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺", layout="centered")

# Sidebar
with st.sidebar:
    st.image("diabetes.png", caption="Pentingnya Deteksi Dini Diabetes", use_column_width=True)
    
    # Informasi Diabetes dengan Tautan
    st.write("### Informasi Terkait Diabetes")
    st.markdown("""
    - **Pola Hidup Sehat:** [Rutin olahraga dan makan sehat](https://www.who.int/news-room/fact-sheets/detail/diabetes).
    - **Obat Diabetes:** [Panduan obat diabetes](https://www.diabetes.org/diabetes/treatment-care).
    - **Cek Gula Darah Rutin:** [Cara memantau gula darah](https://www.cdc.gov/diabetes/managing/monitoring.html).
    - **Edukasi Diabetes:** [Pahami faktor risiko diabetes](https://www.diabetes.org/diabetes/risk-test).
    """)


    st.write("### Instruksi Penggunaan")
    st.markdown("""
    1. Isi semua kolom dengan informasi yang sesuai.
    2. Klik tombol **Prediksi** untuk melihat hasilnya.
    """)

# Header
st.title("🩺 Aplikasi Prediksi Diabetes")
st.markdown("**Deteksi dini risiko diabetes dengan algoritma machine learning.**")
st.write("---")

# Input fields
st.subheader("Masukkan Data:")
data = {}
data["Age"] = st.slider("Berapa usia Anda?", min_value=1, max_value=120, value=25)
data["Gender"] = st.radio("Apa jenis kelamin Anda?", options=["Pria", "Wanita"])
data["Gender"] = 0 if data["Gender"] == "Pria" else 1

data["Polyuria"] = st.radio("Apakah Anda sering buang air kecil secara berlebihan?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["Polydipsia"] = st.radio("Apakah Anda merasa haus secara berlebihan?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["sudden weight loss"] = st.radio("Apakah Anda mengalami penurunan berat badan secara tiba-tiba?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["weakness"] = st.radio("Apakah Anda merasa lemah atau lesu?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["Polyphagia"] = st.radio("Apakah Anda sering merasa lapar secara berlebihan?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["Genital thrush"] = st.radio("Apakah Anda pernah mengalami infeksi genital?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["visual blurring"] = st.radio("Apakah Anda mengalami penglihatan kabur?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["Itching"] = st.radio("Apakah Anda sering merasa gatal?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["Irritability"] = st.radio("Apakah Anda sering merasa mudah marah?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["delayed healing"] = st.radio("Apakah luka Anda memerlukan waktu lama untuk sembuh?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["partial paresis"] = st.radio("Apakah Anda mengalami kesulitan menggerakkan sebagian tubuh?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["muscle stiffness"] = st.radio("Apakah Anda sering mengalami kekakuan otot?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["Alopecia"] = st.radio("Apakah Anda mengalami kebotakan?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
data["Obesity"] = st.radio("Apakah Anda mengalami obesitas?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

st.write("---")
# Predict button
if st.button("Prediksi Risiko Diabetes"):
    try:
        # Prepare input data
        input_data = np.array([data[feature] for feature in features_columns]).reshape(1, -1)

        # Make prediction
        prediction = trained_model.predict(input_data)[0]
        prediction_class = label_encoders['class'].inverse_transform([prediction])[0]

        # Display result
        st.success(f"Hasil Prediksi: **{prediction_class}**")
        if prediction_class == "Positive":
            st.warning("Risiko diabetes Anda tinggi. Harap konsultasi dengan dokter.")
        else:
            st.info("Risiko diabetes Anda rendah. Tetap jaga pola hidup sehat!")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Footer
st.write("---")
st.markdown("💡 **Tips:** Untuk hasil terbaik, pastikan data yang Anda masukkan akurat.")
st.markdown("🔗 **Dikembangkan oleh [Ashari Wahyudi](https://www.instagram.com/ashariwahyudi05/profilecard/?igsh=MW45MG4zZGZ5NXc5MQ== ).**")
