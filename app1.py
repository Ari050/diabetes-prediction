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
    st.image("https://via.placeholder.com/300x200.png?text=Diabetes+Prediction", caption="Prediksi Diabetes", use_column_width=True)
    
    st.write("### Informasi Terkait Diabetes")
    st.markdown("""
    - **Pola Hidup Sehat:** Rutin olahraga dan makan makanan sehat dapat membantu mengelola diabetes.
    - **Obat Diabetes:** Konsultasikan dengan dokter untuk memilih obat yang tepat.
    - **Cek Gula Darah Rutin:** Pantau kadar gula darah secara teratur untuk mencegah komplikasi.
    - **Edukasi:** Pahami faktor risiko diabetes seperti obesitas, pola makan, dan gaya hidup.
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
for feature in features_columns:
    if feature == 'Age':
        data[feature] = st.slider(f"{feature} (Usia):", min_value=1, max_value=120, value=25)
    else:
        data[feature] = st.radio(f"{feature} (0=Tidak, 1=Ya):", options=[0, 1], horizontal=True)

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
st.markdown("🔗 **Dikembangkan oleh [Nama Anda](https://example.com).**")
