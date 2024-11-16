import streamlit as st
import numpy as np
import joblib

# Load the trained model and other components
trained_model = joblib.load('adaboost_model.pkl')  # Pastikan file model ada di direktori
label_encoders = joblib.load('label_encoders.pkl')
features_columns = joblib.load('features_columns.pkl')

# Define the Streamlit app
st.title("Aplikasi Prediksi Diabetes")
st.write("Masukkan informasi berikut untuk memprediksi risiko diabetes:")

# Input fields
data = {}
for feature in features_columns:
    if feature == 'Age':
        data[feature] = st.number_input(f"{feature} (Usia):", min_value=1, max_value=120, value=25)
    else:
        data[feature] = st.radio(f"{feature} (0=Tidak, 1=Ya):", options=[0, 1])

# Predict button
if st.button("Prediksi"):
    try:
        # Prepare input data
        input_data = np.array([data[feature] for feature in features_columns]).reshape(1, -1)

        # Make prediction
        prediction = trained_model.predict(input_data)[0]
        prediction_class = label_encoders['class'].inverse_transform([prediction])[0]

        # Display result
        st.success(f"Hasil Prediksi: {prediction_class}")
    except Exception as e:
        st.error(f"Error: {e}")
