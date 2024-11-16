from flask import Flask, request, jsonify
import numpy as np
import joblib



# Initialize Flask app
app = Flask(__name__)

# Load the trained model and other required objects
try:
    trained_model = joblib.load('adaboost_model.pkl')  # Load the trained model
    label_encoders = joblib.load('label_encoders.pkl')  # Load encoders
    features_columns = joblib.load('features_columns.pkl')  # Load feature column order
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    exit(1)  # Exit if files are missing

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form-urlencoded
        data = request.form.to_dict()

        # Pastikan data diformat sebagai array sesuai urutan kolom
        features = np.array([data[col] for col in features_columns]).reshape(1, -1)

        # Prediksi
        prediction = trained_model.predict(features)[0]
        prediction_class = label_encoders['class'].inverse_transform([prediction])[0]

        return jsonify({"prediction": prediction_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "<h1>Welcome to the Diabetes Prediction API</h1><p>Use the /predict endpoint to get predictions.</p>"


if __name__ == '__main__':
    app.run(debug=True)





