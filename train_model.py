import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib


# 1. Memuat Dataset
# Gantilah 'path_to_your_dataset.csv' dengan path file dataset Anda
data = pd.read_csv('dataset.csv')
data
# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


# 2. Menentukan Fitur dan Label
# Misalkan kolom terakhir adalah label dan kolom lainnya adalah fitur
X = data.iloc[:, :-1].values  # Semua kolom kecuali kolom terakhir
y = data.iloc[:, -1].values    # Kolom terakhir

# 3. Membagi Data menjadi Data Train dan Data Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inisialisasi Model Naive Bayes
base_estimator = GaussianNB()

# 5. Inisialisasi Model AdaBoost dengan Naive Bayes sebagai weak learner
adaboost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)


# 6. Melatih Model
adaboost.fit(X_train, y_train)

# 7. Melakukan Prediksi
y_pred = adaboost.predict(X_test)

# 8. Evaluasi Model

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model, encoders, and feature columns
joblib.dump(adaboost, 'adaboost_model.pkl')  # Simpan model
joblib.dump(label_encoders, 'label_encoders.pkl')  # Simpan label encoder
features_columns = data.columns[:-1].tolist()  # Semua kolom kecuali kolom target
joblib.dump(features_columns, 'features_columns.pkl')  # Simpan urutan kolom fitur

print("Model and encoders have been saved successfully!")
