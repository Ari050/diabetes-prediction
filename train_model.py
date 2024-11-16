import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

# Load dataset
dataset = pd.read_csv('dataset.csv')  # Pastikan dataset Anda ada di folder yang sama

# Encode categorical features
label_encoders = {}
for col in dataset.select_dtypes(include='object').columns:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Separate features (X) and target (y)
X = dataset.drop(columns=['class'])
y = dataset['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize Naive Bayes model and Adaboost classifier
nb = GaussianNB()
adaboost = AdaBoostClassifier(base_estimator=nb, n_estimators=50, random_state=42)

# Train the model
adaboost.fit(X_train, y_train)

# Save the trained model, encoders, and feature columns
joblib.dump(adaboost, 'adaboost_model.pkl')  # Simpan model
joblib.dump(label_encoders, 'label_encoders.pkl')  # Simpan label encoder
joblib.dump(X.columns.tolist(), 'features_columns.pkl')  # Simpan urutan kolom fitur

print("Model and encoders have been saved successfully!")
