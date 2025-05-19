import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Optional: Suppress warnings
#warnings.filterwarnings('ignore')

# === Configurable threshold for classification ===
QUALITY_THRESHOLD = 6.0  # everything >= 6.0 is "good", else "bad"

# === Load the dataset ===
url = "/media/oleksandr/Main/disk_S/homework_sggw/projekt_mad2/Wina/dataset_clean.csv"
wine_data = pd.read_csv(url, delimiter=";")

# === Feature selection ===
X = wine_data[[
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]]

# === Binary target transformation ===
# 1 = Good (quality >= threshold), 0 = Bad
y = (wine_data['quality'] >= QUALITY_THRESHOLD).astype(int)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train classifier ===
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# === Predictions and evaluation ===
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# === Output ===
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)