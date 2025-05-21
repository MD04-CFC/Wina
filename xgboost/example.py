import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import xgboost as xgb

# Wczytaj dane
url = "/media/oleksandr/Main/disk_S/homework_sggw/projekt_mad2/Wina/dataset_clean.csv"
wine_data = pd.read_csv(url, delimiter=";")

# Wybrane cechy
X = wine_data[["fixed acidity","volatile acidity","citric acid","residual sugar",
               "chlorides","free sulfur dioxide","total sulfur dioxide",
               "density","pH","sulphates","alcohol"]]

# Binarna klasyfikacja: jakość >= 6 to dobre wino
y = (wine_data['quality'] >= 7).astype(int)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utwórz i wytrenuj model XGBoost
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Predykcja
y_pred = model.predict(X_test)

# Metryki
print("=== Metryki modelu XGBoost ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1-score: {f1_score(y_test, y_pred):.2f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
