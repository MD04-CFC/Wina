import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def f1(X, y, X_wlasne, y_wlasne, prog, model, czy_norm):
    # Convert to numpy arrays
    X = X.to_numpy(dtype=float)
    X_wlasne = X_wlasne.to_numpy(dtype=float)

    # ✅ Binarize both y and y_wlasne using the threshold (e.g. >= 5)
    y = np.array([int(val >= prog) for val in y])
    y_wlasne = np.array([int(val >= prog) for val in y_wlasne])

    # ✅ Normalize features (NOT labels)
    if czy_norm:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1e-8  # prevent division by zero
        X = (X - mean) / std
        X_wlasne = (X_wlasne - mean) / std

    # Split just for training (you test on your own data)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_wlasne)

    # ✅ Now safe to use binary metrics
    print(f"\n--- Drzewo decyzyjne Results | Threshold: {prog} | Normalized: {czy_norm} ---")
    print(f"Accuracy:  {accuracy_score(y_wlasne, y_pred):.3f}")
    print(f"Precision: {precision_score(y_wlasne, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_wlasne, y_pred, zero_division=0):.3f}")
    print(f"F1 Score:  {f1_score(y_wlasne, y_pred, zero_division=0):.3f}")

# === Data Loading ===

# Your own observations (test)
X_wlasne = pd.read_csv('A:\\disk_S\\homework_sggw\\projekt_mad2\\Wina\\dane\\artificial_wine_observations.csv', delimiter=',')
y_wlasne = X_wlasne['quality']
X_wlasne = X_wlasne.drop(columns=["quality"])

# Training dataset
X = pd.read_csv('A:\\disk_S\\homework_sggw\\projekt_mad2\\Wina\\dane\\dataset_clean.csv', delimiter=';')
y = X['quality']
X = X.drop(columns=["quality"])

# Model
model0 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42)
model1 = RandomForestClassifier(n_estimators=100, random_state=42)

model2 = Perceptron(tol=1e-3, random_state=0)
model3 = KNeighborsClassifier(n_neighbors=3)
model4 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)

# Run
f1(X, y, X_wlasne, y_wlasne, prog=5, model=model4, czy_norm=False)
