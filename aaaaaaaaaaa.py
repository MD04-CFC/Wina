import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model

def f1(X, y, X_wlasne, prog, model):
    # Convert to numpy arrays
    X = X.to_numpy(dtype=float)
    X_wlasne = X_wlasne.to_numpy(dtype=float)

    y = np.array([int(val >= prog) for val in y])

    model.fit(X, y)
    y_pred = model.predict(X_wlasne)

    print(y_pred)

# === Data Loading ===

# Your own observations (test)
X_wlasne = pd.read_csv('A:\\disk_S\\homework_sggw\\projekt_mad2\\Wina\\dane\\artificial_wine_observations.csv', delimiter=',')
X_wlasne = X_wlasne.drop(columns=["quality"])

# Training dataset
X = pd.read_csv('A:\\disk_S\\homework_sggw\\projekt_mad2\\Wina\\dane\\dataset_clean.csv', delimiter=';')
y = X['quality']
X = X.drop(columns=["quality"])

# Model
model0 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42)
model1 = RandomForestClassifier(n_estimators=100, random_state=42)

model2 = Perceptron(tol=1e-3, random_state=0)
model3 = KNeighborsClassifier(n_neighbors=10)
model4 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)

model5 = linear_model.LogisticRegression()

models = [model0, model1, model2, model3, model4, model5]

from sklearn.preprocessing import StandardScaler

# Scale both training and test data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_wlasne_scaled = scaler.transform(X_wlasne)

# Replace your for-loop with this
for m in models:
    print(f"--- {m.__class__.__name__} ---")
    f1(pd.DataFrame(X_scaled), y, pd.DataFrame(X_wlasne_scaled), prog=6, model=m)

