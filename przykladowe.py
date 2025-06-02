#from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,confusion_matrix
from sklearn.linear_model import Perceptron
#from peceptron_klasa import Perceptron as Perceptron_klasa
import pandas as pd
import xgboost as xgb


from sklearn.metrics import accuracy_score

def f1(X, y, X_wlasne, y_wlasne, prog, model, czy_norm):
    X = X.to_numpy(dtype=float)
    y = np.array([int(val >= prog) for val in y])
    # Optional normalization
    if czy_norm:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Utwórz i wytrenuj model XGBoost
    
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_wlasne)

    print(f"\n--- XGboost prog: {prog} | Normalized: {czy_norm} ---")
    print(f"Test Accuracy: {accuracy_score(y_wlasne, y_test_pred):.3f}")
    print()
    print(f"Test Precision: {precision_score(y_wlasne, y_test_pred, zero_division=0):.3f}")
    print()
    print(f"Test Recall: {recall_score(y_wlasne, y_test_pred, zero_division=0):.3f}")
    print()
    print(f"Test F1 Score: {f1_score(y_wlasne, y_test_pred, zero_division=0):.3f}")

X_wlasne = pd.read_csv('A:\\disk_S\\homework_sggw\\projekt_mad2\\Wina\dane\\artificial_wine_observations.csv', delimiter=',')
y_wlasne = X_wlasne['quality']
X_wlasne = X_wlasne.drop(columns=["quality"])

X = pd.read_csv('A:\\disk_S\\homework_sggw\\projekt_mad2\\Wina\dane\\dataset_clean.csv', delimiter=';')
y = X['quality']
X = X.drop(columns=["quality"])

from sklearn.ensemble import RandomForestClassifier


#model1 = RandomForestClassifier(n_estimators=100, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)

f1(X,y,X_wlasne, y_wlasne,5,model,False)