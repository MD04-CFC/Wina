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

# fetch dataset 
#wine_quality = fetch_ucirepo(id=186) 
#features = wine_quality.data.features



from sklearn.metrics import accuracy_score

def f1(X, y, prog, model, czy_norm):
    X = X.to_numpy(dtype=float)
    y = np.array([int(val >= prog) for val in y])
    # Optional normalization
    if czy_norm:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Utwórz i wytrenuj model XGBoost
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\n--- Random Forest Results for Threshold: {prog} | Normalized: {czy_norm} ---")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print()
    print(f"Train Precision: {precision_score(y_train, y_train_pred, zero_division=0):.3f}")
    print(f"Test Precision: {precision_score(y_test, y_test_pred, zero_division=0):.3f}")
    print()
    print(f"Train Recall: {recall_score(y_train, y_train_pred, zero_division=0):.3f}")
    print(f"Test Recall: {recall_score(y_test, y_test_pred, zero_division=0):.3f}")
    print()
    print(f"Train F1 Score: {f1_score(y_train, y_train_pred, zero_division=0):.3f}")
    print(f"Test F1 Score: {f1_score(y_test, y_test_pred, zero_division=0):.3f}")


    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["słabe", "dobre"], yticklabels=["słabe", "dobre"])
    plt.title("Macierz pomyłek (0 = słabe, 1 = dobre)")
    plt.xlabel("Przewidywane")
    plt.ylabel("Rzeczywiste")
    plt.tight_layout()
    plt.show()


X = pd.read_csv('/media/oleksandr/Main/disk_S/homework_sggw/projekt_mad2/Wina/dataset_clean.csv', delimiter=';')

y = X['quality']

X = X.drop(columns=["quality"])


from sklearn.ensemble import RandomForestClassifier


#model1 = RandomForestClassifier(n_estimators=100, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)

f1(X,y,5,model,True)
f1(X,y,6,model,True)
f1(X,y,7,model,True)

print()

f1(X,y,5,model,False)
f1(X,y,6,model,False)
f1(X,y,7,model,False)