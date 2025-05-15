import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from io import StringIO

# === PARAMETRY ===
threshold = 6         # Próg: quality >= threshold → dobre (1), inaczej słabe (0)
binary_classification = True  # True = 0/1, False = klasy 3–8
num_samples = 500

# === Generowanie danych jakości wina od 3 do 8 ===
np.random.seed(42)
qualities = np.random.choice([3, 4, 5, 6, 7, 8], size=num_samples, p=[0.05, 0.1, 0.35, 0.3, 0.15, 0.05])
features = np.random.normal(loc=0, scale=1, size=(num_samples, 11))

columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol']

df = pd.DataFrame(features, columns=columns)
df['quality'] = qualities

# Konwersja do CSV string (z separatorem `;`)
csv_data = df.to_csv(index=False, sep=';')

# === Wczytanie danych ===
data = pd.read_csv(StringIO(csv_data), sep=';')

# === DODATKOWY WYKRES: rozkład klas quality w danych (diagram słupkowy) ===
plt.figure(figsize=(8, 5))
sns.countplot(x="quality", data=data, order=sorted(data["quality"].unique()), palette="crest")
plt.title("Rozkład jakości win (symulowany zbiór danych)")
plt.xlabel("Jakość (quality)")
plt.ylabel("Liczba próbek")
plt.tight_layout()
plt.show()

# === Przygotowanie danych do klasyfikacji ===
if binary_classification:
    data["quality_binary"] = (data["quality"] >= threshold).astype(int)
    X = data.drop(["quality", "quality_binary"], axis=1)
    y = data["quality_binary"]
else:
    X = data.drop("quality", axis=1)
    y = data["quality"]

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Accuracy i raport
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Dodanie accuracy na górę
accuracy_row = pd.DataFrame({"precision": [acc], "recall": [None], "f1-score": [None], "support": [None]}, index=["accuracy"])
report_df = pd.concat([accuracy_row, report_df])

print("=== Raport klasyfikacji ===")
print(report_df)

# === Wizualizacja wyników klasyfikacji ===
if binary_classification:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["słabe", "dobre"], yticklabels=["słabe", "dobre"])
    plt.title("Macierz pomyłek (0 = słabe, 1 = dobre)")
    plt.xlabel("Przewidywane")
    plt.ylabel("Rzeczywiste")
    plt.tight_layout()
    plt.show()
else:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=y_pred, order=sorted(data["quality"].unique()))
    plt.title("Rozkład przewidzianych klas quality (multi-class)")
    plt.xlabel("Przewidziana jakość")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.show()
