import pandas
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets  

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree_params = {
    'max_depth': 3,
    'min_samples_leaf': 10,
    'random_state': 42
}

def cla(threshold, X_train, X_test, y_train, y_test):

    y_train_bin = (y_train >= threshold).astype(int)
    y_test_bin = (y_test >= threshold).astype(int)

    clf = DecisionTreeClassifier(**tree_params)
    clf.fit(X_train, y_train_bin)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test_bin, y_pred)

    # Wizualizacja drzewa
    plt.figure(figsize=(16,8))
    plot_tree(
        clf, 
        feature_names=X.columns, 
        class_names=['Niedobre', 'Dobre'], 
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    plt.title('Drzewo decyzyjne - bez normalizacji')
    plt.show()

    return accuracy


def cla_norm(threshold, X_train, X_test, y_train, y_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_bin = (y_train >= threshold).astype(int)
    y_test_bin = (y_test >= threshold).astype(int)

    clf = DecisionTreeClassifier(**tree_params)
    clf.fit(X_train_scaled, y_train_bin)

    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_bin, y_pred)

    # Wizualizacja drzewa
    plt.figure(figsize=(16,8))
    plot_tree(
        clf, 
        feature_names=X.columns, 
        class_names=['Niedobre', 'Dobre'], 
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    plt.title('Drzewo decyzyjne - z normalizacji')
    plt.show()

    return accuracy

print(cla(7, X_train, X_test, y_train, y_test))
print(cla_norm(7, X_train, X_test, y_train, y_test))
