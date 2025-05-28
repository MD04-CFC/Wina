from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,accuracy_score
from sklearn.metrics import recall_score,confusion_matrix
from sklearn.model_selection import train_test_split

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  


def cal(prog):                                                                        
    X = wine_quality.data.features 
    y = wine_quality.data.targets
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 


    logr = linear_model.LogisticRegression()
    logr.fit(X_train, y_train)
    probs_logreg = logr.predict_proba(X_test)


    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    probs_rf = model.predict_proba(X_test)


    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    probs_knn = model.predict_proba(X_test)


    hybrid_probs = 0.4 * probs_logreg + 0.4 * probs_rf + 0.2 * probs_knn
    hybrid_preds = np.argmax(hybrid_probs, axis=1)                          



    print("Dla progu", prog, ":")
    print("bez normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
    print("Dokładność na testowych:", accuracy_score(hybrid_preds, y_test))   
    print()
    print("Precyzja:", precision_score(hybrid_preds, y_test))
    print()
    print("Czułość:", recall_score(hybrid_preds, y_test))
    print()
    f1 = f1_score(y_test, hybrid_preds)
    print(f"F1 Score: {f1:.2f}")
  
    scores = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring='f1')
    print(f"Crosswalidacja dla lasu losowego: {np.mean(scores):.2f}")





def cal_skal(prog):                                                                        
    X = wine_quality.data.features 
    y = wine_quality.data.targets
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    


    logr = linear_model.LogisticRegression()
    logr.fit(X_train, y_train)
    probs_logreg = logr.predict_proba(X_test)


    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    probs_rf = model.predict_proba(X_test)


    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    probs_knn = model.predict_proba(X_test)


    hybrid_probs = 0.4 * probs_logreg + 0.4 * probs_rf + 0.2 * probs_knn
    hybrid_preds = np.argmax(hybrid_probs, axis=1)                          



    print("Dla progu", prog, ":")
    print("dla normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
    print("Dokładność na testowych:", accuracy_score(hybrid_preds, y_test))   
    print()
    print("Precyzja:", precision_score(hybrid_preds, y_test))
    print()
    print("Czułość:", recall_score(hybrid_preds, y_test))
    print()
    f1 = f1_score(y_test, hybrid_preds)
    print(f"F1 Score: {f1:.2f}")
  
    scores = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring='f1')
    print(f"Crosswalidacja dla lasu losowego: {np.mean(scores):.2f}")
    


cal_skal(5)
cal_skal(6)
cal_skal(7)
