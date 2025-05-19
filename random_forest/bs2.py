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


# fetch dataset 
#wine_quality = fetch_ucirepo(id=186) 
#features = wine_quality.data.features



from sklearn.metrics import accuracy_score, classification_report

def f1(X, y, prog, clf, czy_norm):
    X = X.to_numpy(dtype=float)
    
    # Optional normalization
    if czy_norm:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Convert quality to binary classification based on threshold
    y = np.array([int(val >= prog) for val in y])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit on training data only
    clf.fit(X_train, y_train)

    # Predict
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

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


    # cm = confusion_matrix(y_test, y2_pred)
    # plt.figure(figsize=(5, 4))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["słabe", "dobre"], yticklabels=["słabe", "dobre"])
    # plt.title("Macierz pomyłek (0 = słabe, 1 = dobre)")
    # plt.xlabel("Przewidywane")
    # plt.ylabel("Rzeczywiste")
    # plt.tight_layout()
    # plt.show()

    





def wyniki(X, y, prog):                    
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X, y)
    y1_pred = clf.predict(X_train)
    y2_pred = clf.predict(X_test)


    print("Dla progu", prog, ":")
    print("bez normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Dokładność na treningowych:", clf.score(X_train, y_train))   
    print("Dokładność na testowych:", clf.score(X_test, y_test))   
    print()
    print("Precyzja na treningowych:", precision_score(y1_pred, y_train))
    print("Precyzja:", precision_score(y2_pred, y_test))
    print()
    print("Czułość na treningowych:", recall_score(y1_pred, y_train))
    print("Czułość:", recall_score(y2_pred, y_test))
    print()
    print("F1 na treningowych:", f1_score(y1_pred, y_train))
    print("F1:", f1_score(y2_pred, y_test))



# def wyniki_mojaklasa(X,y,prog):
#     X = X.to_numpy(dtype=float)
#     #X = X.astype(float)                              
#     y = y.values
#     y = np.array([int(x >= prog) for x in y])
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     clf = Perceptron_klasa(eta=0.004, epochs=1000, is_verbose=False)
#     clf.fit(X_train, y_train)
#     y1_pred = clf.predict(X_train)
#     y2_pred = clf.predict(X_test)



#     print('mojaklasa')
#     print("Dla progu", prog, ":")
#     print("bez normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     print("Dokładność na treningowych:", clf.score(X_train, y_train))   
#     print("Dokładność na testowych:", clf.score(X_test, y_test))   
#     print()
#     print("Precyzja na treningowych:", precision_score(y1_pred, y_train))
#     print("Precyzja:", precision_score(y2_pred, y_test))
#     print()
#     print("Czułość na treningowych:", recall_score(y1_pred, y_train))
#     print("Czułość:", recall_score(y2_pred, y_test))
#     print()
#     print("F1 na treningowych:", f1_score(y1_pred, y_train))
#     print("F1:", f1_score(y2_pred, y_test))








def wyniki_skala(X, y, prog):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalizacja cech                    
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X, y)
    y1_pred = clf.predict(X_train)
    y2_pred = clf.predict(X_test)


    print("Dla progu", prog, ":")
    print("dla normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Dokładność na treningowych:", clf.score(X_train, y_train))   
    print("Dokładność na testowych:", clf.score(X_test, y_test))   
    print()
    print("Precyzja na treningowych:", precision_score(y1_pred, y_train))
    print("Precyzja:", precision_score(y2_pred, y_test))
    print()
    print("Czułość na treningowych:", recall_score(y1_pred, y_train))
    print("Czułość:", recall_score(y2_pred, y_test))
    print()
    print("F1 na treningowych:", f1_score(y1_pred, y_train))
    print("F1:", f1_score(y2_pred, y_test))





# def wyniki_skala_mojaklasa(X,y,prog):
#     X = X.to_numpy(dtype=float)
#     X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)                            
#     y = y.values
#     y = np.array([int(x >= prog) for x in y])
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     clf = Perceptron_klasa(eta=0.004, epochs=1000, is_verbose=False)
#     clf.fit(X_train, y_train)
#     y1_pred = clf.predict(X_train)
#     y2_pred = clf.predict(X_test)


#     print('mojaklasa')
#     print("Dla progu", prog, ":")
#     print("dla normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     print("Dokładność na treningowych:", clf.score(X_train, y_train))   
#     print("Dokładność na testowych:", clf.score(X_test, y_test))   
#     print()
#     print("Precyzja na treningowych:", precision_score(y1_pred, y_train))
#     print("Precyzja:", precision_score(y2_pred, y_test))
#     print()
#     print("Czułość na treningowych:", recall_score(y1_pred, y_train))
#     print("Czułość:", recall_score(y2_pred, y_test))
#     print()
#     print("F1 na treningowych:", f1_score(y1_pred, y_train))
#     print("F1:", f1_score(y2_pred, y_test))

    






X = pd.read_csv('/media/oleksandr/Main/disk_S/homework_sggw/projekt_mad2/Wina/dataset_clean.csv', delimiter=';')
#X.columns = X.columns.str.replace('"', '').str.strip()
y = X['quality']
X = X.drop(columns=["quality"])

'''
wyniki(X, y, 5)
print()
wyniki(X, y, 6)
print()
wyniki(X, y,7)
print()



wyniki_skala(X, y,5)
print()
wyniki_skala(X, y,6)
print()
wyniki_skala(X, y,7)
print()


wyniki_mojaklasa(X, y,5)
print()
wyniki_mojaklasa(X, y,6)
print()
wyniki_mojaklasa(X, y,7)
print()

wyniki_skala_mojaklasa(X, y,5)
print()
wyniki_skala_mojaklasa(X, y,6)
print()
wyniki_skala_mojaklasa(X, y,7)


'''




from sklearn.ensemble import RandomForestClassifier


# model1 = Perceptron(tol=1e-3, random_state=0)
# model2 = Perceptron_klasa(eta=0.004, epochs=1000, is_verbose=False)
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
f1(X,y,5,model1,True)
f1(X,y,6,model1,True)
f1(X,y,7,model1,True)


# f1(X,y,5,model1,False)
# f1(X,y,6,model1,False)
# f1(X,y,7,model1,False)















'''

def wyniki_skala(prog):                      
    X = wine_quality.data.features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalizacja cech
    y = wine_quality.data.targets 
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_train2 = X_train[[a]].values
    #X_test2 = X_test[[a]].values

    
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X, y)
    
    y1_pred = clf.predict(X_train)
    y2_pred = clf.predict(X_test)


    print("Dla progu", prog, ":")
    print("dla normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    print("Dokładność na treningowych:", clf.score(X_train, y_train))   
    print("Dokładność na testowych:", clf.score(X_test, y_test))   
    
    print()
    print("Precyzja na treningowych:", precision_score(y1_pred, y_train))
    print("Precyzja:", precision_score(y2_pred, y_test))

    print()
    print("Czułość na treningowych:", recall_score(y1_pred, y_train))
    print("Czułość:", recall_score(y2_pred, y_test))

    print()
    print("F1 na treningowych:", f1_score(y1_pred, y_train))
    print("F1:", f1_score(y2_pred, y_test))









def wyniki_skala_mojaklasa(prog):                      
    X = wine_quality.data.features.to_numpy(dtype=float) 
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalizacja cech
    y = wine_quality.data.targets.values
    y = np.array([int(x >= prog) for x in y])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = Perceptron_klasa(eta=0.004, epochs=1000, is_verbose=False)
    clf.fit(X_train, y_train)
    
    y1_pred = clf.predict(X_train)
    y2_pred = clf.predict(X_test)


    clf = Perceptron_klasa(eta=0.004, epochs=1000, is_verbose=False)
    clf.fit(X, y)
    
    y1_pred = clf.predict(X_train)
    y2_pred = clf.predict(X_test)


    print('mojaklasa')
    print("Dla progu", prog, ":")
    print("dla normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    print("Dokładność na treningowych:", clf.score(X_train, y_train))   
    print("Dokładność na testowych:", clf.score(X_test, y_test))   
    
    print()
    print("Precyzja na treningowych:", precision_score(y1_pred, y_train))
    print("Precyzja:", precision_score(y2_pred, y_test))

    print()
    print("Czułość na treningowych:", recall_score(y1_pred, y_train))
    print("Czułość:", recall_score(y2_pred, y_test))

    print()
    print("F1 na treningowych:", f1_score(y1_pred, y_train))
    print("F1:", f1_score(y2_pred, y_test))
'''
