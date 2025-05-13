from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import Perceptron
from peceptron_klasa import Perceptron as Perceptron_klasa
import pandas as pd


# fetch dataset 
#wine_quality = fetch_ucirepo(id=186) 
#features = wine_quality.data.features



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



def wyniki_mojaklasa(X,y,prog):
    X = X.to_numpy(dtype=float)
    #X = X.astype(float)                              
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = Perceptron_klasa(eta=0.004, epochs=1000, is_verbose=False)
    clf.fit(X_train, y_train)
    y1_pred = clf.predict(X_train)
    y2_pred = clf.predict(X_test)



    print('mojaklasa')
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





def wyniki_skala_mojaklasa(X,y,prog):
    X = X.to_numpy(dtype=float)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)                            
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = Perceptron_klasa(eta=0.004, epochs=1000, is_verbose=False)
    clf.fit(X_train, y_train)
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






X = pd.read_csv('dane\dataset_clean.csv', sep=';')
X.columns = X.columns.str.replace('"', '').str.strip()
y = X['quality']
X = X.drop(columns=["quality"])


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







def f1(X,y,prog, clf,czy_norm):
    X = X.to_numpy(dtype=float)
    if czy_norm == true:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)                            
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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




model1 = Perceptron(tol=1e-3, random_state=0)
model2 = Perceptron_klasa(eta=0.004, epochs=1000, is_verbose=False)
f1(X,y,6,model1,true)

















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
