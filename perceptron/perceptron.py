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

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 



def wyniki(prog):                      
    X = wine_quality.data.features 
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








def wyniki_mojaklasa(prog):                                            
    X = wine_quality.data.features.to_numpy(dtype=float) 
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
wyniki(5)
print()
wyniki(6)
print()
wyniki(7)
'''


'''
wyniki_skala(5)
print()
wyniki_skala(6)
print()
wyniki_skala(7)


'''
'''
wyniki_mojaklasa(5)
print()
wyniki_mojaklasa(6)
print()
wyniki_mojaklasa(7)

'''


wyniki_skala_mojaklasa(5)
print()
wyniki_skala_mojaklasa(6)
print()
wyniki_skala_mojaklasa(7)
