from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 


def wspolczynniki(prog):
    X = wine_quality.data.features 
    y= wine_quality.data.targets 
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    logr = linear_model.LogisticRegression()
    logr.fit(X_train, y_train)                         # dla jednej cechy
    y1_pred = logr.predict(X_train)
    y2_pred = logr.predict(X_test)

    print("bez normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Współczynniki dla progu {prog}:")
    print("Współczynniki:", logr.coef_)
    print("Wyraz wolny:", logr.intercept_)
    print()

    print("Dokładność na treningowych:", logr.score(X_train, y_train))   
    print("Dokładność na testowych:", logr.score(X_test, y_test))   
    
    print()
    print("Precyzja na treningowych:", precision_score(y1_pred, y_train))
    print("Precyzja:", precision_score(y2_pred, y_test))

    print()
    print("Czułość na treningowych:", recall_score(y1_pred, y_train))
    print("Czułość:", recall_score(y2_pred, y_test))

    print()
    print("F1 na treningowych:", f1_score(y1_pred, y_train))
    print("F1:", f1_score(y2_pred, y_test))




def rysuj(a, prog):
    X = wine_quality.data.features 
    y= wine_quality.data.targets 
    x = X[a].values
    y = y.values
    y = np.array([int(x >= prog) for x in y])


    sns.regplot( x=x, y=y, logistic=True, ci=None, scatter_kws={'color': 'black'}, line_kws={'color': 'red'} )
    plt.xlabel(a)
    plt.ylabel(f"Quality >= {prog}")
    plt.title("Logistic Regression on Wine Quality")
    plt.show()



def wspolczynniki_dlawybranej(a, prog):                      
    X = wine_quality.data.features 
    y= wine_quality.data.targets 
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train2 = X_train[[a]].values
    X_test2 = X_test[[a]].values

    logr = linear_model.LogisticRegression()
    logr.fit(X_train2, y_train)                         # dla jednej cechy
    y1_pred = logr.predict(X_train2)
    y2_pred = logr.predict(X_test2)

    print("bez normalizacji!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Współczynniki dla {a} i progu {prog}:")
    print("Współczynniki:", logr.coef_)
    print("Wyraz wolny:", logr.intercept_)
    print()

    print("Dokładność na treningowych:", logr.score(X_train2, y_train))   
    print("Dokładność na testowych:", logr.score(X_test2, y_test))   
    
    print()
    print("Precyzja na treningowych:", precision_score(y1_pred, y_train))
    print("Precyzja:", precision_score(y2_pred, y_test))

    print()
    print("Czułość na treningowych:", recall_score(y1_pred, y_train))
    print("Czułość:", recall_score(y2_pred, y_test))

    print()
    print("F1 na treningowych:", f1_score(y1_pred, y_train))
    print("F1:", f1_score(y2_pred, y_test))



def rysuj(a, prog):
    X = wine_quality.data.features 
    y= wine_quality.data.targets 
    x = X[a].values
    y = y.values
    y = np.array([int(x >= prog) for x in y])


    sns.regplot( x=x, y=y, logistic=True, ci=None, scatter_kws={'color': 'black'}, line_kws={'color': 'red'} )
    plt.xlabel(a)
    plt.ylabel(f"Quality >= {prog}")
    plt.title("Logistic Regression on Wine Quality")
    plt.show()


    





def wspolczynniki_skala(a, prog):                      
    X = wine_quality.data.features 
    y = wine_quality.data.targets 
    y = y.values
    y = np.array([int(x >= prog) for x in y])
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalizacja cech
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train2 = X_train[[a]].values
    X_test2 = X_test[[a]].values

    logr = linear_model.LogisticRegression()
    logr.fit(X_train2, y_train)                         # dla jednej cechy
    y1_pred = logr.predict(X_train2)
    y2_pred = logr.predict(X_test2)

    print("z normalizacją!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print(f"Współczynniki dla {a} i progu {prog}:")
    print("Współczynniki:", logr.coef_)
    print("Wyraz wolny:", logr.intercept_)
    print()

    print("Dokładność na treningowych:", logr.score(X_train2, y_train))   
    print("Dokładność na testowych:", logr.score(X_test2, y_test))   
    
    print()
    print("Precyzja na treningowych:", precision_score(y1_pred, y_train))
    print("Precyzja:", precision_score(y2_pred, y_test))

    print()
    print("Czułość na treningowych:", recall_score(y1_pred, y_train))
    print("Czułość:", recall_score(y2_pred, y_test))

    print()
    print("F1 na treningowych:", f1_score(y1_pred, y_train))
    print("F1:", f1_score(y2_pred, y_test))
   


def rysuj_skala(a, prog):
    X = wine_quality.data.features 
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalizacja cech
    y= wine_quality.data.targets 
    x = X[a].values
    y = y.values
    y = np.array([int(x >= prog) for x in y])


    sns.regplot( x=x, y=y, logistic=True, ci=None, scatter_kws={'color': 'black'}, line_kws={'color': 'red'} )
    plt.xlabel(a)
    plt.ylabel(f"Quality >= {prog}")
    plt.title("Logistic Regression on Wine Quality")
    plt.show()





wspolczynniki(5)
wspolczynniki(6)
wspolczynniki(7)













'''
wspolczynniki_skala('alcohol',7)
#rysuj_skala('alcohol',7)
wspolczynniki('alcohol',7)


wspolczynniki_skala('sulphates',6)
wspolczynniki('sulphates',6)    


wspolczynniki_skala('sulphates',7)
#rysuj_skala('sulphates',7)
wspolczynniki('sulphates',7)


wspolczynniki_skala('pH',7)
#rysuj_skala('pH',7)
wspolczynniki('pH',7)


a = 'density'
prog = 6


wspolczynniki_skala(a, prog)
rysuj_skala(a, prog)
wspolczynniki(a, prog)
rysuj(a, prog)
'''

'''
rysuj('alcohol',6)
rysuj("fixed_acidity",6)
rysuj("volatile_acidity", 6)
rysuj("citric_acid",5)
rysuj("residual_sugar",6)
rysuj("chlorides",4)
rysuj("free_sulfur_dioxide",7)
rysuj("total_sulfur_dioxide",8)
rysuj("density",6)



wspolczynniki('alcohol',7)
rysuj('alcohol',7)

'''