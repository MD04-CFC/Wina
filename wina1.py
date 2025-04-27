from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  


def rysuj(a, prog):
    X = wine_quality.data.features 
    y= wine_quality.data.targets 
    x = X[a].values
    y = y.values
    y = np.array([int(x >= prog) for x in y])

    #logr = linear_model.LogisticRegression()
    #logr.fit(X,y)

    sns.regplot( x=x, y=y, logistic=True, ci=None, scatter_kws={'color': 'black'}, line_kws={'color': 'red'} )
    plt.xlabel(a)
    plt.ylabel(f"Quality >= {prog}")
    plt.title("Logistic Regression on Wine Quality")
    plt.show()


rysuj('alcohol',6)
rysuj("fixed_acidity",6)
rysuj("volatile_acidity", 6)
rysuj("citric_acid",5)
rysuj("residual_sugar",6)
rysuj("chlorides",4)
rysuj("free_sulfur_dioxide",7)
rysuj("total_sulfur_dioxide",8)
rysuj("density",6)
