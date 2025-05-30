import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron
import pandas as pd



X = pd.read_csv('dane\dataset_clean.csv', sep=';')
X.columns = X.columns.str.replace('"', '').str.strip()
y = X['quality']
#X = X.drop(columns=["quality"])

a=8
X=X [X['quality']>=a]
X = X.drop(columns=['quality'])
#print(X)
#print(X.shape[0])


coverage_threshold = 1


thresholds = {}  #słownik

for col in X.columns:
    col_values = X[col].values  #numpy 
    unique_vals = np.unique(col_values)     #unikalne wartości w kolumnie
    
    best_coverage = 0   
    best_direction = None
    best_threshold = None
    
    for val in unique_vals:
        coverage = (X[col] >= val).mean()

        if coverage >= coverage_threshold and coverage > best_coverage:
            best_coverage = coverage
            best_threshold = val
            best_direction = '>='
    
        coverage = (X[col] <= val).mean()
        if coverage >= coverage_threshold and coverage > best_coverage:
            best_coverage = coverage
            best_threshold = val
            best_direction = '<='
    
    if best_direction is not None:
        thresholds[col] = (best_direction, best_threshold)



print(f"Znalezione progi dla większości przypadków quality >= {a}:\n")
for col, (direction, value) in thresholds.items():
    print(f"{col} {direction} {value:.4f}")


mask_all = pd.Series(True, index=X.index)
for col, (direction, value) in thresholds.items():
    if direction == '>=':
        mask_all &= X[col] >= value
    else:
        mask_all &= X[col] <= value


print(f"\nLiczba rekordów w X: {X.shape[0]}")
print(f"Liczba rekordów spełniających wszystkie warunki: {mask_all.sum()}")
print(f"Procent spełniających warunki: {100 * mask_all.mean():.2f}%")

























'''
for col in X_7.columns:
    if col != 'quality':
        sorted_values = X_7[col].sort_values().reset_index(drop=True)
        n = len(sorted_values)
        idx = int(n * 0.5) + 1  
        thresholds[col] = sorted_values.iloc[idx]


for col, threshold in thresholds.items():
    print(f"{col}: próg > {threshold:.4f}, dla wiekszosci")



    
for col in X_7.columns:
    col_values = X_7[col].values
    quantiles = np.linspace(0.01, 0.99, 99)
    best_direction = None
    best_threshold = None

    best_coverage = 0

    for q in quantiles:
        lower_thresh = np.quantile(col_values, q)
        upper_thresh = np.quantile(col_values, 1 - q)

        mask = X_7[col] >= lower_thresh
        coverage = mask.mean()
        if coverage >= coverage_threshold and coverage > best_coverage:
            best_direction = '>='
            best_threshold = lower_thresh
            best_coverage = coverage

        mask = X_7[col] <= upper_thresh
        coverage = mask.mean()
        if coverage >= coverage_threshold and coverage > best_coverage:
            best_direction = '<='
            best_threshold = upper_thresh
            best_coverage = coverage


    if best_direction and best_threshold is not None:
        thresholds[col] = (best_direction, best_threshold)
'''