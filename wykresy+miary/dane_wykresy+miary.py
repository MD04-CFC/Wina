import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


X = pd.read_csv('dane\dataset_clean.csv', sep=';')
X.columns = X.columns.str.replace('"', '').str.strip()
y = X['quality']
X = X.drop(columns=["quality"])
df = pd.DataFrame(X)
print(df.head())

'''

def histogram(a):  
    plt.figure(figsize=(8, 5))
    plt.hist(df[f'{a}'], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {a}')
    plt.xlabel(f'{a}')
    plt.ylabel('Frequency')
    plt.grid(True)
    #plt.show() 
    plt.savefig(f'histogram_{a}.png', dpi=300, bbox_inches='tight')    


def boxplot(a):   
    plt.figure(figsize=(6, 5))
    plt.boxplot(df[f'{a}'].dropna(), vert=True)
    plt.title(f'Boxplot of {a}')
    plt.ylabel(f'{a}')
    plt.grid(True)
    #plt.show()  
    plt.savefig(f'boxplot_{a}.png', dpi=300, bbox_inches='tight')


def calculate_basic_statistical_measures(a):
    mean = df[f"{a}"].mean()
    median = df[f"{a}"].median()
    mode = df[f"{a}"].mode().iloc[0]

    # Dispersion
    std_dev = df[f"{a}"].std()
    variance = df[f"{a}"].var()
    range_val = df[f"{a}"].max() - df[f"{a}"].min()
    iqr = df[f"{a}"].quantile(0.75) - df[f"{a}"].quantile(0.25)

    # Position
    min_val = df[f"{a}"].min()
    max_val = df[f"{a}"].max()

    print(f"Mean: {mean}, Median: {median}, Mode: {mode}, Variance: {variance}")
    print(f"Minimalna wartosc: {min_val}, maxymalna wartosc: {max_val}")
    print(f"Standard Deviation: {std_dev}, IQR: {iqr}, Range: {range_val}")

    file = open(f"statystyki_{a}.txt", "w")
    file.write(f"Mean: {mean}, Median: {median}, Mode: {mode}, Variance: {variance}\n")
    file.write(f"Minimalna wartosc: {min_val}, maxymalna wartosc: {max_val}\n") 
    file.write(f"Standard Deviation: {std_dev}, IQR: {iqr}, Range: {range_val}\n")
    file.close()



features = df.columns
for feature in features:
    histogram(feature)
    boxplot(feature)
    calculate_basic_statistical_measures(feature)
    '''