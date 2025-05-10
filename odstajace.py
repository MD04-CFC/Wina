
import numpy as np
import pandas as pd



def find_outliers(data):
 
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    # quartiles
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    
    # IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    

    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return cleaned_data



def remove_outliers(df, features):
    for column in features:
        cleaned_column = find_outliers(df[column])
        df = df[df[column].isin(cleaned_column)]
    return df



'''
from ucimlrepo import fetch_ucirepo 
wine_quality = fetch_ucirepo(id=186) 
features = wine_quality.data.features
print(features)
'''
#features = [ "fixed_acidity", 'volatile acidity', 'citric acid', 'residualsugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']




file_path = 'dane.csv'  
df = pd.read_csv(file_path)
features = df.select_dtypes(include=[np.number]).columns.tolist()
file_path2 = 'dataset_clean.csv'  
cleaned_df = remove_outliers(df, features)
cleaned_df.to_csv(file_path2, index=False)



