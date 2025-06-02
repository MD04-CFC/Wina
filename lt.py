import pandas as pd

# Load the data
df = pd.read_csv('A:\\disk_S\\homework_sggw\\projekt_mad2\\Wina\\dane\\artificial_wine_observations.csv', delimiter=',')

# Convert to LaTeX table format
for _, row in df.iterrows():
    latex_row = ' & '.join([f"{val:.1f}" if isinstance(val, float) else str(val) for val in row.values]) + ' \\\\'
    print(latex_row)
