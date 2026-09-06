import pandas as pd

df = pd.read_excel('converted_results.xlsx', sheet_name='Results')
print("Columns:", df.columns.tolist())

# Print non-empty entries for Alien
print("\nAlien data:")
alien_data = df[df.iloc[:, 0] == 'Alien']
print(alien_data)
