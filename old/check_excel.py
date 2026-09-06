import pandas as pd
import numpy as np

df = pd.read_excel('converted_results.xlsx', sheet_name='Results')
print("Qbert Rows:")
qbert_df = df[df.iloc[:, 0] == 'Qbert']
print(qbert_df.to_string())
