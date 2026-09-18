import pandas as pd
import numpy as np

df = pd.read_csv('wandb_runs_classification.csv')
print("Qbert Runs from classify:")
q_runs = df[df['Game'] == 'Qbert']
for _, r in q_runs.iterrows():
    if '10' in str(r['Run Name']):
        print(f"Name: {r['Run Name']}, Ret: {r['Retrieval Enable']}, Target: {r['Retrieval Target']}, Seed: {r['Seed']}, RetAvg: {r['Eval Return']}, Created: {r['Created At']}")
