import pandas as pd

df = pd.read_csv("data/heart_disease.csv")
reference = df.sample(frac=0.7, random_state=1)
current = df.drop(reference.index)
reference.to_csv("data/reference.csv", index=False)
current.to_csv("data/current.csv", index=False)
print(f"reference={len(reference)} current={len(current)}")
