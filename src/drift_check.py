df = pd.read_csv("data/heart_disease.csv")
df.sample(frac=0.7, random_state=1).to_csv("data/reference.csv", index=False)
df.drop(df.sample(frac=0.7, random_state=1).index).to_csv("data/current.csv", index=False)
