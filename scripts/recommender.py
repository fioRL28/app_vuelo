import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def recommend_destinations(user_type, budget):
    df = pd.read_csv("processed/cleaned_data.csv")
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[["average_cost", "rating"]])
    df["similarity"] = cosine_similarity([[budget, 5]], df_scaled).flatten()
    df_filtered = df[df["category"].str.contains(user_type, case=False, na=False)]
    return df_filtered.sort_values("similarity", ascending=False).head(5)[["destination", "country", "average_cost", "rating"]]
