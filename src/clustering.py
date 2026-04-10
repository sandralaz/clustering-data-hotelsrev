from sklearn.cluster import KMeans

def cluster_hotels(df_scaled, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(df_scaled)
    return labels



def describe_clusters(df_original, labels):
    df = df_original.copy()
    df["cluster"] = labels

    cluster_summary = []
    for cluster in sorted(df["cluster"].unique()):
        sample = df[df["cluster"] == cluster].head(3).to_dict(orient="records")
        cluster_summary.append(f"Cluster {cluster} ejemplo hoteles: {sample}")

    insights = "\n".join(cluster_summary)
    return insights