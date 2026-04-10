def profile_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels

    # Promedios por cluster
    profile = df.groupby("cluster").mean().round(2)

    # Tamaño de cada cluster
    size = df["cluster"].value_counts()

    return df, profile, size