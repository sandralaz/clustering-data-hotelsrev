from src.preprocessing import preprocess_hotels
from src.clustering import cluster_hotels
from src.ai_segmentation import describe_clusters

def run_pipeline(df, k=4):
      # 1. Preprocessing
    df_scaled, df_original = preprocess_hotels(df)

    # 2. Clustering
    labels = cluster_hotels(df_scaled, k=k)

    # 3. Resumen
    summary = describe_clusters(df_original, labels)

    # 4. IA
    insights = "IA desactivada"

    return {
        "labels": labels,
        "summary": summary,
        "insights": insights
    }