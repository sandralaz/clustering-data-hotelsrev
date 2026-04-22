from openai import OpenAI

def describe_clusters(df_original, labels):
    df = df_original.copy()
    df["cluster"] = labels

    summaries = []

    for cluster in sorted(df["cluster"].unique()):
        cluster_data = df[df["cluster"] == cluster]

        summary = {
            "cluster": int(cluster),
            "avg_hotel_class": round(cluster_data["hotel_class"].mean(), 2) if "hotel_class" in df.columns else None,
            "size": int(len(cluster_data))
        }

        summaries.append(summary)

    return summaries


def generate_segments(cluster_summary):
    client = OpenAI()   

    prompt = f"""
    Eres experto en hoteles.

    {cluster_summary}

    Describe los clusters.
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content