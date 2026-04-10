import streamlit as st
import pandas as pd
from src.pipeline import run_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title(" Hotel Clustering")

# Upload file
uploaded_file = st.file_uploader("Sube tu dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write(" Preview data:")
    st.dataframe(df.head())

    k = st.slider("Número de clusters", 2, 6, 4)

    results = run_pipeline(df, k=k)

    st.subheader(" Resumen clusters")
    st.write(results["summary"])

    # PCA para visualización
    from src.preprocessing import preprocess_hotels
    df_scaled, _ = preprocess_hotels(df)

    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)

    plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=results["labels"])
    plt.title("Clusters visualizados (PCA)")

    st.pyplot(plt)

    st.subheader(" Insights")
    st.write(results["insights"])