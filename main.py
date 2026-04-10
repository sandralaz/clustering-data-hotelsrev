import pandas as pd
from src.pipeline import run_pipeline

def main():
    df = pd.read_csv("data/raw/data.csv")

    print("Columnas:", df.columns)

    results = run_pipeline(df, k=4)

    print("\n📊 RESUMEN CLUSTERS:")
    print(results["summary"])

    print("\n🤖 INSIGHTS IA:")
    print(results["insights"])

if __name__ == "__main__":
    main()