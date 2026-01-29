def get_top_locations(df, n=5):
    df["score"] = (
        df["foot_traffic"] * 0.4 +
        df["income"] * 0.3 -
        df["competition"] * 0.3
    )
    return df.sort_values("score", ascending=False).head(n)
