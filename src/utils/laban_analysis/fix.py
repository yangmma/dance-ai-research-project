import polars as pl 


def main():
    df = pl.read_parquet("./generated_all/generated_features_complete_192.parquet")
    df = df.with_columns(
        name_orig = pl.col("name").str.split(".").list.get(0),
        music = pl.col("name").str.split("-").list.get(1).str.split(".").list.get(0),
        num = pl.col("name").str.split("-").list.get(2).str.split("").list.get(0),
        win = pl.col("name").str.split("-").list.get(3).str.split("").list.get(0),
        win_c = pl.col("name").str.split("-").list.get(4).str.split("").list.get(0),
    ).drop(
        "name"
    ).with_columns(
        name = pl.col("name_orig")
    )
    df.write_parquet("./generated_all/generated_features_complete_192.parquet")
    

if __name__ == "__main__":
    main()