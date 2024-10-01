import generate_features
from tqdm import tqdm
import polars as pl
import os


def main(input_dir: str):
    fnames = os.listdir(input_dir)
    df = pl.DataFrame()
    out_path = "./generated_complete_cleaned"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for name in tqdm(fnames):
        full_path = os.path.join(input_dir, name)
        df = pl.read_parquet(full_path)
        df = df.with_columns(
            name_orig = pl.col("name").str.split(by=".").list.get(0),
            music = pl.col("name").str.split(by="-").list.get(1).str.split(".").list.get(0),
            win = pl.col("name").str.split(by="-").list.get(2).str.split("").list.get(0),
        )
        df = df.drop(
            pl.col("name"),
        ).with_columns(
            name = pl.col("name_orig"),
        )
    
        full_out_path = os.path.join(out_path, name)
        df.write_parquet(full_out_path)

if __name__ == "__main__":
    main("./generated_complete")