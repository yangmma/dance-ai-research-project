import polars as pl
import argparse
import os
import shutil


def main(input_file: str, input_video_dir: str, output_video_dir: str, out_dir: str):
    df = pl.read_csv(input_file)
    input_videos = []
    output_videos = []
    for row in df.rows(named=True):
        input_name = f"{row['name']}.mp4"
        output_name = f"{row['name']}.json-{row['music']}.wav.mp4"

        input_path = os.path.join(input_video_dir, input_name)
        output_path = os.path.join(output_video_dir, output_name)

        input_videos.append(input_path)
        output_videos.append(output_path)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for vid in input_videos + output_videos:
        shutil.copy(vid, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--out_dir")
    parser.add_argument("--vidir_in")
    parser.add_argument("--vidir_out")
    args = parser.parse_args()
    main(args.input, args.vidir_in, args.vidir_out, args.out_dir)