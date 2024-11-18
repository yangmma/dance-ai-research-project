import subprocess
import argparse
import os
import polars as pl

def main(input_dir: str, clustering_file: str, output_file: str):
    
    videos = os.listdir(input_dir)
    nums_list = []
    for video in videos:
        video_name_split = video.split("_")
        video_index = video_name_split[2]
        nums_list.append(int(video_index))
    
    nums_list.sort()
    with open('filelist_with_text.txt', 'w') as file:
        for num in nums_list:
            file_name = f'dance_data_{num}_corrected.mp4'
            path = os.path.join(input_dir, file_name)
            file.write(f"file {path}\n")

    subprocess.run([
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'filelist_with_text.txt', '-c', 'copy', output_file
    ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='./input')
    parser.add_argument("--cluster_file", default='./clustering_output')
    parser.add_argument("--output_file", default='./output.mp4')
    args = parser.parse_args()
    main(args.input_dir, args.cluster_file, args.output_file)