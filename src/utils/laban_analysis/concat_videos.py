import subprocess
import argparse
from multiprocessing import Pool
from functools import partial
import os
import polars as pl

def add_text_to_video(iter):
    input_video, output_video, text = iter
    subprocess.run([
        'ffmpeg', '-i', input_video, '-vf', f"drawtext=text='{text}':x=(w-tw)/2:y=h-th-50:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5",
        '-t', '20',  # Duration for the text overlay
        '-c:a', 'copy', output_video
    ], check=True)


def main(input_dir: str, clustering_file: str, output_file: str):
    df = pl.read_csv(clustering_file)
    df = df.select(
        pl.col("name"),
        pl.col("best"),
        pl.col("clus"),
        pl.col("music"),
    )
    df = df.sort(by=["best"], descending=False)
    names = df["name"]
    clusters = df["best"]
    cluses = df["clus"]
    musics = df["music"]

    temp_files = []
    zip_list = []
    clus, ind = 0, 0
    music_dict = {
        "mBR0": "Break",
        "mPO0": "Pop",
        "mLO0": "Lock",
        "mMH0": "Middle Hip-hop",
        "mLH0": "LA style Hip-hop",
        "mHO0": "House",
        "mWA0": "Waack",
        "mKR0": "Krump",
        "mJS0": "Street Jazz",
        "mJB0": "Ballet Jazz",
    }
    for name, cluster, cluster_name, music in zip(names, clusters, cluses, musics):
        path = os.path.join(input_dir, f'{name}.mp4')
        if not os.path.exists(path):
            continue

        if clus != cluster:
            clus = cluster
            ind = 0
        ind += 1
        temp_file = f'{cluster}_video_{ind}.mp4'
        text = f'{cluster} correlated sequence {ind}\ncluster {cluster_name} and {music_dict[music]} genre'
        zip_list.append((path, temp_file, text))
        # add_text_to_video(path, temp_file, text)
        temp_files.append(temp_file)
    pool = Pool(10)
    partial_func = partial(add_text_to_video)
    pool.map(partial_func, zip_list)
    pool.close()
    pool.join()


    # with open('filelist_with_text.txt', 'w') as file:
    #     for temp_file in temp_files:
    #         file.write(f"file '{temp_file}'\n")

    # subprocess.run([
    #     'ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'filelist_with_text.txt', '-c', 'copy', output_file
    # ], check=True)

    os.remove('filelist_with_text.txt')
    # for temp_file in temp_files:
    #     os.remove(temp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='./input')
    parser.add_argument("--cluster_file", default='./clustering_output')
    parser.add_argument("--output_file", default='./output.mp4')
    args = parser.parse_args()
    main(args.input_dir, args.cluster_file, args.output_file)