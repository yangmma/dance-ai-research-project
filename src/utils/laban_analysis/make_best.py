import os
import polars as pl
import subprocess

def generate_video(list_name: str, inputs_dir: str):
    df = pl.read_csv(f"{list_name}.csv")

    for row in df.iter_rows(named=True):
        name = row["name"]
        music = row["music"]
        videos_dir = "../convert_video/post_processed_convert-out"

        with open(".temp_list.txt", "w") as f:
            input_video_path = os.path.join(inputs_dir, f"{name}.mp4")
            f.write(f"file {input_video_path}\n")
            output_video_path = os.path.join(videos_dir, f"{name}.json-{music}.wav-3win.json-0.mp4")
            f.write(f"file {output_video_path}")

        output_dir = list_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_name = f"{name}_{music}.mp4"
        output_path = os.path.join(output_dir, output_name)
        subprocess.run([
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', '.temp_list.txt', '-c', 'copy', '-y', output_path
        ], check=True)

def main(best_dance_path: str, worst_dance_path: str):
    best_list = os.listdir(best_dance_path)
    for list in best_list:
        list_name = list.split(".")[0]
        list_path = os.path.join(best_dance_path, list_name)
        generate_video(list_path, "input-out")

    worst_list = os.listdir(worst_dance_path)
    for list in worst_list:
        list_name = list.split(".")[0]
        list_path = os.path.join(worst_dance_path, list_name)
        generate_video(list_path, "input-out")



if __name__ == "__main__":
    main("best_dance_music", "worst_dance_music")