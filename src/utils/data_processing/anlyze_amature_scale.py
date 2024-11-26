import os
import json
import math
import urllib
import urllib.parse
import polars as pl
import argparse
from tqdm import tqdm
import numpy as np


JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

JOINT_MAP = {
    "left_wrist": "left_elbow",
    "left_elbow": "left_shoulder",
    "left_ankle": "left_knee",
    "left_knee": "left_hip",
}


def main(input_dir: str):
    fnames = os.listdir(input_dir)
    distances_all = {}
    for name in tqdm(fnames):
        path = os.path.join(input_dir, name)
        print(path)
        with open(path) as f:
            json_data = json.loads(f.read())["dance_array"]
        np_dance = np.array(json_data)
        N, JD = np_dance.shape
        D = 3
        J = JD // D
        assert J == 24, f"{path} does map to 24 joints"
        dance_data_frames = np_dance.reshape(N, J, D)
        for from_joint in JOINT_MAP:
            to_joint = JOINT_MAP[from_joint]
            from_i = JOINT_NAMES.index(from_joint)
            to_i = JOINT_NAMES.index(to_joint)

            distances = []
            for frame in dance_data_frames:
                from_joint_pos = frame[from_i]
                to_joint_pos = frame[to_i]

                direction = from_joint_pos - to_joint_pos
                distance = math.sqrt(pow(direction[0], 2) + pow(direction[1], 2) + pow(direction[2], 2))
                distances.append(distance)

            avg_key = f"{from_joint}_avg"
            if avg_key not in distances_all:
                distances_all[avg_key] = []

            std_key = f"{from_joint}_std"
            if std_key not in distances_all:
                distances_all[std_key] = []

            distances_all[avg_key].append(np.average(distances))
            distances_all[std_key].append(np.std(distances))

    df = pl.from_dict(distances_all)
    print(df)
    save_path = input_dir.replace("/", "_")
    print(save_path)
    df.write_csv(f"{save_path}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    args = parser.parse_args()
    main(args.input)