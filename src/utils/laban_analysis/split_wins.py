import os
import numpy as np
from tqdm import tqdm
import json


def main():
    input_dir = "../convert_video/post_processed_out_split"
    for name in tqdm(os.listdir(input_dir)):
        if name.split("-")[2][0] != "3":
            continue
        elif name.split("-")[3][0] != "0":
            continue

        full_path = os.path.join(input_dir, name)
        with open(full_path) as f:
            result = json.loads(f.read())
            result = np.array(result)
        
        t, j = result.shape
        step = 56
        frame_size = 224
        frame_count = t - frame_size // step
        results=[]
        for start_i in range(0, frame_count, step):
            # NOTE: add 1 since the end of slice is non-inclusive
            end_i = start_i + frame_size + 1
            results.append(result[start_i: end_i, :])


        for i, res in enumerate(results):
            full_out_path = os.path.join("./post_processed_out_split_roll", f"{name}-{i}.json")
            with open(full_out_path, "w") as f:
                f.write(json.dumps(res.tolist()))


if __name__ == "__main__":
    main()