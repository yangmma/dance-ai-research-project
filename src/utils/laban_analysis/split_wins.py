import os
import numpy as np
from tqdm import tqdm
import json

MAX_SPLIT = 12

def main():
    input_dir = "./data/all_inputs_iui_original_music"
    output_dir = f"{input_dir}_roll"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name in tqdm(os.listdir(input_dir)):

        full_path = os.path.join(input_dir, name)
        with open(full_path) as f:
            result = json.loads(f.read())
            result = np.array(result)
        
        t, j = result.shape
        step = 56
        frame_size = 224
        frame_count = (t - frame_size) // step + 1
        results=[]
        for start_i in range(0, min(frame_count, MAX_SPLIT+1)):
            i = start_i * step
            # NOTE: add 1 since the end of slice is non-inclusive
            end_i = i + frame_size
            new = result[i: end_i, :] 
            results.append(new)

        for i, res in enumerate(results):
            full_out_path = os.path.join(output_dir, f"{name[:-9]}_{i}.json")
            with open(full_out_path, "w") as f:
                f.write(json.dumps(res.tolist()))


if __name__ == "__main__":
    main()